from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

GROUPS = ["afr", "ami", "amr", "asj", "eas", "fin", "mid", "nfe", "remaining", "sas"]

INT_KEYS = ["AC", "AN", "nhomalt", "AC_grpmax", "AN_grpmax", "n_alt_alleles"]
FLOAT_KEYS = ["AF", "AF_grpmax", "fafmax_faf95_max", "inbreeding_coeff"]
STR_KEYS = ["grpmax", "fafmax_faf95_max_gen_anc", "allele_type", "variant_type"]
GROUP_AF_KEYS = [f"AF_{g}" for g in GROUPS]
GROUP_AN_KEYS = [f"AN_{g}" for g in GROUPS]
FLAG_KEYS = ["lcr", "segdup", "non_par"]

VALUE_KEYS = INT_KEYS + FLOAT_KEYS + STR_KEYS + GROUP_AF_KEYS + GROUP_AN_KEYS

MIN_GROUP_AN = 2000

SCHEMA = pa.schema(
    [
        ("chrom", pa.dictionary(pa.int8(), pa.string())),
        ("pos", pa.int32()),
        ("ref", pa.string()),
        ("alt", pa.string()),
        ("filter", pa.dictionary(pa.int8(), pa.string())),
        ("allele_type", pa.dictionary(pa.int8(), pa.string())),
        ("variant_type", pa.dictionary(pa.int8(), pa.string())),
        ("n_alt_alleles", pa.int16()),
        ("AC", pa.int32()),
        ("AN", pa.int32()),
        ("AF", pa.float32()),
        ("nhomalt", pa.int32()),
        ("grpmax", pa.dictionary(pa.int8(), pa.string())),
        ("AF_grpmax", pa.float32()),
        ("AC_grpmax", pa.int32()),
        ("AN_grpmax", pa.int32()),
        ("faf95_max", pa.float32()),
        ("faf95_max_gen_anc", pa.dictionary(pa.int8(), pa.string())),
        ("af_max_any_grp", pa.float32()),
        ("inbreeding_coeff", pa.float32()),
        ("lcr", pa.bool_()),
        ("segdup", pa.bool_()),
        ("non_par", pa.bool_()),
    ]
    + [(f"AF_{g}", pa.float32()) for g in GROUPS]
)

CHROM_ORDER = {f"chr{c}": i for i, c in enumerate([str(n) for n in range(1, 23)] + ["X", "Y"])}


def _open_bgz(path: str):
    try:
        from isal import igzip_threaded

        return igzip_threaded.open(path, "rb", threads=1, block_size=4 * 1024 * 1024)
    except Exception:
        pass
    try:
        from isal import igzip

        return igzip.open(path, "rb")
    except Exception:
        import gzip

        return gzip.open(path, "rb")


def chrom_from_path(path: str) -> str:
    m = re.search(r"\.(chr[0-9XYM]+)\.", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def find_vcfs(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for pat in ("*.vcf.bgz", "*.vcf.gz"):
                files += glob.glob(os.path.join(item, pat))
        else:
            files += glob.glob(item) or [item]
    files = sorted(set(files), key=lambda p: CHROM_ORDER.get(chrom_from_path(p), 99))
    return files


def _bytes_col(values):
    arr = pa.array(values, type=pa.binary()).cast(pa.string())
    return pc.if_else(pc.equal(arr, "."), pa.scalar(None, pa.string()), arr)


def _num_col(values, typ):
    arr = _bytes_col(values)
    if pa.types.is_floating(typ):
        out = arr.cast(pa.float64())
        out = pc.if_else(pc.is_nan(out), pa.scalar(None, pa.float64()), out)
        return out.cast(typ)
    return arr.cast(typ)


def _dict_col(values, typ):
    return _bytes_col(values).dictionary_encode().cast(typ)


def _build_table(raw, flags):
    import numpy as np

    out = {
        "chrom": _dict_col(raw["chrom"], SCHEMA.field("chrom").type),
        "pos": _num_col(raw["pos"], pa.int32()),
        "ref": _bytes_col(raw["ref"]),
        "alt": _bytes_col(raw["alt"]),
        "filter": _dict_col(raw["filter"], SCHEMA.field("filter").type),
        "allele_type": _dict_col(raw["allele_type"], SCHEMA.field("allele_type").type),
        "variant_type": _dict_col(raw["variant_type"], SCHEMA.field("variant_type").type),
        "n_alt_alleles": _num_col(raw["n_alt_alleles"], pa.int16()),
        "AC": _num_col(raw["AC"], pa.int32()),
        "AN": _num_col(raw["AN"], pa.int32()),
        "AF": _num_col(raw["AF"], pa.float32()),
        "nhomalt": _num_col(raw["nhomalt"], pa.int32()),
        "grpmax": _dict_col(raw["grpmax"], SCHEMA.field("grpmax").type),
        "AF_grpmax": _num_col(raw["AF_grpmax"], pa.float32()),
        "AC_grpmax": _num_col(raw["AC_grpmax"], pa.int32()),
        "AN_grpmax": _num_col(raw["AN_grpmax"], pa.int32()),
        "faf95_max": _num_col(raw["fafmax_faf95_max"], pa.float32()),
        "faf95_max_gen_anc": _dict_col(raw["fafmax_faf95_max_gen_anc"], SCHEMA.field("faf95_max_gen_anc").type),
        "inbreeding_coeff": _num_col(raw["inbreeding_coeff"], pa.float32()),
    }
    for k in FLAG_KEYS:
        out[k] = pa.array(flags[k], type=pa.bool_())
    best = None
    for g in GROUPS:
        af = _num_col(raw[f"AF_{g}"], pa.float32())
        an = _num_col(raw[f"AN_{g}"], pa.int32())
        out[f"AF_{g}"] = af
        af_np = af.to_numpy(zero_copy_only=False).astype("float64")
        an_np = an.fill_null(0).to_numpy(zero_copy_only=False)
        af_np = np.where(an_np >= MIN_GROUP_AN, af_np, np.nan)
        best = af_np if best is None else np.fmax(best, af_np)
    out["af_max_any_grp"] = pa.array(best, type=pa.float32(), from_pandas=True)
    return pa.table([out[f.name] for f in SCHEMA], schema=SCHEMA)


def process_file(path: str, out_dir: str, batch_rows: int = 1_000_000, limit_lines: int = 0) -> dict:
    chrom = chrom_from_path(path)
    out_parquet = os.path.join(out_dir, f"{chrom}.parquet")
    tmp_parquet = out_parquet + ".partial"
    progress_path = os.path.join(out_dir, f"{chrom}.progress.json")
    done_path = os.path.join(out_dir, f"{chrom}.done.json")
    t0 = time.time()
    total_bytes = os.path.getsize(path)

    fh = _open_bgz(path)

    info_ids: list[str] = []
    filter_ids: list[str] = []
    line = fh.readline()
    while line.startswith(b"##"):
        if line.startswith(b"##INFO=<ID="):
            info_ids.append(line[11:].split(b",", 1)[0].decode())
        elif line.startswith(b"##FILTER=<ID="):
            filter_ids.append(line[13:].split(b",", 1)[0].decode())
        line = fh.readline()
    if not line.startswith(b"#CHROM"):
        raise RuntimeError(f"{path}: no #CHROM header line found")

    header_pos = {k: i for i, k in enumerate(info_ids)}
    missing_in_header = [k for k in VALUE_KEYS + FLAG_KEYS if k not in header_pos]
    ordered_keys = sorted([k for k in VALUE_KEYS if k in header_pos], key=header_pos.get)
    needles = [(k, (";" + k + "=").encode()) for k in ordered_keys]
    flag_needles = [(k, (";" + k + ";").encode()) for k in FLAG_KEYS]

    raw_keys = ["chrom", "pos", "ref", "alt", "filter"] + VALUE_KEYS
    raw = {k: [] for k in raw_keys}
    flags = {k: [] for k in FLAG_KEYS}
    writer = pq.ParquetWriter(tmp_parquet, SCHEMA, compression="zstd", compression_level=3)

    def flush():
        if raw["pos"]:
            writer.write_table(_build_table(raw, flags))
            for v in list(raw.values()) + list(flags.values()):
                v.clear()

    n = 0
    filter_counts: dict[str, int] = {}
    vep_needle = b";vep="
    last_progress = time.time()
    raw_items = [(k, raw[k]) for k in ordered_keys]
    missing_lists = [raw[k] for k in VALUE_KEYS if k not in header_pos]
    flag_items = [(nd, flags[k]) for k, nd in flag_needles]
    r_chrom, r_pos, r_ref, r_alt, r_filter = (raw[k] for k in ["chrom", "pos", "ref", "alt", "filter"])

    for line in fh:
        parts = line.split(b"\t", 8)
        if len(parts) < 8:
            continue
        info = b";" + parts[7].rstrip(b"\r\n") + b";"
        vep_at = info.find(vep_needle)
        bound = vep_at if vep_at >= 0 else len(info)

        cursor = 0
        for (key, nd), (_, lst) in zip(needles, raw_items):
            i = info.find(nd, cursor, bound)
            if i < 0:
                i = info.find(nd, 0, bound)
                if i < 0:
                    i = info.find(nd, bound)
                    if i < 0:
                        lst.append(None)
                        continue
            s = i + len(nd)
            e = info.find(b";", s)
            lst.append(info[s:e])
            cursor = e
        for lst in missing_lists:
            lst.append(None)
        for nd, lst in flag_items:
            lst.append(info.find(nd) >= 0)

        r_chrom.append(parts[0])
        r_pos.append(parts[1])
        r_ref.append(parts[3])
        r_alt.append(parts[4])
        f = parts[6]
        r_filter.append(f)
        filter_counts[f] = filter_counts.get(f, 0) + 1

        n += 1
        if len(r_pos) >= batch_rows:
            flush()
        if n % 50_000 == 0 and time.time() - last_progress > 20:
            last_progress = time.time()
            _write_json(progress_path, _progress(chrom, n, t0, fh, total_bytes))
        if limit_lines and n >= limit_lines:
            break

    flush()
    writer.close()
    fh.close()
    os.replace(tmp_parquet, out_parquet)

    result = {
        "chrom": chrom,
        "source": os.path.basename(path),
        "variants": n,
        "seconds": round(time.time() - t0, 1),
        "filter_counts": {k.decode(): v for k, v in filter_counts.items()},
        "header_filters": filter_ids,
        "missing_info_keys": missing_in_header,
        "limited": bool(limit_lines and n >= limit_lines),
    }
    _write_json(done_path, result)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    return result


def _progress(chrom, n, t0, fh, total_bytes):
    frac = None
    try:
        raw = getattr(fh, "raw", None) or getattr(fh, "fileobj", None)
        if raw is not None and hasattr(raw, "tell"):
            frac = min(raw.tell() / total_bytes, 1.0)
    except Exception:
        pass
    return {"chrom": chrom, "variants": n, "elapsed_s": round(time.time() - t0), "approx_fraction": frac}


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run(inputs, out_dir, workers=4, chroms=None, batch_rows=1_000_000, limit_lines=0, overwrite=False, log=print):
    os.makedirs(out_dir, exist_ok=True)
    files = find_vcfs(inputs)
    if chroms:
        wanted = {c if c.startswith("chr") else f"chr{c}" for c in chroms}
        files = [f for f in files if chrom_from_path(f) in wanted]
    if not files:
        raise SystemExit(f"No .vcf.bgz/.vcf.gz files found in {inputs}")

    todo = []
    for f in files:
        c = chrom_from_path(f)
        if not overwrite and os.path.exists(os.path.join(out_dir, f"{c}.done.json")):
            log(f"[skip] {c} already done")
            continue
        todo.append(f)
    log(f"{len(todo)} chromosome file(s) to process with {workers} worker(s) -> {os.path.abspath(out_dir)}")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_file, f, out_dir, batch_rows, limit_lines): f for f in todo}
        pending = set(futs)
        last_log = time.time()
        while pending:
            done_now = [f for f in pending if f.done()]
            for fut in done_now:
                pending.discard(fut)
                try:
                    r = fut.result()
                    results.append(r)
                    log(f"[done] {r['chrom']}: {r['variants']:,} variants in {r['seconds']/60:.1f} min"
                        + (f"  (missing INFO keys: {r['missing_info_keys']})" if r["missing_info_keys"] else ""))
                except Exception as e:
                    log(f"[FAIL] {chrom_from_path(futs[fut])}: {e!r}")
            if pending and time.time() - last_log > 60:
                last_log = time.time()
                status = []
                for p in glob.glob(os.path.join(out_dir, "*.progress.json")):
                    try:
                        with open(p) as fh:
                            d = json.load(fh)
                        fr = f" ~{d['approx_fraction']*100:.0f}%" if d.get("approx_fraction") else ""
                        status.append(f"{d['chrom']} {d['variants']/1e6:.1f}M{fr}")
                    except Exception:
                        pass
                if status:
                    log(f"[{(time.time()-t0)/60:6.1f} min] running: " + ", ".join(sorted(status)))
            time.sleep(1)
    log(f"All finished in {(time.time()-t0)/60:.1f} min")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Folder(s) with gnomAD per-chromosome .vcf.bgz files, or file paths/globs")
    ap.add_argument("--out", default="gnomad_slim", help="Output folder for parquet files (default: gnomad_slim)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="Chromosomes processed in parallel (default: min(4, cores-1)). Disk speed is usually the limit.")
    ap.add_argument("--chroms", default="", help="Comma-separated subset, e.g. chr21,chr22")
    ap.add_argument("--batch-rows", type=int, default=1_000_000)
    ap.add_argument("--limit-lines", type=int, default=0, help="Stop after N variants per chromosome (testing)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    chroms = [c.strip() for c in a.chroms.split(",") if c.strip()] or None
    run(a.inputs, a.out, a.workers, chroms, a.batch_rows, a.limit_lines, a.overwrite, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
