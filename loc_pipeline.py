from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMP = str.maketrans("ACGTN", "TGCAN")
EXON_BIN = 100_000

MOTIFS = {
    "PUM": {"regex": r"TGTA[ACT]ATA", "core": 1, "to": "C"},
    "IGF2BP_GGAC": {"regex": r"[AG]GGAC", "core": 2, "to": "C"},
    "A2RE": {"regex": r"GCCAAGGAGCC", "core": 5, "to": "C"},
    "QKI": {"regex": r"ACTAA[CT]", "core": 3, "to": "C"},
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def load_adapter():
    spec = importlib.util.spec_from_file_location("model_adapter", HERE / "model_adapter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def window_start(tx_len, max_len, crop):
    if not max_len or tx_len <= max_len:
        return 1
    return tx_len - max_len + 1 if crop == "3p" else 1


def window_end(tx_len, max_len, crop):
    if not max_len or tx_len <= max_len:
        return tx_len
    return tx_len if crop == "3p" else max_len


def fetch(url, dest: Path, mirror=None):
    if mirror:
        url = url.replace("https://ftp.ebi.ac.uk", mirror)
    if dest.exists() and dest.stat().st_size:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "loc-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            with gzip.open(tmp, "rb") as g:
                while g.read(1 << 22):
                    pass
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed: {url} ({last!r})")


def step_transcripts(gtex_out: Path, out: Path, mirror=None):
    pq = out / "transcripts.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    gtf = sorted((gtex_out / "raw").glob("gencode.v*.basic.annotation.gtf.gz"))
    if not gtf:
        raise SystemExit("gtex_out/raw has no GENCODE GTF - run run_gtex.ps1 first")
    release = gtf[0].name.split(".")[1][1:]
    url = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/gencode.v{release}.transcripts.fa.gz"
    fa = fetch(url, gtex_out / "raw" / f"gencode.v{release}.transcripts.fa.gz", mirror)
    canon = pd.read_parquet(gtex_out / "gencode_canonical_transcripts.parquet")
    keep = set(canon["transcript_id"])
    log(f"Reading canonical transcript sequences from {fa.name}")
    recs, cur, buf = [], None, []

    def flush():
        if cur is not None:
            h = cur
            tid = h[0].split(".")[0]
            if tid in keep:
                d = {"transcript_id": tid, "seq": "".join(buf).upper(), "utr5_end": None, "cds_start": None,
                     "cds_end": None, "utr3_start": None}
                for f in h[7:]:
                    if f.startswith("CDS:"):
                        a, b = f[4:].split("-"); d["cds_start"], d["cds_end"] = int(a), int(b)
                    elif f.startswith("UTR3:"):
                        d["utr3_start"] = int(f[5:].split("-")[0])
                    elif f.startswith("UTR5:"):
                        d["utr5_end"] = int(f[5:].split("-")[1])
                recs.append(d)

    with gzip.open(fa, "rt") as f:
        for line in f:
            if line.startswith(">"):
                flush()
                cur, buf = line[1:].strip().split("|"), []
            else:
                buf.append(line.strip())
        flush()
    tx = pd.DataFrame(recs).merge(canon[["transcript_id", "gene_name", "gene_type", "strand", "chrom"]], on="transcript_id")
    tx["tx_len"] = tx["seq"].str.len()
    ex = pd.read_parquet(gtex_out / "gencode_canonical_exons.parquet")
    gtf_len = (ex["end"] - ex["start"] + 1).groupby(ex["transcript_id"]).sum()
    tx["gtf_len"] = tx["transcript_id"].map(gtf_len)
    bad = (tx["gtf_len"] != tx["tx_len"]).sum()
    log(f"  {len(tx):,} canonical transcripts with sequence; length mismatches vs GTF: {bad}")
    tx.to_parquet(pq, index=False)
    return tx


def step_gnomad_exonic(gtex_out: Path, gnomad_dir: Path, out: Path):
    dest = out / "gnomad_exonic"
    dest.mkdir(exist_ok=True)
    done = sorted(p.name.split(".")[0] for p in gnomad_dir.glob("*.done.json")) if gnomad_dir.exists() else []
    if not done:
        log("No finished gnomAD chromosomes found - gnomAD sets will be empty for now")
        return
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE ex AS
        SELECT e.*, t.gene_name, t.gene_type,
               coalesce(sum(e."end" - e.start + 1) OVER (PARTITION BY e.transcript_id ORDER BY e.exon_number
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS tx_offset
        FROM '{(gtex_out / 'gencode_canonical_exons.parquet').as_posix()}' e
        JOIN '{(gtex_out / 'gencode_canonical_transcripts.parquet').as_posix()}' t USING (transcript_id)
    """)
    con.execute(f"""CREATE TABLE cds AS SELECT transcript_id, min(start) cmin, max("end") cmax
                    FROM '{(gtex_out / 'gencode_canonical_cds.parquet').as_posix()}' GROUP BY 1""")
    con.execute("""CREATE TABLE txlen AS SELECT transcript_id, sum("end" - start + 1) tx_len FROM ex GROUP BY 1""")
    con.execute(f"""CREATE TABLE exb AS SELECT *, unnest(range(start // {EXON_BIN}, "end" // {EXON_BIN} + 1)) AS bin FROM ex""")
    for c in done:
        target = dest / f"{c}.parquet"
        if target.exists():
            continue
        src = (gnomad_dir / f"{c}.parquet").as_posix()
        t0 = time.time()
        con.execute(f"""
            COPY (
              WITH g AS (
                SELECT chrom, pos, ref, alt, AC, AN, AF, AF_grpmax, faf95_max,
                       greatest(coalesce(AF,0), coalesce(AF_grpmax,0), coalesce(af_max_any_grp,0)) AS f_max,
                       pos // {EXON_BIN} AS bin
                FROM '{src}'
                WHERE filter = 'PASS' AND NOT lcr AND NOT segdup
                  AND AN >= 0.5 * (SELECT max(AN) FROM '{src}')
              )
              SELECT g.chrom || '_' || g.pos || '_' || g.ref || '_' || g.alt AS variant,
                     g.chrom, g.pos, g.ref, g.alt, g.AC, g.AF, g.AF_grpmax, g.faf95_max, g.f_max,
                     CASE WHEN greatest(coalesce(g.AF,0), coalesce(g.faf95_max,0)) >= 0.01 THEN 'common'
                          WHEN g.f_max < 0.01 THEN 'rare' ELSE 'in-between' END AS gnomad_class,
                     length(g.ref) = 1 AND length(g.alt) = 1 AS is_snv,
                     e.transcript_id, e.gene_name, e.gene_type, e.strand,
                     CASE WHEN e.strand = '+' THEN e.tx_offset + (g.pos - e.start + 1)
                          ELSE e.tx_offset + (e."end" - g.pos + 1) END AS tx_pos,
                     l.tx_len,
                     CASE WHEN s.transcript_id IS NULL THEN 'ncRNA_exon'
                          WHEN g.pos BETWEEN s.cmin AND s.cmax THEN 'CDS'
                          WHEN (e.strand = '+' AND g.pos < s.cmin) OR (e.strand = '-' AND g.pos > s.cmax) THEN '5UTR'
                          ELSE '3UTR' END AS region
              FROM g JOIN exb e ON e.chrom = g.chrom AND e.bin = g.bin AND g.pos BETWEEN e.start AND e."end"
              JOIN txlen l USING (transcript_id) LEFT JOIN cds s USING (transcript_id)
            ) TO '{target.as_posix()}' (FORMAT parquet)
        """)
        log(f"  gnomAD {c}: exonic variants annotated in {time.time()-t0:.0f}s")
    con.close()


def scan_motifs(tx: pd.DataFrame, out: Path) -> pd.DataFrame:
    pq = out / "motif_instances.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    rows = []
    for r in tx.itertuples():
        coding = pd.notna(r.cds_start)
        if coding and pd.isna(r.utr3_start):
            continue
        start = int(r.utr3_start) if coding else 1
        region_seq = r.seq[start - 1:]
        for name, m in MOTIFS.items():
            for hit in re.finditer(f"(?=({m['regex']}))", region_seq):
                s = start + hit.start()
                core = s + m["core"]
                ref = r.seq[core - 1]
                if ref == m["to"]:
                    continue
                rows.append((r.transcript_id, name, s, s + len(hit.group(1)) - 1, core, ref, m["to"],
                             "3UTR" if coding else "ncRNA_exon"))
    mi = pd.DataFrame(rows, columns=["transcript_id", "motif", "motif_start", "motif_end", "core_pos", "core_ref", "core_alt", "region"])
    mi.to_parquet(pq, index=False)
    log(f"  motif instances: " + ", ".join(f"{k} {v:,}" for k, v in mi["motif"].value_counts().items()))
    return mi


def _dist_bin(d):
    return pd.cut(d, [-1, 250, 500, 1000, 2000, 4000, 8000, 10**9], labels=False)


def step_sets(gtex_out: Path, out: Path, tx: pd.DataFrame, n_rare: int, n_common: int, n_per_motif: int,
              neg_per_pos: int, null_per_tx: int, max_null_tx: int, seed: int):
    A = load_adapter()
    rng = np.random.default_rng(seed)
    txi = tx.set_index("transcript_id")
    tx_len = txi["tx_len"]
    ws = {t: window_start(L, A.MAX_LEN, A.CROP) for t, L in tx_len.items()}
    we = {t: window_end(L, A.MAX_LEN, A.CROP) for t, L in tx_len.items()}
    parts = []

    def genome_to_tx(df):
        minus = df["strand"] == "-"
        df["tx_ref"] = np.where(minus, df["ref"].map(revcomp), df["ref"])
        df["tx_alt"] = np.where(minus, df["alt"].map(revcomp), df["alt"])
        df["tx_start"] = np.where(minus, df["tx_pos"] - df["ref"].str.len() + 1, df["tx_pos"]).astype("int64")
        return df

    cand = gtex_out / "gtex_localization_candidates.parquet"
    if cand.exists():
        g = pd.read_parquet(cand)
        log(f"GTEx: using exported candidate set ({len(g):,})")
    else:
        g = pd.read_parquet(gtex_out / "gtex_variants.parquet")
        g = g[g["region"].isin(["3UTR", "5UTR", "ncRNA_exon"]) & (g["max_pip"] >= 0.5) & g["is_snv"]
              & g["is_egene_transcript"].fillna(False) & ~g["crosses_exon_end"].fillna(False)]
        log(f"GTEx: no exported set; default filter (UTR/lncRNA, PIP>=0.5, SNV, own eGene) -> {len(g):,}")
    g = g[g["transcript_id"].isin(tx_len.index)].copy()
    g = genome_to_tx(g)
    g["set"] = "gtex_candidate"
    g = g.rename(columns={"best_egene_name": "gene"})
    parts.append(g[["set", "variant", "chrom", "pos", "ref", "alt", "transcript_id", "region", "tx_start", "tx_ref",
                    "tx_alt", "max_pip", "best_tissue", "gene"]])

    files = sorted((out / "gnomad_exonic").glob("*.parquet"))
    cols_g = ["variant", "chrom", "pos", "ref", "alt", "transcript_id", "region", "tx_start", "tx_ref", "tx_alt",
              "AF", "f_max", "faf95_max", "gnomad_class", "gene"]
    con = duckdb.connect()
    rare = common = snv_cols = None
    if files:
        con.register("txids", pd.DataFrame({"transcript_id": list(tx_len.index)}))
        flist = ", ".join(f"'{f.as_posix()}'" for f in files)
        con.execute(f"""
            CREATE VIEW gn AS
            SELECT *, gene_name AS gene FROM read_parquet([{flist}])
            WHERE region IN ('3UTR', '5UTR', 'ncRNA_exon') AND is_snv
              AND transcript_id IN (SELECT transcript_id FROM txids)
            QUALIFY row_number() OVER (PARTITION BY variant
                    ORDER BY CASE region WHEN '3UTR' THEN 0 WHEN '5UTR' THEN 1 ELSE 2 END) = 1
        """)
        con.execute("CREATE TABLE gn_snv AS SELECT * FROM gn")
        cnt_rare, cnt_comm = con.sql("""SELECT count(*) FILTER (WHERE gnomad_class = 'rare'),
                                           count(*) FILTER (WHERE gnomad_class = 'common' AND AF >= 0.05) FROM gn_snv""").fetchall()[0]
        log(f"gnomAD exonic SNVs (UTR/lncRNA): rare {cnt_rare:,}, common(AF>=5%) {cnt_comm:,}")
        rare = con.sql(f"""SELECT * FROM (SELECT * FROM gn_snv WHERE gnomad_class = 'rare')
                           USING SAMPLE reservoir({n_rare} ROWS) REPEATABLE ({seed})""").df() if cnt_rare else None
        common = con.sql("SELECT * FROM gn_snv WHERE gnomad_class = 'common' AND AF >= 0.05").df() if cnt_comm else None
        if rare is not None:
            rare = genome_to_tx(rare)
            parts.append(rare[cols_g].assign(set="gnomad_rare"))
        if common is not None:
            common = genome_to_tx(common)
            parts.append(common.sample(min(n_common, len(common)), random_state=seed)[cols_g].assign(set="gnomad_common"))

    mi = scan_motifs(tx, out)
    mi = mi[[ws[t] <= p <= we[t] for t, p in zip(mi["transcript_id"], mi["core_pos"])]]
    eng = pd.concat([d.sample(min(n_per_motif, len(d)), random_state=seed) for _, d in mi.groupby("motif")]) if len(mi) \
        else mi.copy()
    eng = eng.assign(set="pos_engineered", tx_start=eng["core_pos"], tx_ref=eng["core_ref"], tx_alt=eng["core_alt"],
                     variant=[f"{t}:{p}{a}>{b}" for t, p, a, b in zip(eng["transcript_id"], eng["core_pos"], eng["core_ref"], eng["core_alt"])])
    eng["gene"] = eng["transcript_id"].map(txi["gene_name"])
    parts.append(eng[["set", "variant", "transcript_id", "region", "tx_start", "tx_ref", "tx_alt", "motif", "gene"]])

    obs = pd.DataFrame()
    if files:
        con.register("mi_df", mi[["transcript_id", "core_pos", "motif"]])
        obs = con.sql("""
            SELECT g.*, m.motif FROM gn_snv g
            JOIN mi_df m ON g.transcript_id = m.transcript_id AND g.tx_pos = m.core_pos
            QUALIFY row_number() OVER (PARTITION BY g.variant ORDER BY m.motif) = 1
        """).df()
        if len(obs):
            obs = genome_to_tx(obs)
            parts.append(obs[cols_g + ["motif"]].assign(set="pos_observed"))
    log(f"positives: engineered {len(eng):,}, observed {len(obs):,}")

    if common is not None and len(common):
        in_cs = set()
        gv = gtex_out / "gtex_variants.parquet"
        if gv.exists():
            in_cs = set(pd.read_parquet(gv, columns=["variant", "max_pip"]).query("max_pip >= 0.01")["variant"])
        allmi = scan_motifs(tx, out)
        spans = {t: list(zip(d.motif_start, d.motif_end)) for t, d in allmi.groupby("transcript_id")}
        pool = common[~common["variant"].isin(in_cs)].copy()
        pool = pool[[not any(a <= p <= b for a, b in spans.get(t, ())) for t, p in zip(pool["transcript_id"], pool["tx_start"])]]
        pool = pool[[ws[t] <= p <= we[t] for t, p in zip(pool["transcript_id"], pool["tx_start"])]]
        pool["dist"] = pool["transcript_id"].map(tx_len) - pool["tx_start"]
        pool["key"] = pool["region"] + "|" + _dist_bin(pool["dist"]).astype(str) + "|" + pool["tx_ref"] + ">" + pool["tx_alt"]
        pos_all = pd.concat([eng[["transcript_id", "region", "tx_start", "tx_ref", "tx_alt"]],
                             obs[["transcript_id", "region", "tx_start", "tx_ref", "tx_alt"]] if len(obs) else None])
        pos_all["dist"] = pos_all["transcript_id"].map(tx_len) - pos_all["tx_start"]
        pos_all["key"] = pos_all["region"] + "|" + _dist_bin(pos_all["dist"]).astype(str) + "|" + pos_all["tx_ref"] + ">" + pos_all["tx_alt"]
        want = pos_all["key"].value_counts() * neg_per_pos
        picks = []
        for key, n in want.items():
            cand_k = pool[pool["key"] == key]
            if len(cand_k):
                picks.append(cand_k.sample(min(n, len(cand_k)), random_state=seed))
        if picks:
            neg = pd.concat(picks).drop_duplicates("variant")
            parts.append(neg[cols_g].assign(set="neg_matched"))
            log(f"negatives: {len(neg):,} matched common SNVs (target {int(want.sum()):,}; pool {len(pool):,})")

    con.close()
    sets = pd.concat(parts, ignore_index=True)

    txs = sets["transcript_id"].dropna().unique()
    if len(txs) > max_null_tx:
        txs = rng.choice(txs, max_null_tx, replace=False)
    nulls = []
    for t in txs:
        a, b = ws[t], we[t]
        seq = txi.at[t, "seq"]
        for p in rng.integers(a, b + 1, size=null_per_tx):
            r = seq[p - 1]
            if r not in "ACGT":
                continue
            alt = rng.choice([x for x in "ACGT" if x != r])
            nulls.append(("null", f"{t}:{p}{r}>{alt}", t, int(p), r, alt))
    sets = pd.concat([sets, pd.DataFrame(nulls, columns=["set", "variant", "transcript_id", "tx_start", "tx_ref", "tx_alt"])],
                     ignore_index=True)

    def ref_ok(t, s, r):
        seq = txi.at[t, "seq"] if t in txi.index else ""
        return seq[s - 1:s - 1 + len(r)] == r
    sets["tx_start"] = sets["tx_start"].astype("int64")
    sets["ref_ok"] = [ref_ok(t, s, r) for t, s, r in zip(sets["transcript_id"], sets["tx_start"], sets["tx_ref"])]
    sets["tx_len"] = sets["transcript_id"].map(tx_len)
    sets["dist_to_3p_end"] = sets["tx_len"] - sets["tx_start"]
    sets["in_window"] = [ws[t] <= s and s + len(r) - 1 <= we[t] for t, s, r in zip(sets["transcript_id"], sets["tx_start"], sets["tx_ref"])]
    sets.to_parquet(out / "variant_sets.parquet", index=False)
    summ = sets.groupby("set").agg(variants=("variant", "size"), ref_ok=("ref_ok", "mean"), in_window=("in_window", "mean"))
    log("Variant sets:\n" + summ.to_string())
    (out / "prepare_settings.json").write_text(json.dumps({"MAX_LEN": A.MAX_LEN, "CROP": A.CROP, "seed": seed}, indent=2))


def step_score(out: Path, chunk: int = 2000):
    A = load_adapter()
    sets = pd.read_parquet(out / "variant_sets.parquet")
    tx = pd.read_parquet(out / "transcripts.parquet", columns=["transcript_id", "seq"]).set_index("transcript_id")["seq"]
    sets = sets[sets["ref_ok"]].reset_index(drop=True)
    settings = json.loads((out / "prepare_settings.json").read_text())
    if settings["MAX_LEN"] != A.MAX_LEN or settings["CROP"] != A.CROP:
        log("WARNING: MAX_LEN/CROP changed since prepare - re-run prepare so controls and nulls sit inside the new window")

    def crop(s):
        if A.MAX_LEN and len(s) > A.MAX_LEN:
            return s[-A.MAX_LEN:] if A.CROP == "3p" else s[:A.MAX_LEN]
        return s

    def mutate(t, start, r, a):
        s = tx[t]
        return s[:start - 1] + a + s[start - 1 + len(r):]

    wt_seq = {t: crop(tx[t]) for t in sets["transcript_id"].unique()}
    mut_seq = [crop(mutate(t, s, r, a)) for t, s, r, a in zip(sets["transcript_id"], sets["tx_start"], sets["tx_ref"], sets["tx_alt"])]
    h = lambda s: hashlib.sha1(s.encode()).hexdigest()
    wt_hash = {t: h(s) for t, s in wt_seq.items()}
    mut_hash = [h(s) for s in mut_seq]

    cache_dir = out / "pred_cache"
    cache_dir.mkdir(exist_ok=True)
    cached = {}
    for f in cache_dir.glob("*.parquet"):
        d = pd.read_parquet(f)
        cached.update(zip(d["hash"], d[A.COMPARTMENTS].to_numpy()))
    todo = {}
    for t, s in wt_seq.items():
        if wt_hash[t] not in cached:
            todo[wt_hash[t]] = s
    for hh, s in zip(mut_hash, mut_seq):
        if hh not in cached:
            todo[hh] = s
    log(f"{len(sets):,} variants; {len(todo):,} sequences to run ({len(cached):,} cached). Model: {'DEMO' if getattr(A, 'IS_DEMO', False) else 'lab model'}")

    model = A.load()
    items = list(todo.items())
    t0 = time.time()
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        preds = []
        for j in range(0, len(part), A.BATCH_SIZE):
            batch = [s for _, s in part[j:j + A.BATCH_SIZE]]
            p = np.asarray(A.predict(model, batch), dtype="float64")
            if A.OUTPUT == "logit":
                p = 1 / (1 + np.exp(-p))
            preds.append(p)
        P = np.vstack(preds)
        d = pd.DataFrame(P, columns=A.COMPARTMENTS)
        d.insert(0, "hash", [k for k, _ in part])
        d.to_parquet(cache_dir / f"part_{int(time.time()*1000)}_{i}.parquet", index=False)
        cached.update(zip(d["hash"], P))
        done = min(i + chunk, len(items))
        rate = done / max(time.time() - t0, 1e-9)
        log(f"  scored {done:,}/{len(items):,} sequences ({rate:.1f}/s, ~{(len(items)-done)/max(rate,1e-9)/60:.1f} min left)")

    W = np.vstack([cached[wt_hash[t]] for t in sets["transcript_id"]])
    M = np.vstack([cached[x] for x in mut_hash])
    for k, c in enumerate(A.COMPARTMENTS):
        sets[f"wt_{c}"] = W[:, k]
        sets[f"mut_{c}"] = M[:, k]
    sets.to_parquet(out / "scores.parquet", index=False)
    (out / "score_meta.json").write_text(json.dumps({"compartments": A.COMPARTMENTS, "is_demo": bool(getattr(A, "IS_DEMO", False)),
                                                     "max_len": A.MAX_LEN, "crop": A.CROP}, indent=2))
    log(f"Wrote {out / 'scores.parquet'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "score", "all"])
    ap.add_argument("--gtex-out", default="gtex_out")
    ap.add_argument("--gnomad-dir", default="gnomad_slim")
    ap.add_argument("--out", default="loc_out")
    ap.add_argument("--n-rare", type=int, default=20000)
    ap.add_argument("--n-common", type=int, default=20000)
    ap.add_argument("--n-per-motif", type=int, default=500)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--null-per-tx", type=int, default=20)
    ap.add_argument("--max-null-tx", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mirror", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    if a.step in ("prepare", "all"):
        tx = step_transcripts(Path(a.gtex_out), out, a.mirror)
        step_gnomad_exonic(Path(a.gtex_out), Path(a.gnomad_dir), out)
        step_sets(Path(a.gtex_out), out, tx, a.n_rare, a.n_common, a.n_per_motif, a.neg_per_pos, a.null_per_tx,
                  a.max_null_tx, a.seed)
    if a.step in ("score", "all"):
        step_score(out)
    log(f"Finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
