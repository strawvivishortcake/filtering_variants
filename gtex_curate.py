from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

PATHS_TABLE_URL = (
    "https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv"
)
METADATA_URL = (
    "https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/data_tables/dataset_metadata_r8.tsv"
)
GENCODE_URLS = [
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz",
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz",
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.basic.annotation.gtf.gz",
]
EBI_HTTPS = "https://ftp.ebi.ac.uk"
UA = {"User-Agent": "gtex-curate/1.0 (research script)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url: str, dest: Path, mirror: str | None = None, retries: int = 4, check_gzip: bool = False) -> Path:
    if mirror:
        url = url.replace(EBI_HTTPS, mirror).replace("https://raw.githubusercontent.com", mirror)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            if check_gzip:
                with gzip.open(tmp, "rb") as g:
                    while g.read(1 << 22):
                        pass
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            last = e
            time.sleep(3 * attempt)
    raise RuntimeError(f"download failed after {retries} tries: {url} ({last!r})")


def step_download(out: Path, quant: str, mirror: str | None, threads: int):
    raw = out / "raw"
    paths_tsv = fetch(PATHS_TABLE_URL, raw / "tabix_ftp_paths.tsv", mirror)
    try:
        fetch(METADATA_URL, raw / "dataset_metadata_r8.tsv", mirror)
    except Exception as e:
        log(f"(metadata table not fetched, continuing: {e})")
    paths = pd.read_csv(paths_tsv, sep="\t")
    gtex = paths[(paths["study_label"] == "GTEx") & (paths["quant_method"] == quant)].copy()
    if gtex.empty:
        raise SystemExit("No GTEx rows found in the eQTL Catalogue path table - has the layout changed?")
    gtex["url"] = gtex["ftp_cs_path"].str.replace("ftp://ftp.ebi.ac.uk", EBI_HTTPS, regex=False)
    gtex["local"] = [str(raw / "credible_sets" / f"{d}.credible_sets.tsv.gz") for d in gtex["dataset_id"]]
    gtex[["dataset_id", "sample_group", "tissue_label", "sample_size", "quant_method", "url", "local"]].to_csv(
        out / "gtex_datasets.tsv", sep="\t", index=False)
    log(f"{len(gtex)} GTEx '{quant}' datasets; downloading credible sets ({threads} at a time)")

    done = 0
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(fetch, u, Path(l), mirror, 4, True): t for u, l, t in zip(gtex["url"], gtex["local"], gtex["sample_group"])}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(futs):
                log(f"  credible sets: {done}/{len(futs)}")

    gtf_path = None
    for u in GENCODE_URLS:
        try:
            gtf_path = fetch(u, raw / Path(u).name, mirror, 3, True)
            break
        except Exception as e:
            log(f"  GENCODE not available at {u} ({e}); trying next")
    if gtf_path is None:
        raise SystemExit("Could not download a GENCODE basic GTF")
    log(f"GENCODE: {gtf_path.name}")
    release = gtf_path.name.split(".")[1][1:]
    fetch(f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/gencode.v{release}.transcripts.fa.gz",
          raw / f"gencode.v{release}.transcripts.fa.gz", mirror, 3, True)
    return gtf_path


def _attrs(s: str) -> dict:
    d, tags = {}, []
    for part in s.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(" ")
        v = v.strip('"')
        if k == "tag":
            tags.append(v)
        else:
            d[k] = v
    d["_tags"] = tags
    return d


def step_gtf(out: Path, gtf_path: Path):
    tx_pq, ex_pq, cds_pq = out / "gencode_canonical_transcripts.parquet", out / "gencode_canonical_exons.parquet", out / "gencode_canonical_cds.parquet"
    if tx_pq.exists() and ex_pq.exists() and cds_pq.exists():
        log("GTF tables already built")
        return
    log(f"Parsing {gtf_path.name} (canonical transcripts only)")
    tx, ex, cds = [], [], []
    keep = {"transcript", "exon", "CDS", "start_codon", "stop_codon"}
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] not in keep or "Ensembl_canonical" not in p[8]:
                continue
            a = _attrs(p[8])
            if "Ensembl_canonical" not in a["_tags"]:
                continue
            chrom, feat, start, end, strand = p[0], p[2], int(p[3]), int(p[4]), p[6]
            tid = a["transcript_id"].split(".")[0]
            if feat == "transcript":
                tx.append({
                    "transcript_id": tid, "gene_id": a["gene_id"].split(".")[0], "gene_name": a.get("gene_name"),
                    "gene_type": a.get("gene_type"), "transcript_type": a.get("transcript_type"),
                    "mane_select": "MANE_Select" in a["_tags"], "chrom": chrom, "tx_start": start, "tx_end": end,
                    "strand": strand,
                })
            elif feat == "exon":
                ex.append({"transcript_id": tid, "chrom": chrom, "start": start, "end": end, "strand": strand,
                           "exon_number": int(a["exon_number"])})
            else:
                cds.append({"transcript_id": tid, "start": start, "end": end})
    pd.DataFrame(tx).to_parquet(tx_pq, index=False)
    pd.DataFrame(ex).to_parquet(ex_pq, index=False)
    pd.DataFrame(cds).to_parquet(cds_pq, index=False)
    log(f"  {len(tx):,} canonical transcripts, {len(ex):,} exons")


def step_annotate(out: Path, pip_min_keep: float):
    con = duckdb.connect(str(out / "work.duckdb"))
    con.execute("PRAGMA threads=%d" % max(1, (os.cpu_count() or 2)))
    ds = pd.read_csv(out / "gtex_datasets.tsv", sep="\t")
    ds["local"] = [Path(p).as_posix() for p in ds["local"]]
    con.register("ds_df", ds)
    con.execute("CREATE OR REPLACE TABLE ds AS SELECT * FROM ds_df")

    files = ", ".join(f"'{p}'" for p in ds["local"])
    log(f"Loading {len(ds)} credible-set files")
    con.execute(f"""
        CREATE OR REPLACE TABLE cs_raw AS
        SELECT * FROM read_csv([{files}], delim='\t', header=true, all_varchar=true,
                               union_by_name=true, filename=true, compression='gzip')
    """)
    cols = {r[0] for r in con.sql("DESCRIBE cs_raw").fetchall()}
    need = {"molecular_trait_id", "variant", "pip", "cs_id"}
    if not need <= cols:
        raise SystemExit(f"Credible-set files are missing columns {need - cols}; found {sorted(cols)}")
    opt = lambda c, expr: expr if c in cols else "NULL"
    con.execute(f"""
        CREATE OR REPLACE TABLE cs AS
        SELECT d.sample_group AS tissue,
               split_part(c.molecular_trait_id, '.', 1)            AS gene_id,
               c.variant,
               'chr' || replace(split_part(c.variant, '_', 1), 'chr', '') AS chrom,
               CAST(split_part(c.variant, '_', 2) AS INTEGER)       AS pos,
               split_part(c.variant, '_', 3)                        AS ref,
               split_part(c.variant, '_', 4)                        AS alt,
               c.cs_id,
               CAST(c.pip AS DOUBLE)                                AS pip,
               {opt('cs_size', 'TRY_CAST(c.cs_size AS INTEGER)')}   AS cs_size,
               {opt('cs_min_r2', 'TRY_CAST(c.cs_min_r2 AS DOUBLE)')} AS cs_min_r2,
               {opt('z', 'TRY_CAST(c.z AS DOUBLE)')}                AS z,
               {opt('posterior_mean', 'TRY_CAST(c.posterior_mean AS DOUBLE)')} AS posterior_mean,
               {opt('rsid', 'c.rsid')}                              AS rsid
        FROM cs_raw c JOIN ds d ON replace(c.filename, '\\', '/') = d.local
    """)
    n_rows = con.sql("SELECT count(*) FROM cs").fetchall()[0][0]
    log(f"  {n_rows:,} credible-set rows")
    con.execute("DROP TABLE cs_raw")
    con.execute(f"COPY cs TO '{(out / 'gtex_credible_sets.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)")

    con.execute("""
        CREATE OR REPLACE TABLE var AS
        SELECT variant, any_value(chrom) chrom, any_value(pos) pos, any_value(ref) ref, any_value(alt) alt,
               any_value(rsid) rsid,
               max(pip)                                       AS max_pip,
               arg_max(tissue, pip)                           AS best_tissue,
               arg_max(gene_id, pip)                          AS best_egene,
               count(DISTINCT tissue)                         AS n_tissues_in_cs,
               count(DISTINCT tissue) FILTER (WHERE pip >= 0.5) AS n_tissues_pip50,
               count(DISTINCT gene_id)                        AS n_egenes,
               string_agg(DISTINCT gene_id, ',')              AS egenes,
               arg_max(cs_size, pip)                          AS best_cs_size,
               arg_max(cs_min_r2, pip)                        AS best_cs_min_r2,
               arg_max(z, pip)                                AS best_z,
               length(any_value(ref)) = 1 AND length(any_value(alt)) = 1 AS is_snv
        FROM cs GROUP BY variant
    """)
    n_var = con.sql("SELECT count(*) FROM var").fetchall()[0][0]
    log(f"  {n_var:,} unique variants in any credible set")

    con.execute(f"CREATE OR REPLACE TABLE tx AS SELECT * FROM '{(out / 'gencode_canonical_transcripts.parquet').as_posix()}'")
    con.execute(f"""
        CREATE OR REPLACE TABLE ex AS
        SELECT *, coalesce(sum(\"end\" - start + 1) OVER (PARTITION BY transcript_id ORDER BY exon_number
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS tx_offset
        FROM '{(out / 'gencode_canonical_exons.parquet').as_posix()}'
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cdsspan AS
        SELECT transcript_id, min(start) cds_min, max("end") cds_max
        FROM '{(out / 'gencode_canonical_cds.parquet').as_posix()}' GROUP BY 1
    """)
    con.execute("""CREATE OR REPLACE TABLE txlen AS SELECT transcript_id, sum("end" - start + 1) AS tx_len FROM ex GROUP BY 1""")

    con.execute("""
        CREATE OR REPLACE TABLE var_tx AS
        SELECT v.variant, e.transcript_id, t.gene_id, t.gene_name, t.gene_type, t.transcript_type, t.mane_select,
               e.strand, e.exon_number,
               CASE WHEN e.strand = '+' THEN e.tx_offset + (v.pos - e.start + 1)
                    ELSE e.tx_offset + (e."end" - v.pos + 1) END AS tx_pos,
               l.tx_len,
               CASE WHEN s.transcript_id IS NULL THEN 'ncRNA_exon'
                    WHEN v.pos BETWEEN s.cds_min AND s.cds_max THEN 'CDS'
                    WHEN (e.strand = '+' AND v.pos < s.cds_min) OR (e.strand = '-' AND v.pos > s.cds_max) THEN '5UTR'
                    ELSE '3UTR' END AS region,
               list_contains(string_split(v.egenes, ','), t.gene_id) AS is_egene_transcript,
               t.gene_id = v.best_egene AS is_best_egene_transcript,
               v.pos + length(v.ref) - 1 > e."end" AS crosses_exon_end
        FROM var v
        JOIN ex e ON v.chrom = e.chrom AND v.pos BETWEEN e.start AND e."end"
        JOIN tx t USING (transcript_id)
        JOIN txlen l USING (transcript_id)
        LEFT JOIN cdsspan s USING (transcript_id)
    """)
    con.execute("ALTER TABLE var_tx ADD COLUMN dist_to_3p_end INTEGER")
    con.execute("UPDATE var_tx SET dist_to_3p_end = tx_len - tx_pos")
    con.execute(f"COPY var_tx TO '{(out / 'gtex_variant_transcript_exonic.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)")

    con.execute("""
        CREATE OR REPLACE TABLE best AS
        SELECT * EXCLUDE (rk) FROM (
            SELECT *, row_number() OVER (PARTITION BY variant ORDER BY is_best_egene_transcript DESC,
                     is_egene_transcript DESC,
                     CASE region WHEN '3UTR' THEN 0 WHEN '5UTR' THEN 1 WHEN 'ncRNA_exon' THEN 2 ELSE 3 END,
                     mane_select DESC) AS rk
            FROM var_tx) WHERE rk = 1
    """)
    con.execute("""
        CREATE OR REPLACE TABLE intronic AS
        SELECT DISTINCT v.variant FROM var v
        JOIN tx t ON v.chrom = t.chrom AND v.pos BETWEEN t.tx_start AND t.tx_end
        WHERE v.variant NOT IN (SELECT variant FROM best)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE gtex_variants AS
        SELECT v.*,
               coalesce(b.region, CASE WHEN i.variant IS NOT NULL THEN 'intronic' ELSE 'intergenic' END) AS region,
               b.transcript_id, b.gene_id AS tx_gene_id, b.gene_name AS tx_gene_name, b.gene_type AS tx_gene_type,
               b.mane_select, b.strand, b.exon_number, b.tx_pos, b.tx_len, b.dist_to_3p_end,
               b.is_egene_transcript, b.is_best_egene_transcript, b.crosses_exon_end,
               g.gene_name AS best_egene_name
        FROM var v
        LEFT JOIN best b USING (variant)
        LEFT JOIN intronic i USING (variant)
        LEFT JOIN (SELECT DISTINCT gene_id, gene_name FROM tx) g ON g.gene_id = v.best_egene
    """)
    con.execute(f"COPY gtex_variants TO '{(out / 'gtex_variants.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"""
        COPY (SELECT * FROM gtex_variants WHERE region IN ('3UTR','5UTR','ncRNA_exon','CDS') AND max_pip >= {pip_min_keep}
              ORDER BY max_pip DESC)
        TO '{(out / 'gtex_exonic_candidates.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    summ = con.sql("""
        SELECT region, count(*) variants,
               count(*) FILTER (WHERE max_pip >= 0.1) pip_ge_0_1,
               count(*) FILTER (WHERE max_pip >= 0.5) pip_ge_0_5,
               count(*) FILTER (WHERE max_pip >= 0.9) pip_ge_0_9
        FROM gtex_variants GROUP BY 1 ORDER BY 2 DESC
    """).df()
    summ.to_csv(out / "gtex_region_summary.csv", index=False)
    log("Region summary:\n" + summ.to_string(index=False))
    con.close()


def step_gnomad(out: Path, gnomad_dir: Path | None):
    if gnomad_dir is None or not gnomad_dir.exists():
        log("No gnomad_slim folder found; skipping gnomAD join (re-run later with --steps gnomad)")
        return
    done = sorted(p.name.split(".")[0] for p in gnomad_dir.glob("*.done.json"))
    if not done:
        log("gnomad_slim has no finished chromosomes yet; skipping")
        return
    con = duckdb.connect()
    files = ", ".join(f"'{(gnomad_dir / f'{c}.parquet').as_posix()}'" for c in done)
    log(f"Joining gnomAD frequencies for {len(done)} chromosome(s): {', '.join(done)}")
    con.execute(f"CREATE TABLE v AS SELECT variant, chrom, pos, ref, alt FROM '{(out / 'gtex_variants.parquet').as_posix()}'")
    con.execute(f"""
        CREATE TABLE j AS
        SELECT v.variant, g.filter AS gnomad_filter, g.AC AS gnomad_AC, g.AN AS gnomad_AN, g.AF AS gnomad_AF,
               g.AF_grpmax AS gnomad_AF_grpmax, g.grpmax AS gnomad_grpmax, g.faf95_max AS gnomad_faf95_max,
               greatest(coalesce(g.AF,0), coalesce(g.AF_grpmax,0), coalesce(g.af_max_any_grp,0)) AS gnomad_f_max,
               g.lcr AS gnomad_lcr, g.segdup AS gnomad_segdup
        FROM v JOIN read_parquet([{files}]) g
          ON g.chrom = v.chrom AND g.pos = v.pos AND g.ref = v.ref AND g.alt = v.alt
    """)
    con.execute(f"COPY j TO '{(out / 'gtex_variants_gnomad.parquet').as_posix()}' (FORMAT parquet)")
    n = con.sql("SELECT count(*) FROM j").fetchall()[0][0]
    (out / "gnomad_join.json").write_text(json.dumps({"chromosomes": done, "matched_variants": n}, indent=2))
    log(f"  matched {n:,} GTEx variants to gnomAD")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="gtex_out")
    ap.add_argument("--quant", default="ge", help="eQTL Catalogue quant method: ge (gene), tx, txrev, exon, leafcutter")
    ap.add_argument("--steps", default="download,gtf,annotate,gnomad")
    ap.add_argument("--gnomad-dir", default="gnomad_slim")
    ap.add_argument("--pip-min", type=float, default=0.1, help="min PIP for the exonic candidates file")
    ap.add_argument("--threads", type=int, default=6, help="parallel downloads")
    ap.add_argument("--mirror", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [s.strip() for s in a.steps.split(",")]
    t0 = time.time()
    gtf = None
    if "download" in steps:
        gtf = step_download(out, a.quant, a.mirror, a.threads)
    if "gtf" in steps:
        gtf = gtf or next((p for p in sorted((out / "raw").glob("gencode.v*.basic.annotation.gtf.gz"))), None)
        if gtf is None:
            raise SystemExit("No GENCODE GTF found; run the download step first")
        step_gtf(out, gtf)
    if "annotate" in steps:
        step_annotate(out, a.pip_min)
    if "gnomad" in steps:
        step_gnomad(out, Path(a.gnomad_dir))
    log(f"Done in {(time.time() - t0) / 60:.1f} min. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
