import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="GTEx fine-mapped eQTLs for RNA localization")


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    BLUE, ORANGE, AQUA, GOLD, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b5b3ad"
    REGIONS = ["3UTR", "5UTR", "ncRNA_exon", "CDS", "intronic", "intergenic"]
    return AQUA, BLUE, GOLD, GRAY, ORANGE, Path, REGIONS, alt, duckdb, json, mo, os, pd


@app.cell(hide_code=True)
def _(Path, duckdb, json, mo, os, pd):
    OUT = Path(os.environ.get("LOCVAR_DATA", str(mo.notebook_dir() or "."))) / "gtex_out"
    mo.stop(not (OUT / "gtex_variants.parquet").exists(), mo.callout(mo.md(f"No results in `{OUT}`"), kind="warn"))
    con = duckdb.connect()
    con.execute(f"CREATE VIEW var AS SELECT * FROM '{(OUT / 'gtex_variants.parquet').as_posix()}'")
    con.execute(f"CREATE VIEW cs AS SELECT * FROM '{(OUT / 'gtex_credible_sets.parquet').as_posix()}'")
    HAS_GNOMAD = (OUT / "gtex_variants_gnomad.parquet").exists()
    if HAS_GNOMAD:
        con.execute(f"CREATE VIEW gn AS SELECT * FROM '{(OUT / 'gtex_variants_gnomad.parquet').as_posix()}'")
        GNOMAD_INFO = json.loads((OUT / "gnomad_join.json").read_text())
    else:
        GNOMAD_INFO = {"chromosomes": [], "matched_variants": 0}
    datasets = pd.read_csv(OUT / "gtex_datasets.tsv", sep="\t")
    _gtf = sorted((OUT / "raw").glob("gencode.v*.gtf.gz"))
    _n = con.sql("SELECT count(*), count(DISTINCT best_egene) FROM var").fetchall()[0]
    _rows = con.sql("SELECT count(*) FROM cs").fetchall()[0][0]
    mo.hstack([
        mo.stat(value=str(datasets["sample_group"].nunique()), label="GTEx tissues"),
        mo.stat(value=f"{_rows:,}", label="credible-set rows"),
        mo.stat(value=f"{_n[0]:,}", label="unique variants"),
        mo.stat(value=f"{_n[1]:,}", label="eGenes (best per variant)"),
        mo.stat(value=_gtf[0].name.split(".")[1] if _gtf else "?", label="GENCODE release"),
    ], widths="equal")
    return GNOMAD_INFO, HAS_GNOMAD, OUT, con, datasets


@app.cell(hide_code=True)
def _(BLUE, REGIONS, alt, con, mo):
    _d = con.sql("""
        SELECT region, count(*) AS "all in credible sets",
               count(*) FILTER (WHERE max_pip >= 0.1) AS "PIP ≥ 0.1",
               count(*) FILTER (WHERE max_pip >= 0.5) AS "PIP ≥ 0.5",
               count(*) FILTER (WHERE max_pip >= 0.9) AS "PIP ≥ 0.9"
        FROM var GROUP BY 1
    """).df().set_index("region").reindex(REGIONS).fillna(0).astype("int64").reset_index()
    region_table = _d
    _p = _d.rename(columns={"PIP ≥ 0.5": "pip50", "PIP ≥ 0.1": "pip10", "PIP ≥ 0.9": "pip90", "all in credible sets": "all_cs"})
    _chart = (
        alt.Chart(_p)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, height=20)
        .encode(
            y=alt.Y("region:N", sort=REGIONS, title=None),
            x=alt.X("pip50:Q", title="Variants with PIP ≥ 0.5 (symlog)", scale=alt.Scale(type="symlog")),
            tooltip=["region", alt.Tooltip("all_cs:Q", format=",", title="all in credible sets"),
                     alt.Tooltip("pip10:Q", format=",", title="PIP ≥ 0.1"), alt.Tooltip("pip50:Q", format=",", title="PIP ≥ 0.5"),
                     alt.Tooltip("pip90:Q", format=",", title="PIP ≥ 0.9")],
        )
        .properties(width=560, height=200, title="Likely-causal eQTL variants by transcript region")
    )
    mo.vstack([_chart, mo.ui.table(_d, selection=None)])
    return (region_table,)


@app.cell(hide_code=True)
def _(ORANGE, REGIONS, alt, con, mo):
    _e = con.sql("""
        WITH t AS (SELECT region,
                          count(*) FILTER (WHERE max_pip >= 0.5)  AS hi,
                          count(*) FILTER (WHERE max_pip < 0.05)  AS lo
                   FROM var GROUP BY 1)
        SELECT region, hi, lo,
               (hi::DOUBLE / sum(hi) OVER ()) / nullif(lo::DOUBLE / sum(lo) OVER (), 0) AS enrichment
        FROM t
    """).df()
    _e = _e.set_index("region").reindex(REGIONS).reset_index()
    _base = alt.Chart(_e).encode(y=alt.Y("region:N", sort=REGIONS, title=None))
    _bars = _base.mark_bar(color=ORANGE, cornerRadiusEnd=4, height=20).encode(
        x=alt.X("enrichment:Q", title="Enrichment of PIP ≥ 0.5 vs PIP < 0.05"),
        tooltip=["region", alt.Tooltip("enrichment:Q", format=".2f"), alt.Tooltip("hi:Q", format=",", title="PIP ≥ 0.5"),
                 alt.Tooltip("lo:Q", format=",", title="PIP < 0.05")])
    _one = alt.Chart(alt.Data(values=[{"x": 1}])).mark_rule(strokeDash=[4, 3], color="#52514e").encode(x="x:Q")
    (_bars + _one).properties(width=560, height=200, title="Fine-mapping enrichment by region (1 = no enrichment)")
    return


@app.cell(hide_code=True)
def _(mo):
    f_regions = mo.ui.multiselect(options=["3UTR", "5UTR", "ncRNA_exon", "CDS"], value=["3UTR", "5UTR", "ncRNA_exon"],
                                  label="Regions")
    f_pip = mo.ui.slider(0.1, 0.99, step=0.05, value=0.5, label="Min PIP (any tissue)", show_value=True)
    f_snv = mo.ui.checkbox(value=True, label="SNVs only")
    f_egene = mo.ui.checkbox(value=True, label="in its own eGene's transcript")
    f_purity = mo.ui.checkbox(value=True, label="credible set purity (min r²) ≥ 0.5")
    mo.hstack([f_regions, f_pip, f_snv, f_egene, f_purity], justify="start", wrap=True)
    return f_egene, f_pip, f_purity, f_regions, f_snv


@app.cell(hide_code=True)
def _(con, f_egene, f_pip, f_purity, f_regions, f_snv, mo):
    _regs = ", ".join(f"'{r}'" for r in (f_regions.value or ["3UTR"]))
    _w = [f"region IN ({_regs})", f"max_pip >= {f_pip.value}", "NOT coalesce(crosses_exon_end, false)"]
    if f_snv.value:
        _w.append("is_snv")
    if f_egene.value:
        _w.append("is_egene_transcript")
    if f_purity.value:
        _w.append("coalesce(best_cs_min_r2, 1) >= 0.5")
    CAND_WHERE = " AND ".join(_w)
    candidates = con.sql(f"""
        SELECT variant, rsid, region, max_pip, best_tissue, n_tissues_pip50, n_tissues_in_cs,
               best_egene_name AS egene, tx_gene_name AS transcript_gene, transcript_id, strand,
               tx_pos, tx_len, dist_to_3p_end, best_z, best_cs_size, n_egenes
        FROM var WHERE {CAND_WHERE} ORDER BY max_pip DESC
    """).df()
    _by = candidates.groupby("region").size().to_dict()
    mo.vstack([
        mo.md(f"**{len(candidates):,} candidate variants** in {candidates['egene'].nunique():,} genes  ·  "
              + "  ·  ".join(f"{k}: {v:,}" for k, v in _by.items())),
        mo.ui.table(candidates, selection=None, page_size=15),
    ])
    return CAND_WHERE, candidates


@app.cell(hide_code=True)
def _(AQUA, BLUE, alt, candidates, mo, pd):
    mo.stop(candidates.empty)
    _t = candidates.groupby("best_tissue").size().reset_index(name="variants").sort_values("variants", ascending=False)
    _tchart = (
        alt.Chart(_t.head(25))
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(y=alt.Y("best_tissue:N", sort="-x", title=None), x=alt.X("variants:Q", title="Candidates (tissue of max PIP)"),
                tooltip=["best_tissue", alt.Tooltip("variants:Q", format=",")])
        .properties(width=300, height=420, title="Top tissues")
    )
    _bins = [0, 500, 1000, 2000, 4000, 8000, 16000, 10**9]
    _labels = ["<0.5k", "0.5–1k", "1–2k", "2–4k", "4–8k", "8–16k", "≥16k"]
    _d = pd.cut(candidates["dist_to_3p_end"], bins=_bins, labels=_labels, right=False).value_counts().reindex(_labels).reset_index()
    _d.columns = ["dist_bin", "variants"]
    _dchart = (
        alt.Chart(_d)
        .mark_bar(color=AQUA, cornerRadiusEnd=4)
        .encode(x=alt.X("dist_bin:N", sort=_labels, title="Nucleotides from transcript 3' end"), y=alt.Y("variants:Q"),
                tooltip=[alt.Tooltip("dist_bin:N", title="nt from 3' end"), alt.Tooltip("variants:Q", format=",")])
        .properties(width=300, height=420, title="Distance from transcript 3' end")
    )
    mo.hstack([_tchart, _dchart], justify="start")
    return


@app.cell(hide_code=True)
def _(BLUE, GRAY, HAS_GNOMAD, ORANGE, AQUA, CAND_WHERE, alt, con, mo):
    mo.stop(not HAS_GNOMAD)
    _g = con.sql(f"""
        SELECT region,
               CASE WHEN g.variant IS NULL THEN 'not in gnomAD join'
                    WHEN greatest(coalesce(g.gnomad_AF,0), coalesce(g.gnomad_faf95_max,0)) >= 0.01 THEN 'common'
                    WHEN g.gnomad_f_max < 0.01 THEN 'rare'
                    ELSE 'in-between / group-specific' END AS gnomad_class,
               count(*) AS variants
        FROM var v LEFT JOIN gn g USING (variant)
        WHERE {CAND_WHERE} GROUP BY 1, 2
    """).df()
    _order = ["common", "in-between / group-specific", "rare", "not in gnomAD join"]
    _chart = (
        alt.Chart(_g)
        .mark_bar(stroke="white", strokeWidth=2)
        .encode(y=alt.Y("region:N", title=None), x=alt.X("variants:Q", stack="zero"),
                color=alt.Color("gnomad_class:N", scale=alt.Scale(domain=_order, range=[ORANGE, AQUA, BLUE, GRAY]), title="gnomAD"),
                tooltip=["region", "gnomad_class", alt.Tooltip("variants:Q", format=",")])
        .properties(width=560, height=140, title="Candidate GTEx variants by gnomAD frequency class")
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    export_btn = mo.ui.run_button(label="Export candidates")
    export_btn
    return (export_btn,)


@app.cell(hide_code=True)
def _(CAND_WHERE, HAS_GNOMAD, OUT, con, export_btn, mo):
    mo.stop(not export_btn.value)
    _join = "LEFT JOIN gn g USING (variant)" if HAS_GNOMAD else ""
    _cols = ", g.* EXCLUDE (variant)" if HAS_GNOMAD else ""
    _q = f"SELECT v.*{_cols} FROM var v {_join} WHERE {CAND_WHERE} ORDER BY max_pip DESC"
    con.execute(f"COPY ({_q}) TO '{(OUT / 'gtex_localization_candidates.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY ({_q}) TO '{(OUT / 'gtex_localization_candidates.csv').as_posix()}' (HEADER)")
    (OUT / "gtex_localization_candidates.filter.txt").write_text(CAND_WHERE)
    mo.callout(mo.md(f"Wrote `gtex_localization_candidates.parquet` / `.csv` to `{OUT}` (filter saved alongside)."), kind="success")
    return


if __name__ == "__main__":
    app.run()
