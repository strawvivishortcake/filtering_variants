import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="gnomAD rare vs. common variant criteria")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    COLORS = {
        "rare": "#2a78d6",
        "common": "#eb6834",
        "in-between / group-specific": "#1baf7a",
        "QC-excluded": "#b5b3ad",
    }
    CLASS_ORDER = list(COLORS)
    return CLASS_ORDER, COLORS, Path, alt, duckdb, json, mo, os, pd, subprocess, sys


@app.cell(hide_code=True)
def _(Path, mo, os):
    OUT = Path(os.environ.get("LOCVAR_DATA", str(mo.notebook_dir() or "."))) / "gnomad_slim"
    reload_btn = mo.ui.button(value=0, on_click=lambda v: v + 1, label="Reload data")
    reload_btn
    return OUT, reload_btn


@app.cell(hide_code=True)
def _(OUT, duckdb, json, mo, pd, reload_btn):
    reload_btn.value
    manifest = pd.DataFrame([json.loads(p.read_text()) for p in OUT.glob("*.done.json")])
    mo.stop(len(manifest) == 0, mo.callout(mo.md(f"No extracted chromosomes in `{OUT}`"), kind="warn"))

    def _key(c):
        c = c.replace("chr", "")
        return (0, int(c)) if c.isdigit() else (1, c)

    CHROMS = sorted(manifest["chrom"].tolist(), key=_key)
    _files = ", ".join(f"'{(OUT / f'{c}.parquet').as_posix()}'" for c in CHROMS)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet([{_files}])")
    con.execute("CREATE TABLE chrom_max AS SELECT chrom, max(AN) AS max_an FROM raw GROUP BY chrom")
    con.execute("""
        CREATE VIEW v AS
        SELECT r.*,
               r.AN::DOUBLE / m.max_an AS an_frac,
               greatest(coalesce(r.AF, 0), coalesce(r.AF_grpmax, 0), coalesce(r.af_max_any_grp, 0)) AS f_max,
               r.filter = 'PASS' AS is_pass
        FROM raw r JOIN chrom_max m USING (chrom)
    """)
    total_variants = con.sql("SELECT count(*) FROM raw").fetchall()[0][0]
    mo.md(f"Loaded **{total_variants:,}** variants across **{len(CHROMS)}** chromosome(s): {', '.join(CHROMS)}")
    return CHROMS, con, total_variants


@app.cell(hide_code=True)
def _(alt, con, mo, pd):
    _q = con.sql("""
        SELECT count(*)                                                        AS "All variants",
               count(*) FILTER (WHERE is_pass)                                 AS "FILTER = PASS",
               count(*) FILTER (WHERE is_pass AND an_frac >= 0.5)              AS "+ AN ≥ 50% of max",
               count(*) FILTER (WHERE is_pass AND an_frac >= 0.5 AND NOT lcr)  AS "+ not low-complexity",
               count(*) FILTER (WHERE is_pass AND an_frac >= 0.5 AND NOT lcr AND NOT segdup) AS "+ not segdup"
        FROM v
    """).df().T.reset_index()
    _q.columns = ["step", "variants"]
    _q["% of all"] = (100 * _q["variants"] / _q["variants"].iloc[0]).round(2)
    _q["removed at this step"] = (-_q["variants"].diff()).fillna(0).astype("int64")
    waterfall = _q

    _filt = con.sql("SELECT filter, count(*) AS variants FROM v GROUP BY 1 ORDER BY 2 DESC").df()
    _filt["%"] = (100 * _filt["variants"] / _filt["variants"].sum()).round(2)

    _chart = (
        alt.Chart(_q)
        .mark_bar(color="#2a78d6", cornerRadiusEnd=4, height=22)
        .encode(
            y=alt.Y("step:N", sort=None, title=None),
            x=alt.X("variants:Q", title="Variants retained"),
            tooltip=["step", alt.Tooltip("variants:Q", format=","), "% of all", alt.Tooltip("removed at this step:Q", format=",")],
        )
        .properties(width=520, height=170, title="Variants retained after each cumulative QC step")
    )
    mo.vstack([
        _chart,
        mo.hstack([mo.ui.table(_q, selection=None), mo.ui.table(_filt, selection=None, label="FILTER values")], align="start"),
    ])
    return (waterfall,)


@app.cell(hide_code=True)
def _(alt, con, mo):
    _d = con.sql("""
        SELECT least(floor(an_frac * 20) / 20, 0.95) AS an_bin, count(*) AS variants
        FROM v WHERE is_pass GROUP BY 1 ORDER BY 1
    """).df()
    _d["an_bin_label"] = _d["an_bin"].map(lambda x: f"{x:.2f}–{x + 0.05:.2f}")
    _chart = (
        alt.Chart(_d)
        .mark_bar(color="#2a78d6", cornerRadiusEnd=4)
        .encode(
            x=alt.X("an_bin_label:N", title="AN / chromosome max AN (PASS variants)", sort=None),
            y=alt.Y("variants:Q", title="Variants", scale=alt.Scale(type="symlog")),
            tooltip=[alt.Tooltip("an_bin_label:N", title="AN fraction"), alt.Tooltip("variants:Q", format=",")],
        )
        .properties(width=620, height=220, title="How completely are PASS sites called? (symlog y)")
    )
    _chart
    return


@app.cell(hide_code=True)
def _(con, mo):
    _qc = "is_pass AND an_frac >= 0.5 AND NOT lcr AND NOT segdup"
    _x = con.sql(f"""
        SELECT CASE WHEN coalesce(AF,0) < 0.01 THEN 'rare by global AF' ELSE 'common by global AF' END AS global_label,
               count(*) FILTER (WHERE f_max < 0.01)                         AS "rare by f_max (all groups)",
               count(*) FILTER (WHERE f_max >= 0.01)                        AS "≥1% in some group (f_max)",
               count(*) FILTER (WHERE coalesce(faf95_max,0) >= 0.01)        AS "faf95 ≥ 1% (confident)"
        FROM v WHERE {_qc} GROUP BY 1 ORDER BY 1 DESC
    """).df()
    _hidden = con.sql(f"""
        SELECT count(*) AS n,
               count(*) FILTER (WHERE coalesce(faf95_max,0) >= 0.01) AS n_conf,
               count(*) FILTER (WHERE coalesce(AF_grpmax,0) < 0.01 AND f_max >= 0.01) AS n_bottleneck_only
        FROM v WHERE {_qc} AND coalesce(AF,0) < 0.01 AND f_max >= 0.01
    """).fetchall()[0]
    mo.vstack([
        mo.ui.table(_x, selection=None),
        mo.hstack([
            mo.stat(value=f"{_hidden[0]:,}", label="AF < 1%, f_max ≥ 1%"),
            mo.stat(value=f"{_hidden[1]:,}", label="of which faf95 ≥ 1%"),
            mo.stat(value=f"{_hidden[2]:,}", label="≥ 1% only in asj/fin/ami/remaining"),
        ], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(alt, con, mo):
    _sfs = con.sql("""
        SELECT allele_type,
               CASE WHEN AC = 1 THEN -9 ELSE floor(log10(AF) * 4) / 4 END AS bin,
               count(*) AS variants
        FROM v
        WHERE is_pass AND an_frac >= 0.5 AND NOT lcr AND NOT segdup AND AF > 0
              AND allele_type IN ('snv', 'ins', 'del')
        GROUP BY 1, 2 ORDER BY 2
    """).df()
    _sfs["AF bin"] = _sfs["bin"].map(lambda b: "singleton (AC=1)" if b == -9 else f"10^{b:.2f}")
    _sfs["log10 AF (bin start)"] = _sfs["bin"].where(_sfs["bin"] != -9, -5.6)
    _palette = alt.Scale(domain=["snv", "del", "ins"], range=["#2a78d6", "#eb6834", "#1baf7a"])
    _lines = (
        alt.Chart(_sfs)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=alt.X("log10 AF (bin start):Q", title="log10 alt-allele frequency (quarter-decade bins; leftmost point = singletons)",
                    scale=alt.Scale(domain=[-5.75, 0])),
            y=alt.Y("variants:Q", scale=alt.Scale(type="log"), title="Variants (log scale)"),
            color=alt.Color("allele_type:N", scale=_palette, title="Allele type"),
            tooltip=["allele_type", "AF bin", alt.Tooltip("variants:Q", format=",")],
        )
    )
    _rules = alt.Chart(alt.Data(values=[{"x": -2, "t": "1%"}, {"x": -1.30103, "t": "5%"}])).mark_rule(
        strokeDash=[4, 3], color="#52514e").encode(x="x:Q")
    _rule_txt = alt.Chart(alt.Data(values=[{"x": -2, "t": "1%"}, {"x": -1.30103, "t": "5%"}])).mark_text(
        dy=-110, dx=12, color="#52514e").encode(x="x:Q", text="t:N")
    (_lines + _rules + _rule_txt).properties(width=640, height=280, title="Site frequency spectrum of QC-passing variants")
    return


@app.cell(hide_code=True)
def _(alt, con, mo, pd):
    _cuts = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    _metrics = {"AF (global)": "coalesce(AF,0)", "AF_grpmax": "coalesce(AF_grpmax,0)",
                "faf95_max": "coalesce(faf95_max,0)", "f_max (all groups)": "f_max"}
    _sel = ", ".join(f"count(*) FILTER (WHERE {e} < {c}) AS \"{m}|{c}\"" for m, e in _metrics.items() for c in _cuts)
    _row = con.sql(f"SELECT count(*) AS n_qc, {_sel} FROM v WHERE is_pass AND an_frac >= 0.5 AND NOT lcr AND NOT segdup").df().iloc[0]
    n_qc = int(_row["n_qc"])
    _long = pd.DataFrame([{"metric": k.split("|")[0], "cutoff": float(k.split("|")[1]), "variants below cutoff": int(v)}
                          for k, v in _row.items() if k != "n_qc"])
    _long["% of QC-passing"] = (100 * _long["variants below cutoff"] / max(n_qc, 1)).round(2)
    _long["cutoff label"] = _long["cutoff"].map(lambda c: f"{c*100:g}%")
    _scale = alt.Scale(domain=list(_metrics), range=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
    _chart = (
        alt.Chart(_long)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=alt.X("cutoff:Q", scale=alt.Scale(type="log"), title="Frequency cutoff (log)", axis=alt.Axis(format="%")),
            y=alt.Y("% of QC-passing:Q", title="% of QC-passing variants below cutoff", scale=alt.Scale(zero=False)),
            color=alt.Color("metric:N", scale=_scale, title="Frequency column"),
            tooltip=["metric", "cutoff label", alt.Tooltip("variants below cutoff:Q", format=","), "% of QC-passing"],
        )
        .properties(width=640, height=280, title=f"Rare-set size vs. cutoff ({n_qc:,} QC-passing variants)")
    )
    sweep = _long.pivot(index="cutoff label", columns="metric", values="variants below cutoff").reindex(
        [f"{c*100:g}%" for c in _cuts]).reset_index()
    mo.vstack([_chart, mo.accordion({"Table view": mo.ui.table(sweep, selection=None)})])
    return (sweep,)


@app.cell(hide_code=True)
def _(CHROMS, mo):
    _freq_opts = {"0.01%": 1e-4, "0.1%": 1e-3, "0.5%": 5e-3, "1%": 1e-2, "2%": 2e-2, "5%": 5e-2}
    ex_chrom = mo.ui.dropdown(options=["all loaded"] + CHROMS, value="chr22" if "chr22" in CHROMS else "all loaded",
                              label="Chromosome")
    ex_rare_metric = mo.ui.dropdown(
        options={"f_max: global, grpmax & any group ≥2,000 alleles": "f_max",
                 "AF_grpmax": "coalesce(AF_grpmax,0)", "AF (global)": "coalesce(AF,0)"},
        value="f_max: global, grpmax & any group ≥2,000 alleles", label="Rare is judged on")
    ex_rare_cut = mo.ui.dropdown(options=_freq_opts, value="1%", label="Rare if <")
    ex_common_metric = mo.ui.dropdown(
        options={"global AF or faf95_max": "greatest(coalesce(AF,0), coalesce(faf95_max,0))",
                 "faf95_max only": "coalesce(faf95_max,0)", "AF (global) only": "coalesce(AF,0)",
                 "AF_grpmax (point estimate)": "greatest(coalesce(AF,0), coalesce(AF_grpmax,0))"},
        value="global AF or faf95_max", label="Common is judged on")
    ex_common_cut = mo.ui.dropdown(options=_freq_opts, value="1%", label="Common if ≥")
    ex_an = mo.ui.slider(0, 0.95, step=0.05, value=0.5, label="Min AN fraction", show_value=True)
    ex_pass = mo.ui.checkbox(value=True, label="FILTER = PASS")
    ex_lcr = mo.ui.checkbox(value=True, label="exclude lcr")
    ex_segdup = mo.ui.checkbox(value=True, label="exclude segdup")
    mo.vstack([
        mo.hstack([ex_chrom, ex_an, ex_pass, ex_lcr, ex_segdup], justify="start", wrap=True),
        mo.hstack([ex_rare_metric, ex_rare_cut], justify="start"),
        mo.hstack([ex_common_metric, ex_common_cut], justify="start"),
    ])
    return (ex_an, ex_chrom, ex_common_cut, ex_common_metric, ex_lcr, ex_pass, ex_rare_cut, ex_rare_metric, ex_segdup)


@app.cell(hide_code=True)
def _():
    def qc_sql(an_frac=0.5, require_pass=True, drop_lcr=True, drop_segdup=True):
        parts = [f"an_frac >= {an_frac}"]
        if require_pass:
            parts.append("is_pass")
        if drop_lcr:
            parts.append("NOT lcr")
        if drop_segdup:
            parts.append("NOT segdup")
        return " AND ".join(parts)

    def class_sql(qc, rare_expr="f_max", rare_cut=0.01,
                  common_expr="greatest(coalesce(AF,0), coalesce(faf95_max,0))", common_cut=0.01):
        return f"""CASE WHEN NOT ({qc}) THEN 'QC-excluded'
                        WHEN {common_expr} >= {common_cut} THEN 'common'
                        WHEN {rare_expr} < {rare_cut} THEN 'rare'
                        ELSE 'in-between / group-specific' END"""

    RECOMMENDED_QC = qc_sql()
    RECOMMENDED_CLASS = class_sql(RECOMMENDED_QC)
    return RECOMMENDED_CLASS, RECOMMENDED_QC, class_sql, qc_sql


@app.cell(hide_code=True)
def _(CLASS_ORDER, COLORS, alt, class_sql, con, ex_an, ex_chrom, ex_common_cut, ex_common_metric, ex_lcr, ex_pass,
      ex_rare_cut, ex_rare_metric, ex_segdup, mo, qc_sql):
    _qc = qc_sql(ex_an.value, ex_pass.value, ex_lcr.value, ex_segdup.value)
    _cls = class_sql(_qc, ex_rare_metric.value, ex_rare_cut.value, ex_common_metric.value, ex_common_cut.value)
    _where = "" if ex_chrom.value == "all loaded" else f"WHERE chrom = '{ex_chrom.value}'"
    with mo.status.spinner(title="Counting…"):
        _d = con.sql(f"""
            SELECT {_cls} AS class,
                   CASE WHEN allele_type = 'snv' THEN 'SNV' ELSE 'indel/other' END AS type,
                   count(*) AS variants
            FROM v {_where} GROUP BY 1, 2
        """).df()
    _tot = _d.groupby("class")["variants"].sum().reindex(CLASS_ORDER).fillna(0).astype("int64")
    _n = int(_tot.sum())
    _tiles = mo.hstack([
        mo.stat(value=f"{int(_tot[c]):,}", label=c, caption=f"{100 * _tot[c] / max(_n, 1):.2f}% of {_n:,}")
        for c in CLASS_ORDER
    ], widths="equal")
    _chart = (
        alt.Chart(_d)
        .mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=2)
        .encode(
            y=alt.Y("type:N", title=None),
            x=alt.X("variants:Q", title="Variants", stack="zero"),
            color=alt.Color("class:N", scale=alt.Scale(domain=CLASS_ORDER, range=[COLORS[c] for c in CLASS_ORDER]),
                            title="Class"),
            order=alt.Order("class_order:Q"),
            tooltip=["class", "type", alt.Tooltip("variants:Q", format=",")],
        )
        .transform_calculate(class_order=f"indexof({CLASS_ORDER}, datum.class)")
        .properties(width=620, height=110, title=f"{ex_chrom.value}: class counts by variant type")
    )
    mo.vstack([_tiles, _chart, mo.accordion({"Table view": mo.ui.table(_d, selection=None)}),
               mo.accordion({"SQL used": mo.md(f"```sql\n-- QC\n{_qc}\n-- class\n{_cls}\n```")})])
    return


@app.cell(hide_code=True)
def _(CHROMS, CLASS_ORDER, COLORS, OUT, RECOMMENDED_CLASS, alt, con, mo, pd):
    with mo.status.spinner(title="Classifying every variant…"):
        _d = con.sql(f"""
            WITH c AS (SELECT chrom, AC, AF, f_max, allele_type, {RECOMMENDED_CLASS} AS class FROM v)
            SELECT chrom,
                   count(*)                                                        AS total,
                   count(*) FILTER (WHERE class = 'QC-excluded')                   AS qc_excluded,
                   count(*) FILTER (WHERE class = 'rare')                          AS rare,
                   count(*) FILTER (WHERE class = 'rare' AND AC = 1)               AS rare_singleton,
                   count(*) FILTER (WHERE class = 'rare' AND f_max < 1e-4)         AS rare_lt_0_01pct,
                   count(*) FILTER (WHERE class = 'rare' AND f_max >= 1e-4 AND f_max < 1e-3) AS rare_0_01_to_0_1pct,
                   count(*) FILTER (WHERE class = 'rare' AND f_max >= 1e-3)        AS rare_0_1_to_1pct,
                   count(*) FILTER (WHERE class = 'in-between / group-specific')   AS in_between,
                   count(*) FILTER (WHERE class = 'common')                        AS common,
                   count(*) FILTER (WHERE class = 'common' AND AF < 0.05)          AS common_1_to_5pct_global,
                   count(*) FILTER (WHERE class = 'common' AND AF >= 0.05)         AS common_ge_5pct_global,
                   count(*) FILTER (WHERE class = 'common' AND AF > 0.5)           AS common_alt_is_major,
                   count(*) FILTER (WHERE class = 'rare' AND allele_type = 'snv')  AS rare_snv,
                   count(*) FILTER (WHERE class = 'common' AND allele_type = 'snv') AS common_snv
            FROM c GROUP BY chrom
        """).df()
    _order = {c: i for i, c in enumerate(CHROMS)}
    _d = _d.sort_values("chrom", key=lambda s: s.map(_order)).reset_index(drop=True)
    _tot = _d.drop(columns="chrom").sum()
    summary = pd.concat([_d, pd.DataFrame([{"chrom": "TOTAL", **_tot.to_dict()}])], ignore_index=True)
    summary.to_csv(OUT / "classification_summary.csv", index=False)

    _long = _d.melt(id_vars="chrom", value_vars=["rare", "common", "in_between", "qc_excluded"],
                    var_name="k", value_name="variants")
    _long["class"] = _long["k"].map({"rare": "rare", "common": "common", "in_between": "in-between / group-specific",
                                     "qc_excluded": "QC-excluded"})
    _long["share"] = _long["variants"] / _long.groupby("chrom")["variants"].transform("sum")
    _chart = (
        alt.Chart(_long)
        .mark_bar(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("chrom:N", sort=CHROMS, title=None),
            y=alt.Y("variants:Q", title="Variants"),
            color=alt.Color("class:N", scale=alt.Scale(domain=CLASS_ORDER, range=[COLORS[c] for c in CLASS_ORDER]),
                            title="Class"),
            order=alt.Order("class_order:Q"),
            tooltip=["chrom", "class", alt.Tooltip("variants:Q", format=","), alt.Tooltip("share:Q", format=".2%")],
        )
        .transform_calculate(class_order=f"indexof({CLASS_ORDER}, datum.class)")
        .properties(width=680, height=300, title="Recommended classification per chromosome")
    )
    _t = _tot
    _n = max(int(_t["total"]), 1)
    mo.vstack([
        mo.hstack([
            mo.stat(value=f"{int(_t['rare']):,}", label="rare", caption=f"{100*_t['rare']/_n:.1f}% of all variants"),
            mo.stat(value=f"{int(_t['common']):,}", label="common", caption=f"{100*_t['common']/_n:.1f}%"),
            mo.stat(value=f"{int(_t['in_between']):,}", label="in-between", caption=f"{100*_t['in_between']/_n:.2f}%"),
            mo.stat(value=f"{int(_t['qc_excluded']):,}", label="QC-excluded", caption=f"{100*_t['qc_excluded']/_n:.1f}%"),
        ], widths="equal"),
        _chart,
        mo.ui.table(summary, selection=None, page_size=30),
    ])
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    export_btn = mo.ui.run_button(label="Export rare + common variants")
    export_btn
    return (export_btn,)


@app.cell(hide_code=True)
def _(CHROMS, OUT, RECOMMENDED_CLASS, con, export_btn, mo):
    mo.stop(not export_btn.value)
    _dir = OUT / "classified"
    _dir.mkdir(exist_ok=True)
    for _c in mo.status.progress_bar(CHROMS, title="Exporting"):
        con.execute(f"""
            COPY (
                SELECT * FROM (
                    SELECT chrom, pos, ref, alt, allele_type, AC, AN, AF, AF_grpmax, grpmax, faf95_max,
                           af_max_any_grp, f_max, {RECOMMENDED_CLASS} AS class
                    FROM v WHERE chrom = '{_c}')
                WHERE class IN ('rare', 'common') ORDER BY pos
            ) TO '{(_dir / f"{_c}.rare_common.parquet").as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """)
    mo.callout(mo.md(f"Wrote {len(CHROMS)} files to `{_dir}`"), kind="success")
    return


if __name__ == "__main__":
    app.run()
