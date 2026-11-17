import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Localization-disrupting variant calibration")


@app.cell(hide_code=True)
def _():
    import importlib.util
    import json
    import os
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    CONTROL_SETS = ["pos_engineered", "pos_observed", "neg_matched", "null"]
    TEST_SETS = ["gtex_candidate", "gnomad_rare", "gnomad_common"]
    METRICS = {
        "max |Δlogit| (any compartment)": "max_abs_dlogit",
        "L2 norm of Δlogit": "l2_dlogit",
        "Jensen–Shannon divergence": "jsd",
        "z vs same-transcript null": "z_max",
    }
    return CONTROL_SETS, METRICS, Path, SLOT, TEST_SETS, alt, importlib, json, mo, np, os, pd, re


@app.cell(hide_code=True)
def _(Path, json, mo, os, pd):
    HERE = Path(mo.notebook_dir() or ".")
    OUT = Path(os.environ.get("LOCVAR_DATA", str(HERE))) / "loc_out"
    mo.stop(not (OUT / "scores.parquet").exists(), mo.callout(mo.md(f"No `{OUT / 'scores.parquet'}`"), kind="warn"))
    META = json.loads((OUT / "score_meta.json").read_text())
    raw = pd.read_parquet(OUT / "scores.parquet")
    sets_all = pd.read_parquet(OUT / "variant_sets.parquet")
    COMPS = META["compartments"]
    _warn = mo.callout(mo.md("**DEMO MODEL**"), kind="danger") if META.get("is_demo") else mo.md("")
    _warn
    return COMPS, HERE, META, OUT, raw, sets_all


@app.cell(hide_code=True)
def _(COMPS, np, pd, raw):
    _eps = 1e-6
    _lg = lambda p: np.log(np.clip(p, _eps, 1 - _eps) / (1 - np.clip(p, _eps, 1 - _eps)))
    W = raw[[f"wt_{c}" for c in COMPS]].to_numpy()
    M = raw[[f"mut_{c}" for c in COMPS]].to_numpy()
    D = _lg(M) - _lg(W)
    df = raw.copy()
    for _k, _c in enumerate(COMPS):
        df[f"dlogit_{_c}"] = D[:, _k]
    df["max_abs_dlogit"] = np.abs(D).max(axis=1)
    _arg = np.abs(D).argmax(axis=1)
    df["top_compartment"] = [COMPS[i] for i in _arg]
    df["top_direction"] = np.where(D[np.arange(len(D)), _arg] > 0, "more", "less")
    df["l2_dlogit"] = np.sqrt((D ** 2).sum(axis=1))
    _P = W / W.sum(1, keepdims=True)
    _Q = M / M.sum(1, keepdims=True)
    _Mid = (_P + _Q) / 2
    _kl = lambda a, b: (a * np.log2(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None))).sum(1)
    df["jsd"] = 0.5 * _kl(_P, _Mid) + 0.5 * _kl(_Q, _Mid)
    _null = df[df["set"] == "null"].groupby("transcript_id")["max_abs_dlogit"].agg(["mean", "std", "size"])
    _g_mean, _g_std = df.loc[df["set"] == "null", "max_abs_dlogit"].agg(["mean", "std"]) if (df["set"] == "null").any() else (0.0, 1.0)
    _m = df["transcript_id"].map(_null["mean"])
    _s = df["transcript_id"].map(_null["std"])
    _n = df["transcript_id"].map(_null["size"]).fillna(0)
    _use = (_n >= 10) & (_s > 0)
    df["z_max"] = np.where(_use, (df["max_abs_dlogit"] - _m) / _s, (df["max_abs_dlogit"] - _g_mean) / (_g_std or 1.0))
    df["z_source"] = np.where(_use, "own transcript", "pooled null")
    scored = df[df["in_window"]].copy()
    return D, scored


@app.cell(hide_code=True)
def _(mo, scored, sets_all):
    _q = sets_all.groupby("set").agg(prepared=("variant", "size"), ref_matches=("ref_ok", "mean"),
                                     inside_model_window=("in_window", "mean")).reset_index()
    _q["scored (ref ok + in window)"] = _q["set"].map(scored.groupby("set").size()).fillna(0).astype(int)
    _q["ref_matches"] = (100 * _q["ref_matches"]).round(2).astype(str) + "%"
    _q["inside_model_window"] = (100 * _q["inside_model_window"]).round(1).astype(str) + "%"
    mo.ui.table(_q, selection=None)
    return


@app.cell(hide_code=True)
def _(METRICS, mo):
    dist_metric = mo.ui.dropdown(options=METRICS, value="max |Δlogit| (any compartment)", label="Score")
    dist_metric
    return (dist_metric,)


@app.cell(hide_code=True)
def _(CONTROL_SETS, SLOT, TEST_SETS, alt, dist_metric, mo, np, pd, scored):
    def _ecdf(sets, colors, title):
        rows = []
        for s in sets:
            v = np.sort(scored.loc[scored["set"] == s, dist_metric.value].dropna().to_numpy())
            if len(v) == 0:
                continue
            q = np.linspace(0, 1, 200)
            rows.append(pd.DataFrame({"set": s, "score": np.quantile(v, q), "fraction ≤ score": q, "n": len(v)}))
        if not rows:
            return mo.md(f"_No data for {title}_")
        d = pd.concat(rows)
        return (alt.Chart(d).mark_line(strokeWidth=2)
                .encode(x=alt.X("score:Q", title=dist_metric.selected_key),
                        y=alt.Y("fraction ≤ score:Q", title="Cumulative fraction"),
                        color=alt.Color("set:N", scale=alt.Scale(domain=sets, range=colors), title=None),
                        tooltip=["set", alt.Tooltip("score:Q", format=".4f"), alt.Tooltip("fraction ≤ score:Q", format=".2f"), "n"])
                .properties(width=330, height=260, title=title))
    mo.hstack([_ecdf(CONTROL_SETS, SLOT[:4], "Controls"), _ecdf(TEST_SETS, SLOT[4:7], "Test sets")], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    pos_choice = mo.ui.multiselect(options=["pos_engineered", "pos_observed"], value=["pos_engineered", "pos_observed"], label="Positives")
    neg_choice = mo.ui.dropdown(options=["neg_matched", "null"], value="neg_matched", label="Negatives")
    mo.hstack([pos_choice, neg_choice], justify="start")
    return neg_choice, pos_choice


@app.cell(hide_code=True)
def _(METRICS, SLOT, alt, mo, neg_choice, np, pd, pos_choice, scored):
    def roc_pr(pos, neg):
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[pos, neg]
        o = np.argsort(-s, kind="mergesort")
        y, s = y[o], s[o]
        tp, fp = np.cumsum(y), np.cumsum(1 - y)
        tpr, fpr = tp / max(len(pos), 1), fp / max(len(neg), 1)
        prec = tp / np.maximum(tp + fp, 1)
        auroc = np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]) if hasattr(np, "trapezoid") else np.trapz(np.r_[0, tpr], np.r_[0, fpr])
        auprc = float(np.sum(np.diff(np.r_[0, tpr]) * prec))
        return fpr, tpr, prec, auroc, auprc

    mo.stop(not pos_choice.value, mo.md("_Pick at least one positive set._"))
    _P = scored[scored["set"].isin(pos_choice.value)]
    _N = scored[scored["set"] == neg_choice.value]
    mo.stop(len(_P) == 0 or len(_N) == 0, mo.callout(mo.md("Need both positives and negatives with scores."), kind="warn"))
    _curves, _summ = [], []
    for _lab, _col in METRICS.items():
        fpr, tpr, prec, au, ap = roc_pr(_P[_col].to_numpy(), _N[_col].to_numpy())
        _k = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
        _curves.append(pd.DataFrame({"score": _lab, "FPR": fpr[_k], "TPR": tpr[_k], "precision": prec[_k]}))
        _summ.append({"score": _lab, "AUROC": round(au, 3), "AUPRC": round(ap, 3),
                      "positives": len(_P), "negatives": len(_N), "base rate": round(len(_P) / (len(_P) + len(_N)), 3)})
    _c = pd.concat(_curves)
    _scale = alt.Scale(domain=list(METRICS), range=SLOT[:4])
    _roc = (alt.Chart(_c).mark_line(strokeWidth=2).encode(
        x=alt.X("FPR:Q", title="False-positive rate"), y=alt.Y("TPR:Q", title="True-positive rate"),
        color=alt.Color("score:N", scale=_scale, title=None),
        tooltip=["score", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")])
        + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 3], color="#b5b3ad").encode(x="x", y="y")
    ).properties(width=320, height=280, title="ROC")
    _pr = alt.Chart(_c).mark_line(strokeWidth=2).encode(
        x=alt.X("TPR:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("score:N", scale=_scale, title=None),
        tooltip=["score", alt.Tooltip("TPR:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")]
    ).properties(width=320, height=280, title="Precision–recall")
    _per_motif = []
    for _m, _d in _P.groupby("motif"):
        for _lab, _col in METRICS.items():
            _per_motif.append({"motif": _m, "score": _lab, "AUROC": round(roc_pr(_d[_col].to_numpy(), _N[_col].to_numpy())[3], 3), "n": len(_d)})
    _pm = pd.DataFrame(_per_motif).pivot(index=["motif", "n"], columns="score", values="AUROC").reset_index() if _per_motif else pd.DataFrame()
    mo.vstack([mo.hstack([_roc, _pr], justify="start"), mo.ui.table(pd.DataFrame(_summ), selection=None),
               mo.md("**AUROC by motif**"),
               mo.ui.table(_pm, selection=None)])
    return (roc_pr,)


@app.cell(hide_code=True)
def _(METRICS, mo):
    call_metric = mo.ui.dropdown(options=METRICS, value="z vs same-transcript null", label="Score used for calling")
    call_fpr = mo.ui.dropdown(options={"1%": 0.01, "2%": 0.02, "5%": 0.05, "10%": 0.10}, value="5%", label="False-positive rate")
    mo.hstack([call_metric, call_fpr], justify="start")
    return call_fpr, call_metric


@app.cell(hide_code=True)
def _(SLOT, alt, call_fpr, call_metric, mo, np, pd, scored):
    _neg = scored.loc[scored["set"] == "neg_matched", call_metric.value].dropna()
    mo.stop(len(_neg) < 20, mo.callout(mo.md("Too few matched negatives to set a threshold."), kind="warn"))
    THRESH = float(np.quantile(_neg, 1 - call_fpr.value))
    calls = scored.assign(called=scored[call_metric.value] >= THRESH)

    def _rate(d):
        n, k = len(d), int(d["called"].sum())
        p = k / n if n else np.nan
        se = np.sqrt(p * (1 - p) / n) if n else np.nan
        return pd.Series({"variants": n, "called": k, "call rate": p, "lo": max(p - 1.96 * se, 0), "hi": p + 1.96 * se})
    _by_set = calls.groupby("set").apply(_rate, include_groups=False).reset_index()
    _order = ["pos_engineered", "pos_observed", "gtex_candidate", "gnomad_rare", "gnomad_common", "neg_matched", "null"]
    _by_set = _by_set.set_index("set").reindex([s for s in _order if s in set(_by_set["set"])]).reset_index()
    _bars = alt.Chart(_by_set).mark_bar(color=SLOT[0], cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("set:N", sort=_order, title=None), x=alt.X("call rate:Q", axis=alt.Axis(format="%"), title="Fraction called"),
        tooltip=["set", "variants", "called", alt.Tooltip("call rate:Q", format=".2%")])
    _err = alt.Chart(_by_set).mark_rule(color="#52514e").encode(y=alt.Y("set:N", sort=_order), x="lo:Q", x2="hi:Q")
    _fpr_rule = alt.Chart(pd.DataFrame({"x": [call_fpr.value]})).mark_rule(strokeDash=[4, 3], color="#e34948").encode(x="x:Q")

    _g = calls[calls["set"].isin(["gnomad_rare", "gnomad_common", "neg_matched"]) & calls["AF"].notna()].copy()
    _g["AF bin"] = pd.cut(_g["AF"], [0, 1e-4, 1e-3, 1e-2, 0.05, 1.0], labels=["<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "≥5%"])
    _af = _g.groupby("AF bin", observed=True).apply(_rate, include_groups=False).reset_index()
    _afc = (alt.Chart(_af).mark_line(color=SLOT[1], strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
        x=alt.X("AF bin:N", sort=["<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "≥5%"], title="gnomAD allele frequency"),
        y=alt.Y("call rate:Q", axis=alt.Axis(format="%"), title="Fraction called"),
        tooltip=["AF bin", "variants", "called", alt.Tooltip("call rate:Q", format=".2%")])
        + alt.Chart(_af).mark_rule(color=SLOT[1]).encode(x=alt.X("AF bin:N", sort=["<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "≥5%"]), y="lo:Q", y2="hi:Q")
    ).properties(width=320, height=240, title="Call rate by allele frequency (depletion test)")
    mo.vstack([
        mo.md(f"**Threshold:** {call_metric.selected_key} ≥ **{THRESH:.4g}** (calls {call_fpr.selected_key} of matched negatives)."),
        mo.hstack([(_bars + _err + _fpr_rule).properties(width=340, height=230, title="Call rate by set (dashed = target FPR)"), _afc],
                  justify="start"),
    ])
    return THRESH, calls


@app.cell(hide_code=True)
def _(OUT, call_metric, calls, mo):
    hits = (calls[calls["called"] & calls["set"].isin(["gtex_candidate", "gnomad_rare"])]
            .sort_values(call_metric.value, ascending=False))
    _cols = ["set", "variant", "gene", "region", "transcript_id", "tx_start", "tx_ref", "tx_alt", "dist_to_3p_end",
             call_metric.value, "top_compartment", "top_direction", "max_pip", "best_tissue", "AF", "f_max"]
    hits_view = hits[[c for c in _cols if c in hits.columns]].reset_index(drop=True)
    hits_view.to_csv(OUT / "called_variants.csv", index=False)
    mo.vstack([mo.md(f"**{len(hits_view):,}** called (saved to `loc_out/called_variants.csv`)."),
               mo.ui.table(hits_view, selection=None, page_size=12)])
    return (hits_view,)


@app.cell(hide_code=True)
def _(hits_view, mo):
    _opts = {f"{r.gene} · {r.variant} · {r.set}": r.variant for r in hits_view.head(60).itertuples()} if len(hits_view) else {}
    case_pick = mo.ui.dropdown(options=_opts, value=next(iter(_opts), None), label="Variant")
    ism_flank = mo.ui.slider(10, 60, step=5, value=30, label="ISM flank (nt each side)", show_value=True)
    run_ism = mo.ui.run_button(label="Run ISM")
    mo.hstack([case_pick, ism_flank, run_ism], justify="start")
    return case_pick, ism_flank, run_ism


@app.cell(hide_code=True)
def _(COMPS, SLOT, alt, case_pick, mo, pd, scored):
    mo.stop(case_pick.value is None, mo.md("_No called variants yet._"))
    case = scored[scored["variant"] == case_pick.value].iloc[0]
    _d = pd.DataFrame({"compartment": COMPS, "WT": [case[f"wt_{c}"] for c in COMPS], "MUT": [case[f"mut_{c}"] for c in COMPS],
                       "Δlogit": [case[f"dlogit_{c}"] for c in COMPS]})
    _bar = alt.Chart(_d).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("compartment:N", title=None), x=alt.X("Δlogit:Q", title="Δ log-odds (MUT − WT)"),
        color=alt.condition("datum['Δlogit'] > 0", alt.value(SLOT[0]), alt.value(SLOT[1])),
        tooltip=["compartment", alt.Tooltip("WT:Q", format=".3f"), alt.Tooltip("MUT:Q", format=".3f"), alt.Tooltip("Δlogit:Q", format=".3f")]
    ).properties(width=320, height=150, title="Predicted change by compartment")
    _info = {k: case.get(k) for k in ["variant", "set", "gene", "region", "transcript_id", "tx_start", "tx_ref", "tx_alt",
                                      "dist_to_3p_end", "max_pip", "best_tissue", "AF", "motif", "z_max", "z_source"] if k in case.index}
    mo.hstack([mo.ui.table(pd.DataFrame({"field": list(_info), "value": [str(v) for v in _info.values()]}), selection=None),
               _bar], justify="start", align="start")
    return (case,)


@app.cell(hide_code=True)
def _(COMPS, HERE, META, OUT, SLOT, alt, case, importlib, ism_flank, mo, np, pd, re, run_ism):
    mo.stop(not run_ism.value)
    _spec = importlib.util.spec_from_file_location("model_adapter", HERE / "model_adapter.py")
    A = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(A)
    _seq = pd.read_parquet(OUT / "transcripts.parquet", columns=["transcript_id", "seq"]).set_index("transcript_id").at[case["transcript_id"], "seq"]

    def _crop(s):
        if A.MAX_LEN and len(s) > A.MAX_LEN:
            return s[-A.MAX_LEN:] if A.CROP == "3p" else s[:A.MAX_LEN]
        return s
    _p0 = int(case["tx_start"])
    _lo, _hi = max(1, _p0 - ism_flank.value), min(len(_seq), _p0 + ism_flank.value)
    _muts, _meta = [], []
    for _p in range(_lo, _hi + 1):
        for _b in "ACGT":
            if _b != _seq[_p - 1]:
                _muts.append(_crop(_seq[:_p - 1] + _b + _seq[_p:]))
                _meta.append((_p, _b))
    with mo.status.spinner(title=f"Scoring {len(_muts) + 1} sequences with the model…"):
        _m = A.load()
        _wt = np.asarray(A.predict(_m, [_crop(_seq)]), dtype=float)
        _out = np.vstack([np.asarray(A.predict(_m, _muts[i:i + A.BATCH_SIZE]), dtype=float) for i in range(0, len(_muts), A.BATCH_SIZE)])
    if A.OUTPUT == "logit":
        _wt, _out = 1 / (1 + np.exp(-_wt)), 1 / (1 + np.exp(-_out))
    _lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    _eff = np.abs(_lg(_out) - _lg(_wt)).max(axis=1)
    ism = pd.DataFrame(_meta, columns=["tx_pos", "base"]).assign(effect=_eff)
    ism["ref base"] = [_seq[p - 1] for p in ism["tx_pos"]]
    ism["is_variant"] = (ism["tx_pos"] == _p0) & (ism["base"] == case["tx_alt"])
    _heat = alt.Chart(ism).mark_rect().encode(
        x=alt.X("tx_pos:O", title="Transcript position", axis=alt.Axis(labelOverlap=True)),
        y=alt.Y("base:N", sort=list("ACGT"), title="Mutant base"),
        color=alt.Color("effect:Q", scale=alt.Scale(scheme="blues"), title="max |Δlogit|"),
        tooltip=["tx_pos", "ref base", "base", alt.Tooltip("effect:Q", format=".4f")])
    _mark = alt.Chart(ism[ism["is_variant"]]).mark_rect(fill=None, stroke=SLOT[1], strokeWidth=2).encode(
        x="tx_pos:O", y=alt.Y("base:N", sort=list("ACGT")))
    _ctx = _seq[_lo - 1:_hi]
    _ps = importlib.util.spec_from_file_location("loc_pipeline", HERE / "loc_pipeline.py")
    _lp = importlib.util.module_from_spec(_ps)
    _ps.loader.exec_module(_lp)
    _MOT = _lp.MOTIFS
    _hits = [f"{n} at {_lo + m.start()}" for n, d in _MOT.items() for m in re.finditer(f"(?=({d['regex']}))", _ctx)]
    mo.vstack([
        (_heat + _mark).properties(width=760, height=130, title="In-silico mutagenesis around the variant (orange box = the variant)"),
        mo.md(f"Context (transcript {_lo}–{_hi}): `{_ctx}`  \nCurated motif hits in window: {', '.join(_hits) or 'none'}"),
    ])
    return (ism,)


@app.cell(hide_code=True)
def _(mo):
    save_case = mo.ui.run_button(label="Save case study")
    save_case
    return (save_case,)


@app.cell(hide_code=True)
def _(COMPS, OUT, case, ism, mo, save_case):
    mo.stop(not save_case.value)
    _d = OUT / "case_studies"
    _d.mkdir(exist_ok=True)
    _name = str(case["variant"]).replace(":", "_").replace(">", "-")
    ism.to_csv(_d / f"{_name}.ism.csv", index=False)
    _top = ism.sort_values("effect", ascending=False).head(10)
    _lines = [f"# Case study: {case.get('gene')} {case['variant']}", "",
              f"- Set: {case['set']}  ·  region: {case.get('region')}  ·  transcript {case['transcript_id']} pos {case['tx_start']} "
              f"({case['tx_ref']}>{case['tx_alt']}), {case.get('dist_to_3p_end')} nt from 3' end",
              f"- GTEx max PIP: {case.get('max_pip')} ({case.get('best_tissue')})  ·  gnomAD AF: {case.get('AF')}",
              f"- z vs same-transcript null: {case['z_max']:.2f} ({case['z_source']})", "",
              "| compartment | WT | MUT | Δlogit |", "|---|---|---|---|"]
    _lines += [f"| {c} | {case[f'wt_{c}']:.4f} | {case[f'mut_{c}']:.4f} | {case[f'dlogit_{c}']:+.4f} |" for c in COMPS]
    _lines += ["", "Strongest ISM positions:", "", "| pos | ref | mut | effect |", "|---|---|---|---|"]
    _lines += [f"| {r.tx_pos} | {r._4} | {r.base} | {r.effect:.4f} |" for r in _top.itertuples()]
    (_d / f"{_name}.md").write_text("\n".join(_lines))
    mo.callout(mo.md(f"Saved `{_d / (_name + '.md')}` and the ISM table."), kind="success")
    return


if __name__ == "__main__":
    app.run()
