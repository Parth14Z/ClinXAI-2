"""ClinXAI - Streamlit dashboard.

Launch:  streamlit run app.py
Requires the artefacts produced by `python train_model.py` and `python evaluate.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src import lime_explainer as lx
from src import plotting as viz
from src import shap_explainer as sx
from src import ui
from src.comparison import compare, rank_features
from src.data import format_patient_id, load_split_csv
from src.model import load_metadata, load_pipeline, transform_features
from src.preprocessing import imputed_mask, validate_patient_input

st.set_page_config(page_title="ClinXAI · Diabetes XAI", page_icon="\U0001FA7A", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PAGES = [
    "Home & clinical overview",
    "Diabetes prediction",
    "LIME patient explanation",
    "SHAP explainability",
    "Explanation comparison",
    "Experimental evaluation",
    "Case study explorer",
]
PAGE_ICONS = ["\U0001F3E0", "\U0001FA7A", "\U0001F52C", "\U0001F9EE", "⚖️", "\U0001F4CA", "\U0001F4C1"]


# --------------------------------------------------------------------------- #
# Resources (cached)
# --------------------------------------------------------------------------- #
@dataclass
class Resources:
    xgb: object
    meta: dict
    X_train: pd.DataFrame
    X_train_t: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    ids_test: np.ndarray
    probs_test: np.ndarray
    tree: object
    shap_test: object
    pos: dict


@st.cache_resource(show_spinner="Loading model and explainers…")
def load_resources() -> Resources:
    xgb = load_pipeline(config.XGB_MODEL_PATH)
    X_train, _, _ = load_split_csv(config.TRAIN_CSV)
    X_test, y_test, ids_test = load_split_csv(config.TEST_CSV)
    tree = sx.build_tree_explainer(xgb)
    return Resources(
        xgb=xgb, meta=load_metadata(), X_train=X_train, X_train_t=transform_features(xgb, X_train),
        X_test=X_test, y_test=y_test, ids_test=ids_test,
        probs_test=xgb.predict_proba(X_test)[:, 1], tree=tree,
        shap_test=sx.explain(tree, xgb, X_test), pos={int(p): i for i, p in enumerate(ids_test)},
    )


# A fresh LIME explainer per call: the explainer's random state advances with every explanation,
# so a shared explainer would make a patient's result depend on which patients were viewed before.
@st.cache_data(show_spinner="Running LIME perturbations…", max_entries=48)
def lime_for(values: tuple, seed: int, num_samples: int) -> lx.LimeResult:
    R = load_resources()
    x_t = transform_features(R.xgb, pd.DataFrame([values], columns=config.FEATURES))
    return lx.explain_patient(R.xgb, lx.build_lime_explainer(R.X_train_t, seed), x_t, num_samples, seed,
                              keep_neighborhood=True)


@st.cache_data(show_spinner="Re-running LIME across seeds…", max_entries=48)
def stability_for(values: tuple, seeds: tuple, num_samples: int) -> lx.StabilityResult:
    R = load_resources()
    x_t = transform_features(R.xgb, pd.DataFrame([values], columns=config.FEATURES))
    return lx.lime_stability(R.xgb, R.X_train_t, x_t, seeds, num_samples)


@st.cache_data
def load_json(path: str):
    p = config.PROJECT_ROOT / path
    return json.loads(p.read_text("utf-8")) if p.exists() else None


@st.cache_data
def load_csv(path: str):
    p = config.PROJECT_ROOT / path
    return pd.read_csv(p) if p.exists() else None


def show_fig(make_fig, *args, **kwargs) -> None:
    """Build and render a matplotlib figure without interleaving with other users' sessions."""
    with viz.exclusive_figure(make_fig, *args, **kwargs) as fig:
        st.pyplot(fig, width="stretch")


def show_image(rel_path: str, caption: str | None = None) -> None:
    p = config.PROJECT_ROOT / rel_path
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.warning(f"Missing figure `{rel_path}` - run `python evaluate.py`.")


# --------------------------------------------------------------------------- #
# Current patient profile (held-out patient or what-if edit)
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    x_raw: pd.DataFrame
    patient_id: int
    label: str
    y_true: int | None
    hypothetical: bool

    @property
    def key(self) -> tuple:
        return tuple(float(v) for v in self.x_raw.iloc[0].to_numpy())


def case_lookup() -> dict[int, str]:
    cases = load_json("outputs/case_studies/case_studies.json") or []
    return {c["patient_id"]: f"Case {c['case_id']}" for c in cases if c.get("available")}


def current_profile(R: Resources) -> Profile:
    pid = st.session_state.get("patient_id", int(R.ids_test[0]))
    i = R.pos[pid]
    whatif = st.session_state.get("whatif")
    if whatif:
        return Profile(validate_patient_input(whatif), pid,
                       f"What-if profile (edited from {format_patient_id(pid)})", None, True)
    return Profile(R.X_test.iloc[[i]].reset_index(drop=True), pid, format_patient_id(pid),
                   int(R.y_test.iloc[i]), False)


def clear_whatif() -> None:
    st.session_state["whatif"] = None


def open_patient(pid: int) -> None:
    st.session_state["patient_id"] = pid
    st.session_state["whatif"] = None


def profile_banner(prof: Profile, prob: float) -> None:
    truth = ("" if prof.y_true is None else
             ui.pill("Recorded outcome: positive" if prof.y_true else "Recorded outcome: negative",
                     "pos" if prof.y_true else "neg"))
    pred = ui.pill("Model class: positive" if prob >= config.DECISION_THRESHOLD else "Model class: negative",
                   "pos" if prob >= config.DECISION_THRESHOLD else "neg")
    kind = ui.pill("Hypothetical edit", "neutral") if prof.hypothetical else ui.pill("Held-out patient", "neutral")
    st.markdown(f"**{ui.esc(prof.label)}** &nbsp; {kind}{pred}{truth} "
                f"<span style='color:#5B6675'>P(positive) = {prob:.3f}</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(R: Resources) -> str:
    with st.sidebar:
        st.markdown(f"<div style='font-size:1.45rem;font-weight:700;color:{config.PALETTE['teal_dark']}'>"
                    f"ClinXAI</div><div style='color:#5B6675;font-size:.85rem;margin-bottom:10px'>"
                    f"Transparent diabetes classification<br>XGBoost · LIME · SHAP</div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, key="nav",
                        format_func=lambda p: f"{PAGE_ICONS[PAGES.index(p)]}  {p}")
        st.divider()
        cases = case_lookup()
        order = sorted(R.ids_test.tolist(), key=lambda p: (p not in cases, cases.get(p, ""), p))
        st.session_state.setdefault("patient_id", order[0])
        st.session_state.setdefault("whatif", None)

        def fmt(pid):
            i = R.pos[pid]
            tag = f" · {cases[pid]}" if pid in cases else ""
            return f"{format_patient_id(pid)}{tag} · p={R.probs_test[i]:.2f}"

        st.selectbox("Held-out patient (test set)", order, key="patient_id", format_func=fmt,
                     on_change=clear_whatif,
                     help="154 patients never used for training or model selection. "
                          "Case-study patients are listed first.")
        if st.session_state.get("whatif"):
            st.info("A what-if edit is active on the explanation pages.")
            st.button("Reset to recorded values", on_click=clear_whatif, width="stretch")
        st.divider()
        st.caption("Educational demonstration on a historical research dataset. "
                   "Not a diagnostic tool; not medical advice.")
    return page


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home(R: Resources) -> None:
    st.markdown(ui.hero("Trustworthy & Responsible AI · ADCOR1PC303 · Module 1",
                        "ClinXAI: Transparent Multi-Modal Clinical Diagnostics Using LIME and SHAP",
                        "An XGBoost diabetes classifier on the Pima Indians Diabetes Database, "
                        "explained patient-by-patient with LIME and TreeSHAP - and a study of when the "
                        "two explanation methods agree."), unsafe_allow_html=True)
    st.markdown(ui.note(ui.DISCLAIMER, "disclaimer"), unsafe_allow_html=True)

    mm = load_json("outputs/metrics/model_metrics.json")
    study = load_json("outputs/metrics/explanation_study_summary.json")
    c = st.columns(4)
    c[0].markdown(ui.kpi("Records", "768", "500 negative · 268 positive"), unsafe_allow_html=True)
    c[1].markdown(ui.kpi("Held-out test set", f"{len(R.X_test)}", "stratified 20%, seed 42"),
                  unsafe_allow_html=True)
    if mm:
        x = mm["xgboost"]
        c[2].markdown(ui.kpi("XGBoost test ROC-AUC", f"{x['roc_auc']:.3f}",
                             f"95% CI {x['roc_auc_ci95'][0]:.2f}-{x['roc_auc_ci95'][1]:.2f}"),
                      unsafe_allow_html=True)
    if study:
        c[3].markdown(ui.kpi("LIME-SHAP top-3 overlap", f"{study['mean_top3_overlap']:.2f}",
                             f"mean over {study['n_patients']} test patients"), unsafe_allow_html=True)

    st.subheader("Project objective")
    a, b, d = st.columns(3)
    a.markdown(ui.card("Predict", "<p>Estimate a score for the diabetes-positive class from eight routine "
                                  "tabular measurements using a gradient-boosted tree ensemble (XGBoost).</p>"),
               unsafe_allow_html=True)
    b.markdown(ui.card("Explain", "<p>Explain each prediction locally with LIME (perturbation + linear "
                                  "surrogate) and with TreeSHAP (exact Shapley values for trees), plus global "
                                  "SHAP importance.</p>"), unsafe_allow_html=True)
    d.markdown(ui.card("Interrogate", "<p>Measure how consistently the two methods explain the same "
                                      "prediction across confidence levels: overlap, rank correlation, "
                                      "direction agreement, surrogate fidelity and seed stability.</p>"),
               unsafe_allow_html=True)

    st.subheader("What the model classifies")
    st.markdown(
        "The outcome is the dataset's binary label **`tested_positive`** (class 1) versus "
        "**`tested_negative`** (class 0). In the source documentation, positive means the patient showed "
        "signs of diabetes by World Health Organization criteria - a 2-hour post-load plasma glucose of at "
        "least 200 mg/dL at a survey examination, or diabetes found during routine medical care. The model "
        "outputs an **uncalibrated score** for class 1; scores at or above 0.5 are labelled positive. "
        "Because the label is partly defined by a glucose-tolerance measurement, a dominant role for "
        "`Glucose` is expected rather than a discovery.")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Data source & provenance")
        prov = load_json("data/raw/provenance.json") or {}
        st.markdown(
            f"- **Dataset:** Pima Indians Diabetes Database, obtained from **OpenML dataset 37** "
            f"(`{config.OPENML_ARFF_URL}`)\n"
            f"- **Original owner:** {prov.get('original_owner', 'NIDDK')}; donated to the UCI repository by "
            f"V. Sigillito (Johns Hopkins APL), 1990\n"
            f"- **Integrity:** MD5 `{prov.get('md5_actual', '?')}` matches OpenML's published checksum\n"
            f"- **Licence field on OpenML:** {prov.get('openml_licence_field', 'Public')}\n"
            f"- **Population:** {prov.get('population', '')}")
        st.markdown(ui.note(
            "<b>Limited population.</b> All 768 records are women aged 21+ of Pima heritage from one "
            "community near Phoenix, Arizona, collected decades ago. Results must not be generalised to men, "
            "other ancestries, other age groups or today's clinical populations."), unsafe_allow_html=True)
    with right:
        st.subheader("Data quality: zero placeholders")
        show_image("outputs/figures/dataset_overview.png")
        st.caption("The documentation states 'no missing values', but zeros for glucose, blood pressure, "
                   "skin fold, insulin and BMI are physiologically impossible. They are converted to missing "
                   "and imputed with medians learned from the training split only.")

    st.subheader("Feature definitions (from the source documentation)")
    st.dataframe(pd.DataFrame([{"Feature": f, "Definition": d, "Unit": u,
                                "Zero treated as missing": "yes" if f in config.ZERO_AS_MISSING else "no"}
                               for f, (d, u) in config.FEATURE_DEFINITIONS.items()]),
                 hide_index=True, width="stretch")

    with st.expander("System architecture"):
        show_image("outputs/figures/architecture.png")
    with st.expander("About the word “Multi-Modal” in the assigned title"):
        st.markdown(
            "This prototype uses **one data modality: tabular clinical measurements**. It does not process "
            "medical images, waveforms, free text or any other modality. What it does combine are two "
            "explanation paradigms (perturbation-based LIME and game-theoretic SHAP) and several "
            "presentation modes (charts, tables and plain-language summaries). The title is the assigned "
            "course title and is not a claim of multi-modal data fusion.")


def page_prediction(R: Resources) -> None:
    st.header("Diabetes prediction")
    prof = current_profile(R)
    prob = float(R.xgb.predict_proba(prof.x_raw)[0, 1])
    profile_banner(prof, prob)

    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown(ui.kpi("Model score · P(diabetes-positive)", f"{prob:.3f}",
                           "uncalibrated XGBoost output"), unsafe_allow_html=True)
        st.markdown(ui.probability_meter(prob), unsafe_allow_html=True)
        cls = "diabetes-positive (1)" if prob >= config.DECISION_THRESHOLD else "diabetes-negative (0)"
        st.markdown(f"**Model class at threshold {config.DECISION_THRESHOLD}:** {cls}")
        if prof.y_true is not None:
            st.markdown(f"**Recorded outcome in dataset:** "
                        f"{'diabetes-positive (1)' if prof.y_true else 'diabetes-negative (0)'}")
        margin = float(sx.predict_margin(R.xgb, prof.x_raw)[0])
        st.caption(f"Raw model margin (log-odds) = {margin:+.3f}; sigmoid(margin) = {sx.sigmoid(margin):.3f}.")
        st.markdown(ui.note(
            "<b>Uncalibrated probability.</b> This is the model's score for class 1, not a validated risk "
            "estimate. No calibration was fitted or validated, so a score of 0.7 should not be read as a 70% "
            "chance of disease."), unsafe_allow_html=True)
    with c2:
        mask = imputed_mask(prof.x_raw).iloc[0]
        model_in = transform_features(R.xgb, prof.x_raw).iloc[0]
        table = pd.DataFrame({
            "Feature": config.FEATURES,
            "Recorded value": [prof.x_raw.iloc[0][f] for f in config.FEATURES],
            "Value used by model": [model_in[f] for f in config.FEATURES],
            "Imputed": ["yes (train median)" if mask[f] else "" for f in config.FEATURES],
            "Unit": [config.FEATURE_DEFINITIONS[f][1].split(" (")[0] for f in config.FEATURES],
        })
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Recorded value": st.column_config.NumberColumn(format="%.3g"),
                                    "Value used by model": st.column_config.NumberColumn(format="%.3g")})

    st.subheader("What-if editor")
    st.caption("Edit values to explore model behaviour. A zero for Glucose, BloodPressure, SkinThickness, "
               "Insulin or BMI is treated as missing and imputed. Edits create a hypothetical profile and are "
               "explained on the LIME, SHAP and comparison pages.")
    base_vals = R.X_test.iloc[R.pos[prof.patient_id]]
    with st.form("whatif_form"):
        cols = st.columns(4)
        new_vals = {}
        for k, f in enumerate(config.FEATURES):
            lo, hi = config.INPUT_BOUNDS[f]
            is_float = f in ("BMI", "DiabetesPedigreeFunction")
            start = (st.session_state["whatif"] or {}).get(f, base_vals[f])
            new_vals[f] = cols[k % 4].number_input(
                f, min_value=float(lo), max_value=float(hi), value=float(start),
                step=0.01 if f == "DiabetesPedigreeFunction" else (0.1 if is_float else 1.0),
                format="%.3f" if f == "DiabetesPedigreeFunction" else ("%.1f" if is_float else "%.0f"),
                help=config.FEATURE_DEFINITIONS[f][0])
        submitted = st.form_submit_button("Apply what-if profile", type="primary")
    if submitted:
        try:
            validate_patient_input(new_vals)
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
        else:
            st.session_state["whatif"] = new_vals
            st.rerun()


def page_lime(R: Resources) -> None:
    st.header("LIME patient explanation")
    prof = current_profile(R)
    c1, c2 = st.columns([1, 1])
    num_samples = c1.select_slider("Perturbed samples", [1000, 2500, 5000, 10000], value=config.LIME_NUM_SAMPLES)
    seed = int(c2.number_input("LIME random seed", 0, 10_000, config.LIME_PRIMARY_SEED, step=1))
    res = lime_for(prof.key, seed, num_samples)
    profile_banner(prof, res.model_prob)

    k = st.columns(4)
    k[0].markdown(ui.kpi("XGBoost P(positive)", f"{res.model_prob:.3f}", "the value LIME explains"),
                  unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Surrogate prediction", f"{res.local_pred:.3f}",
                         f"gap {res.fidelity_gap:.3f} vs model"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Surrogate fidelity R²", f"{res.r2:.3f}",
                         "weighted, on the neighbourhood"), unsafe_allow_html=True)
    nb = res.neighborhood_summary()
    k[3].markdown(ui.kpi("Neighbourhood predicted positive", f"{100 * nb['share_predicted_positive']:.0f}%",
                         f"of {nb['n_perturbations']:,} perturbations"), unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        show_fig(viz.plot_lime_explanation, res.weights, res.rules, f"{prof.label}: LIME local explanation")
    with right:
        t = res.table()
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"lime_weight": st.column_config.NumberColumn("weight", format="%+.4f")})
        st.caption(f"Intercept {res.intercept:.3f}. Weights are coefficients of a weighted ridge regression "
                   f"on binary 'same quartile bin as the patient' indicators, in P(positive) units.")

    st.subheader("How the perturbation neighbourhood was built")
    a, b = st.columns([1, 1.1])
    with a:
        st.markdown(
            f"1. The patient's preprocessed values are discretised into **training-set quartile bins**.\n"
            f"2. LIME draws **{num_samples:,} synthetic neighbours** by sampling a bin for each feature from "
            f"the training distribution and a value inside that bin (seed {seed}).\n"
            f"3. Every neighbour is scored by the **same fitted pipeline** (imputer + XGBoost); column 1, "
            f"P(diabetes-positive), is the explained target.\n"
            f"4. Neighbours are weighted by an exponential kernel (width {res.kernel_width:.2f}) on their "
            f"distance from the patient in the binary bin space.\n"
            f"5. A weighted ridge model is fitted; its coefficients are the explanation and its weighted R² "
            f"is the fidelity score.")
        st.markdown(ui.note(
            "Low R² means the straight-line surrogate cannot reproduce the trees' behaviour around this "
            "patient, so the weights should be read with extra caution."), unsafe_allow_html=True)
    with b:
        show_fig(viz.plot_neighborhood, res.neighborhood_probs, res.model_prob)

    st.subheader("Seed stability check")
    if st.toggle("Re-run LIME with 5 different seeds", key="lime_stab_toggle"):
        stab = stability_for(prof.key, tuple(range(5)), num_samples)
        m = stab.metrics
        s = st.columns(4)
        s[0].metric("Top-3 Jaccard across seeds", f"{m['stability_jaccard_top3']:.2f}")
        s[1].metric("Rank ρ across seeds", f"{m['stability_spearman']:.2f}")
        s[2].metric("Features with stable sign", f"{100 * m['sign_consistency']:.0f}%")
        s[3].metric("R² mean ± sd", f"{m['r2_mean']:.2f} ± {m['r2_std']:.2f}")
        a, b = st.columns([1.25, 1])
        with a:
            show_fig(viz.plot_stability, stab.weights, f"{prof.label}: LIME weights")
        with b:
            st.markdown(ui.note(
                "Each dot is one seed's weight for a feature. Tight clusters on one side of zero mean the "
                "explanation does not depend on the random neighbourhood. A cluster straddling zero means the "
                "feature's direction is not reliable for this patient."), unsafe_allow_html=True)


def page_shap(R: Resources) -> None:
    st.header("SHAP explainability (TreeSHAP)")
    st.markdown(ui.note(
        "<b>Units.</b> TreeExplainer explains XGBoost's raw margin, so every SHAP value here is in "
        "<b>log-odds</b> of the diabetes-positive class: base value + Σ SHAP = margin, and "
        "P(positive) = sigmoid(margin). SHAP values cannot be added to a probability."), unsafe_allow_html=True)
    tab_g, tab_p = st.tabs(["Global (test set)", "Selected patient"])
    with tab_g:
        imp = sx.global_importance(R.shap_test)
        a, b = st.columns(2)
        with a:
            show_fig(viz.plot_shap_importance, imp)
        with b:
            show_fig(viz.plot_shap_beeswarm, R.shap_test)
        st.caption("Beeswarm: each dot is one held-out patient; colour is the feature's (imputed) value from "
                   "low (light) to high (dark); position is its SHAP contribution.")
        chk = load_json("outputs/metrics/shap_additivity.json")
        if chk:
            t = chk["test"]
            st.markdown(f"**Additivity verified on {t['n']} test patients:** max |base + Σφ - margin| = "
                        f"`{t['max_abs_error_logodds']:.2e}` log-odds; max |sigmoid(base + Σφ) - P| = "
                        f"`{t['max_abs_error_probability_via_sigmoid']:.2e}`. For contrast, wrongly adding "
                        f"log-odds SHAP values to the base probability is off by up to "
                        f"`{t['max_abs_error_naive_probability_sum']:.2f}` probability units.")
    with tab_p:
        prof = current_profile(R)
        sp = sx.explain_patient(R.tree, R.xgb, prof.x_raw)
        profile_banner(prof, sp.probability)
        k = st.columns(4)
        k[0].markdown(ui.kpi("Base value E[f(X)]", f"{sp.base_value:+.3f}",
                             f"log-odds ≈ P {sx.sigmoid(sp.base_value):.3f}"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Σ SHAP values", f"{sum(sp.values.values()):+.3f}", "log-odds"),
                      unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Model margin f(x)", f"{sp.margin:+.3f}",
                             f"additivity error {sp.additivity_error:.1e}"), unsafe_allow_html=True)
        k[3].markdown(ui.kpi("P(positive) = sigmoid(f(x))", f"{sp.probability:.3f}", "uncalibrated"),
                      unsafe_allow_html=True)
        exp_row = sx.explain(R.tree, R.xgb, prof.x_raw)[0]
        a, b = st.columns([1.1, 1])
        with a:
            show_fig(viz.plot_shap_waterfall, exp_row, f"{prof.label}: SHAP waterfall")
        with b:
            st.dataframe(sp.table(), hide_index=True, width="stretch",
                         column_config={"shap_logodds": st.column_config.NumberColumn("SHAP (log-odds)",
                                                                                      format="%+.4f"),
                                        "model_input_value": st.column_config.NumberColumn("model input",
                                                                                           format="%.3g")})
        st.markdown("**Force plot**")
        show_fig(viz.plot_shap_force, sp.base_value, exp_row.values, exp_row.display_data)


def page_comparison(R: Resources) -> None:
    st.header("Explanation comparison: LIME vs SHAP")
    prof = current_profile(R)
    res = lime_for(prof.key, config.LIME_PRIMARY_SEED, config.LIME_NUM_SAMPLES)
    sp = sx.explain_patient(R.tree, R.xgb, prof.x_raw)
    cmp = compare(res.weights, sp.values)
    profile_banner(prof, sp.probability)
    st.caption(f"LIME: seed {config.LIME_PRIMARY_SEED}, {config.LIME_NUM_SAMPLES:,} samples. Both methods explain "
               f"the diabetes-positive class at the same preprocessed input.")

    k = st.columns(5)
    k[0].metric("Top-3 overlap", f"{cmp['top3_overlap']:.2f}")
    k[1].metric("Top-5 overlap", f"{cmp['top5_overlap']:.2f}")
    k[2].metric("Top-3 Jaccard", f"{cmp['jaccard_top3']:.2f}")
    rho = cmp["rank_spearman"]
    k[3].metric("Rank Spearman ρ", "n/a" if np.isnan(rho) else f"{rho:.2f}")
    k[4].metric("Direction disagreements", f"{cmp['n_direction_disagreements']}")

    show_fig(viz.plot_side_by_side, res.weights, sp.values, f"{prof.label}: LIME vs SHAP",
             cmp["direction_disagreement_features"])
    a, b = st.columns([1, 1.2])
    with a:
        show_fig(viz.plot_rank_slope, res.weights, sp.values)
    with b:
        sr = {f: i + 1 for i, f in enumerate(rank_features(sp.values))}
        lr = {f: i + 1 for i, f in enumerate(rank_features(res.weights))}
        dis = set(cmp["direction_disagreement_features"])
        table = pd.DataFrame([{
            "feature": f, "SHAP rank": sr[f], "LIME rank": lr[f],
            "SHAP (log-odds)": sp.values[f], "LIME weight": res.weights[f],
            "same sign": "yes" if np.sign(sp.values[f]) == np.sign(res.weights[f]) else "no",
            "material disagreement": "yes" if f in dis else "",
        } for f in config.FEATURES]).sort_values("SHAP rank")
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"SHAP (log-odds)": st.column_config.NumberColumn(format="%+.3f"),
                                    "LIME weight": st.column_config.NumberColumn(format="%+.3f")})
        st.caption(f"A sign difference counts as material only if |LIME| ≥ {config.LIME_SIGN_TOL} and "
                   f"|SHAP| ≥ {config.SHAP_SIGN_TOL} log-odds.")

    st.subheader("Why the two methods can differ")
    st.markdown(
        "- **Different questions.** TreeSHAP distributes the margin exactly among features relative to the "
        "model's average output, following the trees' own paths. LIME fits a straight-line surrogate to model "
        "outputs on synthetic neighbours, relative to *other quartile bins*.\n"
        "- **Different units.** SHAP is in log-odds, LIME weights in probability units of a local surrogate. "
        "Only ranks, overlaps and signs are compared here - never raw magnitudes.\n"
        "- **Discretisation.** LIME only sees which quartile a value falls in. A glucose of 145 and one of 195 "
        "can share the same top bin, so LIME may under-weight a feature whose exact value matters to the trees.\n"
        "- **Sampling noise and fidelity.** LIME depends on its random neighbourhood and on how well a linear "
        "model fits it (R² shown on the LIME page). TreeSHAP is deterministic for a fixed model.\n"
        "- **Saturation.** For very confident predictions most perturbations stay on the same side of the "
        "threshold; probability changes become small, which can make LIME weights small and noisy.")
    if not np.isnan(rho):
        verdict = ("largely consistent" if cmp["top3_overlap"] >= 2 / 3 and rho >= 0.6 else
                   "partially consistent" if cmp["top3_overlap"] >= 1 / 3 else "inconsistent")
        st.markdown(ui.note(
            f"For this profile the explanations are <b>{verdict}</b>: {int(round(cmp['top3_overlap'] * 3))}/3 "
            f"shared top features (LIME: {', '.join(cmp['lime_top3'])}; SHAP: {', '.join(cmp['shap_top3'])}), "
            f"rank ρ = {rho:.2f}, LIME R² = {res.r2:.2f}."), unsafe_allow_html=True)


def page_evaluation(R: Resources) -> None:
    st.header("Experimental evaluation")
    mm = load_json("outputs/metrics/model_metrics.json")
    if mm is None:
        st.warning("No evaluation outputs found. Run `python evaluate.py`.")
        return
    tab_m, tab_s, tab_st = st.tabs(["Model performance", "LIME vs SHAP study", "Fidelity & stability"])
    with tab_m:
        x, b = mm["xgboost"], mm["logistic_regression"]
        keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision",
                "brier_score"]
        st.dataframe(pd.DataFrame({"metric": keys, "XGBoost": [x[k] for k in keys],
                                   "Logistic regression (baseline)": [b[k] for k in keys]}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("XGBoost", "Logistic regression (baseline)")})
        d = mm["auc_difference_xgb_minus_lr"]
        cv_x, cv_b = R.meta["cv_xgb_train_split"]["roc_auc"], R.meta["cv_baseline_train_split"]["roc_auc"]
        st.markdown(
            f"Held-out test set, n = {x['n']}, threshold {x['threshold']}. XGBoost ROC-AUC 95% bootstrap CI "
            f"{x['roc_auc_ci95'][0]:.3f}-{x['roc_auc_ci95'][1]:.3f}. Paired bootstrap AUC difference "
            f"(XGB - LR) = {d['observed_diff']:+.3f}, 95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}. "
            f"Training-split 5-fold CV AUC: XGBoost {cv_x['mean']:.3f} ± {cv_x['std']:.3f}, "
            f"logistic regression {cv_b['mean']:.3f} ± {cv_b['std']:.3f}.")
        st.markdown(ui.note(
            "<b>Reading the comparison.</b> When the confidence interval of the AUC difference includes zero, "
            "the data do not show XGBoost to be better than the linear baseline. XGBoost is used here because "
            "it captures non-linear thresholds and interactions natively, handles unscaled tabular features, "
            "and admits exact, fast TreeSHAP explanations - not because it wins on this small dataset."),
            unsafe_allow_html=True)
        best = R.meta["best_params"]
        st.caption("Selected XGBoost hyper-parameters (5-fold CV on training split only): " +
                   ", ".join(f"{k}={v}" for k, v in best.items()))
        c = st.columns(3)
        with c[0]:
            show_image("outputs/figures/confusion_matrix.png")
        with c[1]:
            show_image("outputs/figures/roc_curve.png")
        with c[2]:
            show_image("outputs/figures/pr_curve.png")
        show_image("outputs/figures/model_comparison.png")
    with tab_s:
        s = load_json("outputs/metrics/explanation_study_summary.json")
        rel = load_csv("outputs/experiments/relationships.csv")
        if s is None or rel is None:
            st.warning("Study outputs missing. Run `python evaluate.py`.")
            return
        st.markdown(f"**Question:** how consistently do LIME and SHAP explain XGBoost predictions across "
                    f"patient profiles and confidence levels? All {s['n_patients']} held-out patients; LIME "
                    f"{s['lime_num_samples']:,} samples; stability over {len(s['lime_seeds'])} seeds.")
        k = st.columns(5)
        k[0].metric("Same top feature", f"{100 * s['top1_match_rate']:.0f}%")
        k[1].metric("Mean top-3 overlap", f"{s['mean_top3_overlap']:.2f}")
        k[2].metric("Median rank ρ", f"{s['median_rank_spearman']:.2f}")
        k[3].metric("Patients with a sign conflict", f"{100 * s['share_with_direction_disagreement']:.0f}%")
        k[4].metric("Mean LIME R²", f"{s['mean_lime_r2']:.2f}")
        show_image("outputs/figures/agreement_distribution.png")
        show_image("outputs/figures/confidence_relationships.png")
        st.dataframe(rel[["relationship", "n", "spearman_rho", "p_value", "strength", "significant_alpha_0.05",
                          "significant_bonferroni"]], hide_index=True, width="stretch",
                     column_config={"spearman_rho": st.column_config.NumberColumn(format="%+.3f"),
                                    "p_value": st.column_config.NumberColumn(format="%.2g")})
        st.caption(f"Eight correlations are tested, so a Bonferroni threshold of "
                   f"{rel['bonferroni_threshold'].iloc[0]:.4f} is shown alongside the nominal 0.05. Correlation "
                   f"does not imply causation, and n = {s['n_patients']} limits precision.")
        show_image("outputs/figures/band_summary.png")
        bands = load_csv("outputs/experiments/band_summary.csv")
        if bands is not None:
            st.dataframe(bands, hide_index=True, width="stretch")
        findings = config.OUTPUTS_DIR / "findings.md"
        if findings.exists():
            with st.expander("Full generated findings (outputs/findings.md)"):
                st.markdown(findings.read_text("utf-8"))
    with tab_st:
        s = load_json("outputs/metrics/explanation_study_summary.json") or {}
        if s:
            k = st.columns(4)
            k[0].metric("Mean seed top-3 Jaccard", f"{s['mean_stability_jaccard_top3']:.2f}")
            k[1].metric("Mean seed rank ρ", f"{s['mean_stability_spearman']:.2f}")
            k[2].metric("Patients with a material sign flip",
                        f"{100 * s['share_material_sign_flip_across_seeds']:.0f}%",
                        help=f"Some seed gives a weight >= +{config.LIME_SIGN_TOL} and another <= "
                             f"-{config.LIME_SIGN_TOL} for the same feature. Counting any sign change, "
                             f"including near-zero weights: {100 * s['share_any_sign_flip_across_seeds']:.0f}%.")
            k[3].metric("Median LIME R²", f"{s['median_lime_r2']:.2f}")
        show_image("outputs/figures/stability_distribution.png")
        study = load_csv("outputs/experiments/study.csv")
        if study is not None:
            st.markdown("**Per-patient study table** (sortable)")
            cols = ["patient", "y_true", "prob_positive", "confidence_band", "lime_r2", "top3_overlap",
                    "rank_spearman", "n_direction_disagreements", "stability_jaccard_top3",
                    "material_sign_flip_features"]
            st.dataframe(study[cols], hide_index=True, width="stretch", height=360,
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("prob_positive", "lime_r2", "top3_overlap", "rank_spearman",
                                                  "stability_jaccard_top3")})


def page_cases(R: Resources) -> None:
    st.header("Case study explorer")
    cases = load_json("outputs/case_studies/case_studies.json")
    if not cases:
        st.warning("No case studies found. Run `python evaluate.py`.")
        return
    st.caption("Five genuine held-out patients selected by fixed rules (see `src/case_studies.py`); "
               "a patient is never reused across cases.")
    labels = {c["case_id"]: f"Case {c['case_id']}" for c in cases}
    cid = st.segmented_control("Case", list(labels), format_func=lambda i: labels[i], default=1,
                               key="case_choice", required=True)
    c = next(x for x in cases if x["case_id"] == cid)
    st.subheader(f"Case {c['case_id']}: {c['title']}")
    st.markdown(f"*Selection rule:* {c['rule']}")
    if not c["available"]:
        st.warning(c["selection_note"])
        return
    m = c["metrics"]
    k = st.columns(5)
    k[0].markdown(ui.kpi("Patient", c["patient"], m["confidence_band"]), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("P(positive)", f"{c['prob_positive']:.3f}", "uncalibrated"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Model class", "positive" if c["y_pred"] else "negative",
                         "correct" if c["y_pred"] == c["y_true"] else "misclassified"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Recorded outcome", "positive" if c["y_true"] else "negative", "ground truth"),
                  unsafe_allow_html=True)
    k[4].markdown(ui.kpi("SHAP margin", f"{c['shap_margin_logodds']:+.2f}",
                         f"base {c['shap_base_value_logodds']:+.2f} log-odds"), unsafe_allow_html=True)
    st.markdown("**Evidence-based interpretation**")
    st.markdown("\n".join(f"- {line}" for line in c["interpretation"]))
    st.button(f"Open {c['patient']} in the live explanation pages", on_click=open_patient,
              args=(c["patient_id"],))
    f = c["figures"]
    show_image(f["side_by_side"])
    a, b = st.columns(2)
    with a:
        show_image(f["shap_waterfall"])
        show_image(f["stability"])
    with b:
        show_image(f["lime"])
        show_image(f["rank_slope"])
    show_image(f["shap_force"])
    with st.expander("Patient values"):
        st.dataframe(pd.DataFrame({"feature": config.FEATURES,
                                   "recorded": [c["raw_values"][x] for x in config.FEATURES],
                                   "model input": [c["model_input_values"][x] for x in config.FEATURES],
                                   "imputed": ["yes" if c["imputed"][x] else "" for x in config.FEATURES]}),
                     hide_index=True, width="stretch")
    with st.expander("All five cases at a glance"):
        summary = load_csv("outputs/case_studies/case_studies_summary.csv")
        if summary is not None:
            st.dataframe(summary, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        R = load_resources()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Model artefacts are not available: {exc}")
        st.code("python train_model.py\npython evaluate.py\nstreamlit run app.py")
        st.stop()
    page = sidebar(R)
    {
        PAGES[0]: page_home, PAGES[1]: page_prediction, PAGES[2]: page_lime, PAGES[3]: page_shap,
        PAGES[4]: page_comparison, PAGES[5]: page_evaluation, PAGES[6]: page_cases,
    }[page](R)


main()
