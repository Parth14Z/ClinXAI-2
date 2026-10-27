"""Chart generation and Streamlit rendering."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import config
from src import lime_explainer as lx
from src import plotting as viz
from src import shap_explainer as sx
from src.model import transform_features

W = dict(zip(config.FEATURES, [0.3, -0.2, 0.1, 0.05, -0.04, 0.02, 0.01, -0.005]))


@pytest.mark.parametrize("make", [
    lambda: viz.plot_confusion_matrix(np.array([[80, 20], [21, 33]])),
    lambda: viz.plot_roc_curves({"A": (np.array([0, .2, 1]), np.array([0, .7, 1]), .8)}),
    lambda: viz.plot_pr_curves({"A": (np.array([0, .5, 1]), np.array([1, .7, .35]), .7)}, 0.35),
    lambda: viz.plot_lime_explanation(W, {f: f"{f} <= 1" for f in config.FEATURES}),
    lambda: viz.plot_shap_contributions(W),
    lambda: viz.plot_side_by_side(W, W, disagree=["Glucose"]),
    lambda: viz.plot_rank_slope(W, dict(reversed(list(W.items())))),
    lambda: viz.plot_stability(pd.DataFrame([W, W], columns=config.FEATURES)),
    lambda: viz.plot_neighborhood(np.random.default_rng(0).uniform(size=200), 0.6),
    lambda: viz.plot_shap_importance(pd.Series(np.abs(list(W.values())), index=config.FEATURES)),
    lambda: viz.plot_model_comparison(pd.DataFrame({"XGBoost": [.8, .7], "LR": [.78, .6]},
                                                   index=["roc_auc", "f1"])),
    lambda: viz.plot_architecture(),
])
def test_chart_functions_return_figures_and_save(make, tmp_path):
    fig = make()
    assert isinstance(fig, Figure)
    out = viz.save(fig, tmp_path / "chart.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_shap_native_plots_render(fitted, split, tmp_path):
    tree = sx.build_tree_explainer(fitted)
    exp = sx.explain(tree, fitted, split.X_test.iloc[:40])
    for name, fig in (("bee", viz.plot_shap_beeswarm(exp)),
                      ("wf", viz.plot_shap_waterfall(exp[0], "t")),
                      ("force", viz.plot_shap_force(float(exp.base_values[0]), exp.values[0],
                                                    exp.display_data[0]))):
        assert viz.save(fig, tmp_path / f"{name}.png").stat().st_size > 5_000


def test_live_lime_chart_from_real_explanation(fitted, split, lime_expl, tmp_path):
    x_t = transform_features(fitted, split.X_test.iloc[[0]])
    res = lx.explain_patient(fitted, lime_expl, x_t, 600, keep_neighborhood=True)
    assert viz.save(viz.plot_lime_explanation(res.weights, res.rules), tmp_path / "l.png").exists()


# ------------------------------ Streamlit ---------------------------------- #
ARTEFACTS = [config.XGB_MODEL_PATH, config.TRAIN_CSV, config.TEST_CSV, config.MODEL_METADATA_PATH,
             config.CASE_STUDIES_DIR / "case_studies.json"]


@pytest.fixture(scope="module")
def app():
    missing = [str(p) for p in ARTEFACTS if not p.exists()]
    if missing:
        pytest.fail(f"Run train_model.py and evaluate.py first; missing {missing}")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=180)
    at.run()
    return at


PAGES = ["Home & clinical overview", "Diabetes prediction", "LIME patient explanation",
         "SHAP explainability", "Explanation comparison", "Experimental evaluation", "Case study explorer"]


@pytest.mark.parametrize("page", PAGES)
def test_every_dashboard_page_renders(app, page):
    app.sidebar.radio(key="nav").set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(page.split(" ")[0].lower() in h.value.lower() for h in app.header) or page.startswith("Home")


def test_home_contains_disclaimer(app):
    app.sidebar.radio(key="nav").set_value(PAGES[0]).run()
    text = " ".join(m.value for m in app.markdown)
    assert "not a medical device" in text and "Multi-Modal" in text


def test_prediction_page_shows_uncalibrated_warning(app):
    app.sidebar.radio(key="nav").set_value(PAGES[1]).run()
    assert "Uncalibrated probability" in " ".join(m.value for m in app.markdown)


def test_patient_switch_and_case_navigation(app):
    app.sidebar.radio(key="nav").set_value(PAGES[1]).run()
    options = app.sidebar.selectbox(key="patient_id").options
    assert len(options) == 154
    app.sidebar.selectbox(key="patient_id").select_index(5).run()
    assert not app.exception
    app.sidebar.radio(key="nav").set_value(PAGES[6]).run()
    app.button[0].click().run()
    assert not app.exception


def _render_waterfalls(exp, patients, rounds, use_lock=True):
    """Render SHAP waterfalls for several patients from concurrent threads; return PNG bytes and errors."""
    import io
    import threading

    import matplotlib.pyplot as plt

    def build(i):
        return viz.plot_shap_waterfall(exp[i], f"patient {i}")

    out, errors = {i: [] for i in patients}, []

    def worker(i):
        for _ in range(rounds):
            try:
                buf = io.BytesIO()
                if use_lock:
                    with viz.exclusive_figure(build, i) as fig:
                        fig.savefig(buf, format="png")
                else:
                    fig = build(i)
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                out[i].append(buf.getvalue())
            except Exception as exc:  # noqa: BLE001 - any failure is a thread-safety failure here
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in patients]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, errors


def test_concurrent_sessions_get_their_own_shap_figures(fitted, split):
    """Dashboard sessions run in parallel threads; SHAP draws on pyplot's global current figure."""
    exp = sx.explain(sx.build_tree_explainer(fitted), fitted, split.X_test.iloc[:4])
    patients = [0, 1, 2, 3]
    baseline, _ = _render_waterfalls(exp, patients, rounds=1)
    out, errors = _render_waterfalls(exp, patients, rounds=4)
    assert not errors
    assert all(png == baseline[i][0] for i in patients for png in out[i])
