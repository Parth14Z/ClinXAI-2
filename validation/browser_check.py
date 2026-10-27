"""Real-browser click-through of the ClinXAI dashboard with Playwright.

Usage (Playwright is a validation-only dependency, not part of requirements.txt):
    pip install playwright
    streamlit run app.py --server.port 8501 --server.headless true      # in another terminal
    python validation/browser_check.py [--url http://localhost:8501] [--channel msedge]

Drives an installed Microsoft Edge (or Chromium) through all seven pages, the patient selector,
the what-if editor, the LIME stability toggle, every SHAP plot and all five case studies.
Writes screenshots to outputs/screenshots/ and results to outputs/validation/browser_check.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "outputs" / "screenshots"
OUT = ROOT / "outputs" / "validation"
results: list[dict] = []
console_errors: list[str] = []
page_errors: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)


def wait_idle(page: Page, timeout: float = 180) -> None:
    """Wait until Streamlit has finished rerunning and every image has loaded."""
    time.sleep(0.6)
    t0 = time.time()
    while time.time() - t0 < timeout:
        busy = page.evaluate("""() => {
            const status = document.querySelector('[data-testid="stStatusWidget"]');
            const running = status && /running/i.test(status.innerText);
            const spinner = document.querySelector('[data-testid="stSpinner"]');
            const pending = [...document.images].some(i => !i.complete);
            return Boolean(running || spinner || pending);
        }""")
        if not busy:
            time.sleep(0.5)
            if not page.evaluate("""() => Boolean(document.querySelector('[data-testid="stSpinner"]'))"""):
                return
        time.sleep(0.3)
    raise TimeoutError("Streamlit did not become idle")


def main_text(page: Page) -> str:
    return page.locator('[data-testid="stMain"]').inner_text()


def image_stats(page: Page) -> dict:
    return page.evaluate("""() => {
        const imgs = [...document.querySelectorAll('[data-testid="stMain"] [data-testid="stImage"] img, '
                                                  + '[data-testid="stMain"] [data-testid="stImageContainer"] img')];
        const uniq = [...new Set(imgs)].filter(i => i.offsetParent !== null);
        return {n: uniq.length, broken: uniq.filter(i => !(i.complete && i.naturalWidth > 50)).length};
    }""")


def no_exceptions(page: Page) -> int:
    return page.locator('[data-testid="stException"]').count()


def shot(page: Page, name: str, full: bool = True) -> str:
    """Screenshot; for full pages, grow the viewport to the main container's scroll height."""
    path = SHOTS / f"{name}.png"
    if full:
        h, prev = 0, -1
        while h != prev:  # wait until the page height has settled (late images / reruns)
            prev = h
            time.sleep(0.8)
            h = page.evaluate("""() => {
            const b = document.querySelector('[data-testid="stMainBlockContainer"]');
            const m = document.querySelector('[data-testid="stMain"]');
            return Math.max(b ? b.scrollHeight + 120 : 0, m ? m.scrollHeight : 0);
        }""")
        page.set_viewport_size({"width": 1440, "height": min(int(h) + 40, 7000)})
        time.sleep(0.8)
        page.screenshot(path=str(path))
        page.set_viewport_size({"width": 1440, "height": 900})
        time.sleep(0.3)
    else:
        page.screenshot(path=str(path))
    return str(path.relative_to(ROOT)).replace("\\", "/")


def nav(page: Page, label: str) -> None:
    page.locator('[data-testid="stSidebar"] label').filter(has_text=label).first.click()
    wait_idle(page)


def banner_prob(page: Page) -> float:
    m = re.search(r"P\(positive\) = ([0-9.]+)", main_text(page))
    return float(m.group(1)) if m else float("nan")


def select_patient(page: Page, pid: str) -> None:
    box = page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]')
    box.click()
    page.keyboard.type(pid)
    page.get_by_role("option", name=re.compile(pid)).first.click()
    wait_idle(page)


def selected_patient(page: Page) -> str:
    return page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"] input').input_value()


def run(url: str, channel: str | None) -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        check("browser launched", True, f"{browser.browser_type.name} {browser.version} (channel={channel})")

        page.goto(url, wait_until="networkidle")
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=120_000)
        wait_idle(page)
        evidence: dict[str, str] = {}

        # 1 Home ---------------------------------------------------------------
        t = main_text(page)
        check("page 1 Home renders with hero and disclaimer",
              "ClinXAI" in t and "Educational prototype only" in t and "not a medical device" in t)
        check("Home: KPIs show test ROC-AUC 0.822 and top-3 overlap 0.77", "0.822" in t and "0.77" in t)
        check("Home: provenance MD5 shown", "3cbaa3e54586aa88cf6aacb4033e4470" in t)
        page.locator('[data-testid="stExpander"] summary').filter(has_text="System architecture").click()
        wait_idle(page)
        im = image_stats(page)
        check("Home: dataset chart + architecture diagram load", im["n"] >= 2 and im["broken"] == 0, str(im))
        check("Home: no Python exception", no_exceptions(page) == 0)
        evidence["home"] = shot(page, "01_home")
        evidence["home_top"] = shot(page, "01_home_viewport", full=False)

        # 2 Prediction ---------------------------------------------------------
        nav(page, "Diabetes prediction")
        p245 = banner_prob(page)
        t = main_text(page)
        check("page 2 Prediction renders for default patient PIMA-0245", "PIMA-0245" in t and abs(p245 - 0.934) < 5e-4,
              f"P = {p245}")
        check("Prediction: uncalibrated-probability note shown", "Uncalibrated probability" in t)
        evidence["prediction"] = shot(page, "02_prediction_PIMA-0245")
        select_patient(page, "PIMA-0032")
        p32 = banner_prob(page)
        check("patient selection updates probability (PIMA-0245 -> PIMA-0032)",
              "PIMA-0032" in main_text(page) and abs(p32 - 0.011) < 5e-4, f"{p245} -> {p32}")
        select_patient(page, "PIMA-0078")
        p78 = banner_prob(page)
        check("patient selection updates probability (-> PIMA-0078)", abs(p78 - 0.508) < 5e-4, f"P = {p78}")
        select_patient(page, "PIMA-0245")
        gl = page.locator('[data-testid="stNumberInput"]').filter(has_text="Glucose").locator("input")
        before = gl.input_value()
        gl.fill("120")
        page.get_by_role("button", name="Apply what-if profile").click()
        wait_idle(page)
        pw = banner_prob(page)
        t = main_text(page)
        check("what-if: Glucose 184 -> 120 for PIMA-0245 lowers P 0.934 -> 0.735",
              before.startswith("184") and "What-if profile" in t and abs(pw - 0.735) < 5e-4,
              f"input was {before}; P = {pw}")
        evidence["whatif"] = shot(page, "03_prediction_whatif_glucose120")
        nav(page, "Explanation comparison")
        check("what-if profile propagates to the comparison page", "What-if profile" in main_text(page))
        page.get_by_role("button", name="Reset to recorded values").click()
        wait_idle(page)
        check("reset restores the recorded patient", abs(banner_prob(page) - 0.934) < 5e-4
              and "What-if profile" not in main_text(page), f"P = {banner_prob(page)}")

        # 3 LIME ---------------------------------------------------------------
        nav(page, "LIME patient explanation")
        t = main_text(page)
        im = image_stats(page)
        check("page 3 LIME renders explanation chart + neighbourhood histogram",
              "surrogate fidelity r²" in t.lower() and im["n"] >= 2 and im["broken"] == 0 and no_exceptions(page) == 0, str(im))
        m = re.search(r"surrogate fidelity r²\s*([0-9.]+)", t, flags=re.I)
        live_r2 = float(m.group(1)) if m else float("nan")
        evidence["lime"] = shot(page, "04_lime_PIMA-0245")
        page.locator('[data-testid="stSidebar"], [data-testid="stMain"]').locator("label").filter(
            has_text="Re-run LIME with 5 different seeds").click()
        wait_idle(page)
        page.wait_for_function("""n => [...document.querySelectorAll('[data-testid="stMain"] [data-testid="stImage"] img')]
                                 .filter(i => i.complete && i.naturalWidth > 50).length > n""",
                               arg=im["n"], timeout=180_000)
        wait_idle(page)
        t = main_text(page)
        im2 = image_stats(page)
        check("LIME stability toggle runs 5 seeds and shows metrics + chart",
              "Top-3 Jaccard across seeds" in t and im2["n"] >= im["n"] + 1 and no_exceptions(page) == 0,
              f"images {im['n']} -> {im2['n']}; " + " ".join(re.findall(r"(?:across seeds|stable sign|mean ± sd)\s+([0-9.% ±]+)", t)))
        evidence["lime_stability"] = shot(page, "05_lime_stability_PIMA-0245")

        # 4 SHAP ---------------------------------------------------------------
        nav(page, "SHAP explainability")
        t = main_text(page)
        im = image_stats(page)
        check("page 4 SHAP global tab: importance + beeswarm render, additivity text shown",
              im["n"] >= 2 and im["broken"] == 0 and "Additivity verified on 154 test patients" in t, str(im))
        evidence["shap_global"] = shot(page, "06_shap_global_beeswarm")
        page.get_by_role("tab", name="Selected patient").click()
        wait_idle(page)
        t = main_text(page)
        im = image_stats(page)
        check("SHAP patient tab: waterfall + force plot render; margin +2.645, base -0.622",
              im["n"] >= 2 and im["broken"] == 0 and "+2.645" in t and "-0.622" in t and "Force plot" in t, str(im))
        evidence["shap_patient"] = shot(page, "07_shap_waterfall_force_PIMA-0245")

        # 5 Comparison ---------------------------------------------------------
        nav(page, "Explanation comparison")
        t = main_text(page)
        im = image_stats(page)
        check("page 5 Comparison: metrics, side-by-side and rank-slope render",
              "Top-3 overlap" in t and "Why the two methods can differ" in t and im["n"] >= 2 and im["broken"] == 0
              and no_exceptions(page) == 0, str(im))
        evidence["comparison"] = shot(page, "08_comparison_PIMA-0245")

        # 6 Evaluation ---------------------------------------------------------
        nav(page, "Experimental evaluation")
        t = main_text(page)
        im = image_stats(page)
        nd = page.locator('[data-testid="stDataFrame"]').count()
        check("page 6 Evaluation / Model performance: table + confusion, ROC, PR, comparison charts",
              "+0.009" in t and "0.746-0.883" in t and im["n"] >= 4 and im["broken"] == 0 and nd >= 1,
              f"{im}, dataframes {nd}")
        evidence["eval_models"] = shot(page, "09_evaluation_model_performance")
        page.get_by_role("tab", name="LIME vs SHAP study").click()
        wait_idle(page)
        t = main_text(page)
        im = image_stats(page)
        check("Evaluation / study tab: KPIs, 3 figures, relationship + band tables",
              "73%" in t and "0.82" in t and "26%" in t and im["n"] >= 3 and im["broken"] == 0, str(im))
        evidence["eval_study"] = shot(page, "10_evaluation_lime_shap_study")
        page.get_by_role("tab", name="Fidelity & stability").click()
        wait_idle(page)
        t = main_text(page)
        im = image_stats(page)
        check("Evaluation / fidelity & stability tab: KPIs, distribution figure, per-patient table",
              "0.88" in t and "11%" in t and im["n"] >= 1 and im["broken"] == 0, str(im))
        evidence["eval_stability"] = shot(page, "11_evaluation_fidelity_stability")

        # 7 Cases --------------------------------------------------------------
        nav(page, "Case study explorer")
        expected = {1: ("PIMA-0245", "0.934"), 2: ("PIMA-0032", "0.011"), 3: ("PIMA-0078", "0.508"),
                    4: ("PIMA-0351", "0.455"), 5: ("PIMA-0486", "0.459")}
        for cid, (pid, prob) in expected.items():
            page.locator('[data-testid="stButtonGroup"] button').filter(has_text=f"Case {cid}").click()
            wait_idle(page)
            t = main_text(page)
            im = image_stats(page)
            check(f"page 7 Case {cid} loads: {pid}, P {prob}, 6 figures",
                  f"Case {cid}:" in t and pid in t and prob in t and im["n"] >= 6 and im["broken"] == 0
                  and no_exceptions(page) == 0, str(im))
            evidence[f"case{cid}"] = shot(page, f"{12 + cid - 1:02d}_case{cid}_{pid}")
        page.get_by_role("button", name=re.compile("Open PIMA-0486")).click()
        wait_idle(page)
        nav(page, "Explanation comparison")
        t = main_text(page)
        check("'Open patient' button carries Case 5 patient into the live pages",
              "PIMA-0486" in selected_patient(page) and "PIMA-0486" in t, selected_patient(page))
        evidence["case5_live_comparison"] = shot(page, "17_case5_live_comparison_PIMA-0486")

        # Console ---------------------------------------------------------------
        check("no uncaught JavaScript page errors", not page_errors, "; ".join(page_errors[:5]))
        check("no browser console errors", not console_errors, "; ".join(console_errors[:5]))
        browser.close()

    n_fail = sum(not r["passed"] for r in results)
    (OUT / "browser_check.json").write_text(json.dumps({
        "url": url, "channel": channel, "n_checks": len(results), "n_passed": len(results) - n_fail,
        "n_failed": n_fail, "checks": results, "console_errors": console_errors, "page_errors": page_errors,
        "live_lime_r2_PIMA-0245_seed42": live_r2, "screenshots": evidence}, indent=2), "utf-8")
    print(f"\n{len(results) - n_fail}/{len(results)} browser checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge", help="msedge, chrome, or '' for bundled Chromium")
    a = ap.parse_args()
    raise SystemExit(run(a.url, a.channel or None))
