"""Concurrent-user smoke test for the ClinXAI dashboard (validation only; needs Playwright).

Usage:  python validation/concurrency_check.py [--url http://localhost:8501] [--sessions 4] [--rounds 3]

Several browser sessions open the SHAP "Selected patient" view for different patients at the same
time; every session must finish with its charts loaded and no Python exception on the page.
This is a smoke test. Pixel-exact isolation of concurrent figures is tested deterministically by
tests/test_plotting_and_app.py::test_concurrent_sessions_get_their_own_shap_figures. (Comparing
chart bytes through the browser is unreliable: Streamlit's media URLs expire on rerun.)
Writes outputs/validation/concurrency_check.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parents[1]
PATIENTS = ["PIMA-0245", "PIMA-0032", "PIMA-0351", "PIMA-0486"]


async def wait_idle(page: Page, timeout: float = 180) -> None:
    await page.wait_for_timeout(600)
    await page.wait_for_function("""() => {
        const s = document.querySelector('[data-testid="stStatusWidget"]');
        const running = s && /running/i.test(s.innerText);
        return !running && !document.querySelector('[data-testid="stSpinner"]')
               && [...document.images].every(i => i.complete);
    }""", timeout=timeout * 1000)
    await page.wait_for_timeout(500)


async def shap_charts(browser, url: str, pid: str) -> dict:
    ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
    page = await ctx.new_page()
    await page.goto(url)
    await page.wait_for_selector('[data-testid="stSidebar"] [data-testid="stSelectbox"]', timeout=120_000)
    await wait_idle(page)
    box = page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]')
    for _ in range(5):  # under load a keystroke can land while the widget re-renders; retry
        await box.click()
        await page.keyboard.type(pid)
        await page.get_by_role("option", name=re.compile(pid)).first.click()
        await wait_idle(page)
        if pid in await box.locator("input").input_value():
            break
    await page.locator('[data-testid="stSidebar"] label').filter(has_text="SHAP explainability").first.click()
    await wait_idle(page)
    tab = page.get_by_role("tab", name="Selected patient")
    main = page.locator('[data-testid="stMain"]')
    for _ in range(20):  # a rerun finishing after the click re-mounts the tabs on the first one; retry
        await tab.click()
        await wait_idle(page)
        if await tab.get_attribute("aria-selected") == "true" and "Force plot" in await main.inner_text():
            break
    n_charts = await page.evaluate("""() => [...document.querySelectorAll('[data-testid="stMain"] [data-testid="stImage"] img')]
                                          .filter(i => i.offsetParent !== null && i.complete && i.naturalWidth > 50).length""")
    text = await page.locator('[data-testid="stMain"]').inner_text()
    errors = await page.locator('[data-testid="stException"]').count()
    selected = await box.locator("input").input_value()
    shown = sorted(set(re.findall(r"PIMA-\d{4}", text)))
    await ctx.close()
    return {"patient": pid, "selected": selected, "shown": shown, "banner_ok": shown == [pid],
            "n_charts": n_charts, "exceptions": errors}


async def main(url: str, sessions: int, rounds: int, channel: str | None) -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel=channel, headless=True)
        concurrent = []
        for r in range(rounds):
            batch = [PATIENTS[(r + i) % len(PATIENTS)] for i in range(sessions)]
            concurrent += await asyncio.gather(*(shap_charts(browser, url, pid) for pid in batch))
        await browser.close()

    bad = [c for c in concurrent if c["exceptions"] or c["n_charts"] < 2 or not c["banner_ok"]]
    ok = not bad
    report = {"url": url, "sessions_per_round": sessions, "rounds": rounds,
              "concurrent_sessions": len(concurrent), "failed_sessions": bad, "passed": ok,
              "sessions": concurrent}
    out = ROOT / "outputs" / "validation" / "concurrency_check.json"
    out.write_text(json.dumps(report, indent=2), "utf-8")
    print(f"{len(concurrent)} concurrent sessions ({rounds} rounds x {sessions}); {len(bad)} failed "
          f"(exception, missing charts or wrong patient) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--channel", default="msedge")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main(a.url, a.sessions, a.rounds, a.channel or None)))
