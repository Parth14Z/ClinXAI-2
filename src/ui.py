"""Styling and small presentational helpers for the Streamlit dashboard."""
from __future__ import annotations

import html

from . import config

P = config.PALETTE

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {P['slate']}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {P['teal_pale']} 0%, {P['purple_pale']} 100%);
  border-right: 1px solid #DCE6E5;
}}
.cx-hero {{
  border-radius: 18px; padding: 26px 30px; margin-bottom: 18px; color: white;
  background: linear-gradient(120deg, {P['teal_dark']} 0%, {P['teal']} 55%, {P['purple_dark']} 120%);
}}
.cx-hero h1 {{ color: white; margin: 0 0 6px 0; font-size: 2.0rem; }}
.cx-hero p {{ margin: 0; opacity: 0.93; font-size: 1.02rem; }}
.cx-eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; opacity: .85;
  margin-bottom: 6px; }}
.cx-card {{
  background: white; border: 1px solid #E3E8EE; border-radius: 14px; padding: 16px 18px;
  height: 100%;
}}
.cx-card h4 {{ margin: 0 0 6px 0; color: {P['teal_dark']}; font-size: 1.0rem; }}
.cx-card p, .cx-card li {{ color: #475467; font-size: .92rem; line-height: 1.5; }}
.cx-kpi {{ background: white; border: 1px solid #E3E8EE; border-radius: 14px; padding: 14px 16px; }}
.cx-kpi .label {{ color: #5B6675; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
.cx-kpi .value {{ color: {P['slate']}; font-size: 1.65rem; font-weight: 700; line-height: 1.2; }}
.cx-kpi .sub {{ color: #5B6675; font-size: .8rem; }}
.cx-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
  font-weight: 600; margin-right: 6px; }}
.cx-pill.pos {{ background: {P['purple_pale']}; color: {P['purple_dark']}; border: 1px solid {P['purple_light']}; }}
.cx-pill.neg {{ background: {P['teal_pale']}; color: {P['teal_dark']}; border: 1px solid {P['teal_light']}; }}
.cx-pill.neutral {{ background: #F2F4F7; color: #475467; border: 1px solid #E3E8EE; }}
.cx-note {{ border-left: 4px solid {P['purple']}; background: {P['purple_pale']}; padding: 10px 14px;
  border-radius: 6px; color: #3F3A55; font-size: .9rem; margin: 8px 0 14px 0; }}
.cx-disclaimer {{ border-left: 4px solid {P['teal']}; background: {P['teal_pale']}; padding: 10px 14px;
  border-radius: 6px; color: #234; font-size: .86rem; margin: 6px 0 12px 0; }}
.cx-meter {{ height: 14px; border-radius: 999px; background: #EEF1F4; position: relative; margin: 8px 0 4px 0; }}
.cx-meter .fill {{ height: 100%; border-radius: 999px; }}
.cx-meter .tick {{ position: absolute; top: -4px; width: 2px; height: 22px; background: {P['slate']}; }}
.cx-meter-scale {{ display: flex; justify-content: space-between; color: #5B6675; font-size: .75rem; }}
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def hero(eyebrow: str, title: str, subtitle: str) -> str:
    return (f'<div class="cx-hero"><div class="cx-eyebrow">{esc(eyebrow)}</div>'
            f'<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="cx-kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')


def card(title: str, body_html: str) -> str:
    return f'<div class="cx-card"><h4>{esc(title)}</h4>{body_html}</div>'


def pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="cx-pill {kind}">{esc(text)}</span>'


def note(text: str, kind: str = "note") -> str:
    cls = "cx-disclaimer" if kind == "disclaimer" else "cx-note"
    return f'<div class="{cls}">{text}</div>'


def probability_meter(p: float, threshold: float = config.DECISION_THRESHOLD) -> str:
    color = P["purple"] if p >= threshold else P["teal"]
    return (f'<div class="cx-meter"><div class="fill" style="width:{100 * p:.1f}%;background:{color}"></div>'
            f'<div class="tick" style="left:{100 * threshold:.1f}%"></div></div>'
            f'<div class="cx-meter-scale"><span>0.0</span><span>decision threshold {threshold:.1f}</span>'
            f'<span>1.0</span></div>')


DISCLAIMER = ("<b>Educational prototype only.</b> ClinXAI is a coursework demonstration of explainable "
              "machine learning on a historical research dataset. It is not a medical device, has not been "
              "clinically validated, does not diagnose diabetes, does not predict future disease onset, and "
              "must not inform any decision about a real person's care.")
