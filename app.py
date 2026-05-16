#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
課程精準行銷與會員推薦儀表板

安裝：
pip install -r requirements.txt

啟動：
streamlit run app.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="課程精準行銷與會員推薦儀表板",
    page_icon="◆",
    layout="wide",
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#1B2E4B", "#E05643", "#E57362", "#6B7A90", "#F2B8A8", "#2F4858"]

APP_DIR = Path(__file__).resolve().parent
outputs_path = APP_DIR / "outputs"
if not outputs_path.exists():
    outputs_path = Path.cwd() / "outputs"


st.markdown(
    """
    <style>
    :root {
        --bg: #F4F6F9;
        --card: #FFFFFF;
        --navy: #1B2E4B;
        --sidebar: #212B36;
        --accent: #E05643;
        --accent-soft: #E57362;
        --accent-faint: #FFF3F0;
        --ink: #1E293B;
        --muted: #64748B;
        --line: #E2E8F0;
        --shadow: 0 14px 36px rgba(15, 23, 42, .08);
        --accent-shadow: 0 18px 42px rgba(224, 86, 67, .20);
    }
    @keyframes pageSlideUp {
        from {opacity: 0; transform: translateY(18px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes softGlow {
        0%, 100% {box-shadow: 0 12px 32px rgba(224, 86, 67, .09);}
        50% {box-shadow: 0 18px 48px rgba(224, 86, 67, .16);}
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #F9FBFD 0%, var(--bg) 45%, #EEF2F6 100%) !important;
        color: var(--ink) !important;
    }
    .main .block-container {
        padding-top: 1.05rem;
        max-width: 1400px;
        animation: pageSlideUp .55s ease-out both;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #212B36 0%, #18222E 100%) !important;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * {color: #DDE3EA !important;}
    [data-testid="stSidebar"] h1 {color: #FFFFFF !important;}
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.12) !important;}
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: rgba(221, 227, 234, .48) !important;
        font-size: .78rem !important;
        line-height: 1.45;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 12px;
        padding: .28rem .45rem;
        transition: all .2s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.08);
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(224, 86, 67, .18);
        box-shadow: inset 4px 0 0 var(--accent);
    }
    h1, h2, h3 {
        letter-spacing: -.02em;
        color: var(--ink) !important;
    }
    p, span, div, label {color: var(--ink);}
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, #C84736 100%) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 24px rgba(224, 86, 67, .22);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 34px rgba(224, 86, 67, .30);
    }
    .hero-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(27, 46, 75, .08);
        background: #FFFFFF;
        color: var(--ink) !important;
        border-radius: 12px;
        padding: 1.28rem 1.45rem;
        margin: .25rem 0 1.15rem 0;
        box-shadow: 0 10px 26px rgba(27, 46, 75, .055);
    }
    .hero-card:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(180deg, var(--navy), var(--accent));
    }
    .hero-card * {color: #111827 !important;}
    .metric-card {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-left: 5px solid var(--red);
        background: var(--card);
        color: var(--ink) !important;
        border-radius: 16px;
        padding: 1rem 1.08rem;
        min-height: 106px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, .06);
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.015);
        border-color: rgba(224, 86, 67, .34);
        box-shadow: var(--accent-shadow);
    }
    .metric-card * {color: #111827 !important;}
    .metric-label {font-size: .92rem; color: #64748b !important;}
    .metric-value {font-size: 1.75rem; font-weight: 800; margin-top: .25rem; line-height: 1.12; color: var(--navy) !important;}
    .metric-card.dark {
        background: linear-gradient(145deg, #1B2E4B 0%, #223B60 100%);
        border-left-color: #E05643;
    }
    .metric-card.dark *, .metric-card.coral *, .metric-card.soft-coral * {color: #FFFFFF !important;}
    .metric-card.dark .small-note, .metric-card.coral .small-note, .metric-card.soft-coral .small-note {color: rgba(255,255,255,.82) !important;}
    .metric-card.dark .metric-label, .metric-card.coral .metric-label, .metric-card.soft-coral .metric-label {color: rgba(255,255,255,.78) !important;}
    .metric-card.dark .metric-value, .metric-card.coral .metric-value, .metric-card.soft-coral .metric-value {color: #FFFFFF !important;}
    .metric-card.coral {
        background: linear-gradient(135deg, #E05643 0%, #CF4A39 100%);
        border-left-color: #FFFFFF;
        box-shadow: 0 16px 42px rgba(224, 86, 67, .28);
    }
    .metric-card.soft-coral {
        background: linear-gradient(135deg, #E57362 0%, #D86656 100%);
        border-left-color: #FFFFFF;
    }
    .metric-card.wide {
        background: #FFFFFF;
        border-left-color: var(--navy);
    }
    .metric-card.wide .metric-value {font-size: 1.45rem; color: var(--ink) !important;}
    .tooltip-icon {
        float: right;
        display: inline-flex;
        width: 18px;
        height: 18px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        font-size: .72rem;
        font-weight: 800;
        background: rgba(255,255,255,.22);
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,.35);
        cursor: help;
    }
    .metric-card.wide .tooltip-icon, .metric-card:not(.dark):not(.coral):not(.soft-coral) .tooltip-icon {
        color: var(--accent) !important;
        background: #FFF3F0;
        border-color: #F5C1B8;
    }
    .action-card {
        border: 1px solid rgba(27, 46, 75, .10);
        border-left: 5px solid var(--accent);
        background: #FFFFFF;
        color: var(--ink) !important;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: .75rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, .05);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .action-card:hover {transform: translateY(-3px); box-shadow: 0 14px 32px rgba(27, 46, 75, .10);}
    .action-card * {color: #111827 !important;}
    .explain-box {
        border: 1px solid rgba(27, 46, 75, .10);
        background: #ffffff;
        color: var(--ink) !important;
        border-radius: 14px;
        padding: .9rem 1rem;
        margin: .65rem 0 1rem 0;
        box-shadow: 0 7px 20px rgba(15, 23, 42, .05);
    }
    .explain-box * {color: #111827 !important;}
    .small-note {font-size: .92rem; color: #64748b; line-height: 1.55;}
    .term-pill {
        display: inline-block;
        background: #FFF1F2;
        color: #B91C1C !important;
        border: 1px solid #FECDD3;
        border-radius: 999px;
        padding: .18rem .55rem;
        margin: .15rem .25rem .15rem 0;
        font-size: .88rem;
    }
    .ai-banner {
        border: 1px solid rgba(224, 86, 67, .32);
        background: linear-gradient(135deg, #FFF6F3 0%, #FFFFFF 52%, #F9E7E2 100%);
        color: var(--ink) !important;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        margin: .8rem 0 1rem 0;
        box-shadow: 0 16px 38px rgba(224, 86, 67, .12);
        position: relative;
        overflow: hidden;
    }
    .ai-banner:after {
        content: "";
        position: absolute;
        width: 140px;
        height: 140px;
        right: -42px;
        top: -58px;
        background: radial-gradient(circle, rgba(224, 86, 67, .22), transparent 68%);
    }
    .ai-banner b, .red-strong {color: var(--accent) !important; font-weight: 850;}
    .profile-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: .75rem 0 1.2rem 0;
    }
    .profile-card {
        background: #FFFFFF;
        border: 1px solid rgba(27, 46, 75, .10);
        border-top: 5px solid var(--accent);
        border-radius: 16px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 10px 26px rgba(27, 46, 75, .06);
        min-height: 230px;
    }
    .profile-card h4 {
        margin: 0 0 .55rem 0;
        color: var(--navy) !important;
        font-size: 1.05rem;
    }
    .profile-card .tag {
        display: inline-block;
        background: #FFF3F0;
        color: var(--accent) !important;
        border-radius: 999px;
        padding: .12rem .5rem;
        margin: .1rem .15rem .35rem 0;
        font-size: .82rem;
        font-weight: 700;
    }
    .profile-card p {
        margin: .28rem 0;
        line-height: 1.55;
        color: var(--ink) !important;
    }
    .profile-card b {color: var(--navy) !important;}
    .management-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-top: .8rem;
    }
    @media (max-width: 900px) {
        .profile-grid, .management-grid {grid-template-columns: 1fr;}
    }
    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid rgba(27, 46, 75, .08);
        border-radius: 12px;
        padding: .8rem .8rem .2rem .8rem;
        box-shadow: 0 10px 26px rgba(27, 46, 75, .06);
    }
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 28px rgba(15, 23, 42, .06);
        border: 1px solid #E2E8F0;
    }
    @media (max-width: 900px) {
        .main .block-container {padding-left: 1rem; padding-right: 1rem;}
        .metric-card {min-height: auto;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 讀檔工具
# -----------------------------
def safe_read_csv(path: Path | None, show_warning: bool = True) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        if show_warning:
            st.warning("尚未找到需要的 CSV，請確認 outputs 資料夾是否已有分析結果。")
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    if show_warning:
        st.warning(f"檔案讀取失敗：{path.name}")
    return None


def safe_read_txt(path: Path | None) -> Optional[str]:
    if path is None or not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


def safe_load_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        st.warning(f"尚未找到圖檔：{path.name}")
        return None
    try:
        return Image.open(path)
    except Exception:
        st.warning(f"圖檔無法載入：{path.name}")
        return None


def find_file(names: list[str]) -> Optional[Path]:
    for name in names:
        p = outputs_path / name
        if p.exists():
            return p
    return None


def parse_model_report(text: str | None) -> dict:
    result = {"attr": {}, "course": {}, "cm_attr": None, "cm_course": None}
    if not text:
        return result

    def block(label: str, next_label: str) -> str:
        m = re.search(rf"\[{label}.*?\](.*?)(?:\n\[{next_label}|\Z)", text, re.S)
        return m.group(1) if m else ""

    def metrics(part: str) -> dict:
        mapping = {
            "Accuracy": "預測準確度",
            "Precision": "命中率",
            "Recall": "找回率",
            "F1": "平衡分數",
            "ROC_AUC": "ROC-AUC",
            "PR_AUC": "PR-AUC",
        }
        out = {}
        for raw, zh in mapping.items():
            m = re.search(rf"{raw}\s*:\s*([0-9.]+)", part)
            if m:
                out[zh] = float(m.group(1))
        return out

    def cm(part: str):
        m = re.search(r"Confusion Matrix:\s*\[\[([0-9\s]+)\]\s*\[([0-9\s]+)\]\]", part, re.S)
        if not m:
            return None
        return np.array([[int(x) for x in m.group(1).split()], [int(x) for x in m.group(2).split()]])

    b1 = block("XGBoost_1A", "XGBoost_1B")
    b2 = block("XGBoost_1B", "MNL")
    result["attr"] = metrics(b1)
    result["course"] = metrics(b2)
    result["cm_attr"] = cm(b1)
    result["cm_course"] = cm(b2)
    return result


def section(title: str, desc: str = ""):
    st.markdown(f"## {title}")
    if desc:
        st.caption(desc)


def page_header(title: str, desc: str, action: str = ""):
    action_html = f"<br><b>現在要做：</b>{escape(action)}" if action else ""
    html = (
        f'<div class="hero-card">'
        f'<h1 style="margin:0 0 .35rem 0;">{escape(title)}</h1>'
        f'<div>{escape(desc)}{action_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def tooltip_icon(text: str) -> str:
    return f'<span class="tooltip-icon" title="{escape(text, quote=True)}">i</span>' if text else ""


def html_text(value) -> str:
    raw = str(value)
    allowed = ("<span", "</span>")
    if any(tag in raw for tag in allowed):
        return raw
    return escape(raw)


def html_block(tag: str, content: str, class_name: str = "") -> str:
    class_attr = f' class="{escape(class_name, quote=True)}"' if class_name else ""
    return f"<{tag}{class_attr}>{content}</{tag}>"


def html_card(label: str, value, note: str = "", variant: str = "", tooltip: str = "") -> str:
    safe_variant = escape(variant, quote=True)
    note_html = html_block("div", escape(note), "small-note") if note else ""
    return (
        f'<div class="metric-card {safe_variant}">'
        f'<div class="metric-label">{escape(label)}{tooltip_icon(tooltip)}</div>'
        f'<div class="metric-value">{html_text(value)}</div>'
        f'{note_html}'
        f'</div>'
    )


def metric_card(label: str, value, note: str = "", variant: str = "", tooltip: str = ""):
    st.markdown(html_card(label, value, note, variant, tooltip), unsafe_allow_html=True)


def executive_kpi_cards(items: list[tuple[str, str, str, str, str]]):
    cards = "".join(html_card(label, value, note, variant, tooltip) for label, value, note, variant, tooltip in items)
    html = (
        '<div style="display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); '
        'gap: 1rem; margin: .4rem 0 1.15rem 0;">'
        f"{cards}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def ai_banner(title: str, body: str):
    st.markdown(
        f'<div class="ai-banner"><b>{escape(title)}</b><br>{body}</div>',
        unsafe_allow_html=True,
    )


def segment_profile_cards():
    profiles = [
        {
            "title": "第一群：證照／多元進修型",
            "tags": ["多元探索", "價格敏感", "入門引導"],
            "interest": "興趣較分散，對證照、管理、財會與一般進修都有可能接觸。",
            "price": "偏好 3000 以下或低門檻方案。",
            "behavior": "報名頻次較低，需要更多信任與誘因。",
            "strategy": "用免費講座、體驗課、入門方案與廣度觸及先養名單。",
        },
        {
            "title": "第二群：國貿實務型",
            "tags": ["國貿實務", "證照需求", "高轉換"],
            "interest": "集中於國貿、進出口流程、貿易文件、報關與證照。",
            "price": "可接受 4000-5000，若看見實用價值可再提高。",
            "behavior": "報名意願較高，也較可能出現複購行為。",
            "strategy": "主推深度課程、組合方案與實務案例，搭配 LINE 或電話邀約。",
        },
        {
            "title": "第三群：商業語文型",
            "tags": ["商業語文", "職場應用", "深度課程"],
            "interest": "集中於商業語文、商務溝通、客戶應對與國際工作情境。",
            "price": "可接受 4000-5000，但需要清楚看到職場實用性。",
            "behavior": "報名頻次較高，適合推進階或系列課程。",
            "strategy": "強調職場立即可用、國際溝通力與客戶應對能力。",
        },
    ]
    cards = []
    for p in profiles:
        tags = "".join(f'<span class="tag">{escape(t)}</span>' for t in p["tags"])
        cards.append(
            f'<div class="profile-card"><h4>{escape(p["title"])}</h4>{tags}'
            f'<p><b>主要興趣：</b>{escape(p["interest"])}</p>'
            f'<p><b>價格敏感度：</b>{escape(p["price"])}</p>'
            f'<p><b>報名行為：</b>{escape(p["behavior"])}</p>'
            f'<p><b>落地策略：</b>{escape(p["strategy"])}</p></div>'
        )
    st.markdown(f'<div class="profile-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def management_cards():
    items = [
        ("先分眾，再投放", "會員不是同一種人。先用三個客群拆開，再設計不同課程主打與文案。"),
        ("先抓高潛力，再培養中低潛力", "高潛力用 LINE 或電話，中潛力用 EDM，低潛力用內容慢慢養。"),
        ("課程主打要說人話", "證照要說成職涯加值，國貿要說成案例實戰，線上要說成彈性學習。"),
        ("用成效資料回收優化", "每次投放後回填開信、點擊與報名，下一輪就能更知道哪個客群、通路與文案有效。"),
    ]
    cards = "".join(f'<div class="action-card"><b>{escape(t)}</b><br>{escape(b)}</div>' for t, b in items)
    st.markdown(f'<div class="management-grid">{cards}</div>', unsafe_allow_html=True)


def dataframe_with_progress(df: pd.DataFrame, *, height: int | None = None):
    config = {}
    if "預測報名機率" in df.columns:
        config["預測報名機率"] = st.column_config.ProgressColumn(
            "預測報名機率",
            help="模型估計這位會員對推薦課程的報名可能性；越滿代表越值得優先聯絡。",
            min_value=0.0,
            max_value=1.0,
            format="%.2f",
        )
    dataframe_kwargs = {
        "use_container_width": True,
        "hide_index": True,
        "column_config": config,
    }
    if height is not None:
        dataframe_kwargs["height"] = height
    st.dataframe(df, **dataframe_kwargs)


def explain_box(how: str, finding: str, action: str):
    st.markdown(
        f'<div class="explain-box"><b>怎麼看：</b>{escape(how)}<br>'
        f'<b>目前看到：</b>{escape(finding)}<br>'
        f'<b>下一步：</b>{escape(action)}</div>',
        unsafe_allow_html=True,
    )


def show_image(file_name: str, title: str, how: str, finding: str, action: str):
    st.markdown(f"### {title}")
    img = safe_load_image(outputs_path / file_name)
    if img is not None:
        st.image(img, use_container_width=True)
    explain_box(how, finding, action)


def safe_load_asset(file_name: str) -> Optional[Image.Image]:
    path = APP_DIR / "assets" / file_name
    if not path.exists():
        return None
    try:
        return Image.open(path)
    except Exception:
        return None


# -----------------------------
# 資料載入
# -----------------------------
model_report = safe_read_txt(outputs_path / "topic1_model_performance_report.txt")
metrics = parse_model_report(model_report)
xgb_imp_df = safe_read_csv(outputs_path / "topic1_xgboost_feature_importance.csv", False)
shap_df = safe_read_csv(outputs_path / "topic1_shap_importance.csv", False)
top1_df = safe_read_csv(outputs_path / "topic1_recommendation_top1.csv", False)
top3_df = safe_read_csv(outputs_path / "topic1_recommendation_top3.csv", False)
cluster_df = safe_read_csv(outputs_path / "topic2_cluster_profile.csv", False)
chi_df = safe_read_csv(outputs_path / "topic2_chi_square_results.csv", False)


# -----------------------------
# 行銷邏輯
# -----------------------------
SEGMENTS = pd.DataFrame(
    [
        {
            "客群": "商業語文／運務導向型",
            "輪廓": "重視商務溝通、客戶應對、跨國工作與運務場景。",
            "主要興趣": "商業語文、商務溝通",
            "建議賣點": "國際溝通、職場應用、客戶應對",
            "建議通路": "EDM、LINE",
        },
        {
            "客群": "國貿實務導向型",
            "輪廓": "重視進出口流程、貿易文件、報關應用與案例實戰。",
            "主要興趣": "國貿實務、國際市場",
            "建議賣點": "案例實戰、流程拆解、即戰力",
            "建議通路": "LINE、電話邀約",
        },
        {
            "客群": "證照／多元進修型",
            "輪廓": "重視考照、履歷加分、轉職升遷與多元學習。",
            "主要興趣": "專業證照、財務會計、管理進修",
            "建議賣點": "職涯加值、證照認證、高 CP 值",
            "建議通路": "Facebook、Instagram、再行銷廣告",
        },
    ]
)


COURSE_NAME_MAP = {
    "1": "課程 1｜實體證照實務班",
    "2": "課程 2｜線上證照實務班",
    "3": "課程 3｜線上國貿實務班",
    "4": "課程 4｜實體商業語文班",
    "5": "課程 5｜實體國貿實務班",
    "6": "課程 6｜線上國貿實務班",
    "7": "課程 7｜線上數位實務班",
    "8": "課程 8｜線上稅務實務班",
}


def course_name(course_id) -> str:
    if pd.isna(course_id):
        return "未指定課程"
    text = str(course_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return COURSE_NAME_MAP.get(text, f"課程 {text}")


def priority_label(prob: float) -> str:
    if pd.isna(prob):
        return "低"
    if prob >= 0.7:
        return "高"
    if prob >= 0.5:
        return "中"
    return "低"


def infer_segment(row: pd.Series) -> str:
    reason = str(row.get("推薦原因", ""))
    course_id = str(row.get("推薦課程ID", ""))
    if course_id.endswith(".0"):
        course_id = course_id[:-2]
    if course_id in {"3", "5", "6", "8"}:
        return "國貿實務導向型"
    if course_id in {"4", "7"}:
        return "商業語文／運務導向型"
    if "國貿" in reason:
        return "國貿實務導向型"
    if "商業語文" in reason or "數位" in reason:
        return "商業語文／運務導向型"
    return "證照／多元進修型"


def channel_for(priority: str, segment: str) -> str:
    if priority == "高":
        return "LINE＋電話邀約"
    if priority == "中":
        return "EDM＋再行銷廣告"
    if "商業語文" in segment:
        return "EDM"
    return "社群內容培養"


def copy_for(row: pd.Series) -> str:
    segment = row.get("客群", "會員")
    course = row.get("推薦課程名稱", course_name(row.get("推薦課程ID", "")))
    if "國貿" in segment:
        return f"{course}：用案例快速補強進出口流程與職場即戰力。"
    if "商業語文" in segment:
        return f"{course}：強化商務溝通與客戶應對，讓工作場景更好用。"
    return f"{course}：主打證照與職涯加值，適合想提升履歷競爭力的會員。"


def prepare_recommendations(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    d = df.copy()
    if "預測報名機率" in d.columns:
        d["預測報名機率"] = pd.to_numeric(d["預測報名機率"], errors="coerce")
    else:
        d["預測報名機率"] = np.nan
    if "推薦課程ID" in d.columns:
        d["推薦課程名稱"] = d["推薦課程ID"].apply(course_name)
    elif "推薦課程名稱" not in d.columns:
        d["推薦課程名稱"] = "未指定課程"
    d["優先級"] = d["預測報名機率"].apply(priority_label)
    d["客群"] = d.apply(infer_segment, axis=1)
    d["建議通路"] = d.apply(lambda r: channel_for(r["優先級"], r["客群"]), axis=1)
    d["建議文案"] = d.apply(copy_for, axis=1)
    return d


top1_exec = prepare_recommendations(top1_df)
top3_exec = prepare_recommendations(top3_df)


def marketing_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["證照", "職涯加值", "強調考照、專業認證、履歷加分。"],
            ["國貿實務", "案例實戰", "強調進出口流程、貿易文件、報關應用。"],
            ["線上", "彈性學習", "強調可回放、不受地點限制、下班後也能學。"],
            ["價格帶", "高 CP 值", "強調小額投資、短期進修、職涯升級。"],
        ],
        columns=["模型看到的因素", "行銷翻譯", "可使用的文案方向"],
    )


def generate_copy(course, segment, appeal, platform, tone) -> dict:
    appeal_map = {
        "證照": "考照、專業認證、履歷加分與職涯競爭力",
        "職涯加值": "升遷、轉職、加薪與履歷亮點",
        "國貿實務": "進出口流程、貿易文件、報關應用與案例實戰",
        "商業語文": "商務情境、客戶應對、國際溝通與跨國工作",
        "高 CP 值": "小額投資、高價值回報、短期進修與職涯升級",
        "線上彈性": "可回放、不受地點限制、下班後學習與彈性進修",
        "實務案例": "真實案例、工作流程拆解與立即可用的技能",
    }
    core = appeal_map.get(appeal, "課程價值與職場應用")
    title = f"給{segment}的{course}：把{appeal}變成下一個職涯優勢"
    social = f"想讓進修真的用在工作上？{course} 適合{segment}，本期主打{core}。用一門課補上工作需要的能力，讓學習更接近職場成果。"
    line = f"{course} 推薦給你：主打{appeal}，適合{segment}。想看課程內容與適合對象嗎？點開了解本期推薦。"
    edm = f"【{course}】給{segment}的進修方案：{appeal}現在開始"
    script = f"0-3秒：點出工作痛點。3-8秒：介紹 {course}。8-12秒：強調{core}。12-15秒：提醒立即了解課程與名額。"
    prompt = f"生成一張{platform}廣告視覺，主題為{course}，客群為{segment}，語氣{tone}，畫面呈現專業學習情境、清楚課程標題、報名按鈕，重點包含{core}。"
    return {"廣告標題": title, "社群貼文": social, "LINE 推播文": line, "EDM 標題": edm, "15 秒短影片腳本": script, "AI 圖像 / 影片生成 prompt": prompt}


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("行銷工作台")
page = st.sidebar.radio(
    "請選擇功能",
    [
        "客群洞察與市場區隔",
        "高潛力推薦名單",
        "課程推薦系統",
        "行銷總覽",
        "AI 文案與素材產生器",
        "成效追蹤與 A/B Test",
        "模型解釋與預測方法",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"資料來源：{outputs_path.resolve()}")


# -----------------------------
# 1. 行銷總覽
# -----------------------------
if page == "行銷總覽":
    page_header(
        "行銷總覽",
        "這頁只回答三件事：今天要優先聯絡誰、主推哪門課、用什麼賣點說服會員。",
        "先處理高潛力名單，再依客群套用文案與通路。",
    )

    learners = top1_exec["學員ID"].nunique() if "學員ID" in top1_exec.columns else 0
    courses = top3_exec["推薦課程ID"].nunique() if "推薦課程ID" in top3_exec.columns else 0
    high_count = int((top1_exec["優先級"] == "高").sum()) if not top1_exec.empty else 0
    mid_count = int((top1_exec["優先級"] == "中").sum()) if not top1_exec.empty else 0
    top_course = top1_exec["推薦課程名稱"].mode().iloc[0] if "推薦課程名稱" in top1_exec.columns and not top1_exec.empty else "N/A"
    roc = metrics.get("course", {}).get("ROC-AUC", np.nan)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.8])
    with c1:
        metric_card("總會員數", learners, "目前可被推薦的會員", "dark")
    with c2:
        metric_card("課程數", courses, "模型可推薦的課程", "dark")
    with c3:
        metric_card(
            "高潛力名單",
            high_count,
            "建議先用 LINE 或電話",
            "coral",
            "模型判斷比較可能報名的人，適合優先聯絡。",
        )
    with c4:
        metric_card(
            "中潛力名單",
            mid_count,
            "建議用 EDM 培養",
            "soft-coral",
            "有興趣但需要更多資訊的人，適合用 EDM 或再行銷慢慢培養。",
        )
    with c5:
        metric_card(
            "本週主推課程",
            top_course,
            f"最多會員被推薦的課程；ROC-AUC {roc:.3f}" if not pd.isna(roc) else "最多會員被推薦的課程",
            "wide",
            "ROC-AUC 是模型排序能力，越接近 1，越能把可能報名的人排在前面。",
        )

    if not top1_exec.empty:
        st.markdown("<h3 style='margin-bottom: 1.1rem;'>今天先看這三張圖</h3>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            priority_counts = top1_exec["優先級"].value_counts().rename_axis("優先級").reset_index(name="人數")
            fig = px.bar(priority_counts, x="優先級", y="人數", color="優先級", title="名單優先級分布")
            st.plotly_chart(fig, use_container_width=True)
            explain_box("看高、中、低潛力各有多少人。", "高潛力名單是最值得先處理的一群。", "先排 LINE 或電話聯絡高潛力會員。")
        with g2:
            course_counts = top1_exec["推薦課程名稱"].value_counts().head(8).rename_axis("課程名稱").reset_index(name="推薦人數")
            fig = px.bar(course_counts, x="推薦人數", y="課程名稱", orientation="h", title="最多人被推薦的課程")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            explain_box("看哪門課被最多會員推薦。", f"目前最值得主推的是 {top_course}。", "把它放在 EDM、LINE 或首頁主視覺。")
        with g3:
            segment_source = top3_exec if not top3_exec.empty else top1_exec
            segment_counts = segment_source["客群"].value_counts().rename_axis("客群").reset_index(name="人數")
            fig = px.pie(segment_counts, names="客群", values="人數", title="會員主要客群")
            st.plotly_chart(fig, use_container_width=True)
            explain_box("看會員主要分成哪幾群。", "分群重點是興趣與學習目的，不是性別或職稱。", "每一群用不同文案，不要同一封訊息打全部人。")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### 主要行銷發現")
        for title, body in [
            ("三個主要客群", "商業語文／運務導向型、國貿實務導向型、證照／多元進修型。"),
            ("主打賣點要翻成顧客語言", "證照＝職涯加值，國貿實務＝案例實戰，線上＝彈性學習。"),
            ("推薦名單可直接分層投放", "高潛力用 LINE 與電話，中潛力用 EDM，低潛力用內容培養。"),
            ("分眾要看興趣", "性別與職稱不是主要差異，興趣更能說明會員想學什麼。"),
        ]:
            st.markdown(f'<div class="action-card"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)
    with right:
        points = marketing_points()
        st.markdown("### 模型發現轉成行銷語言")
        dataframe_with_progress(points)

    st.markdown("### 管理意涵解釋")
    management_cards()
    ai_banner(
        "高層結論",
        "這套系統的價值不是單純看模型分數，而是把會員分群、課程推薦、文案產出與成效追蹤串成一條行銷作業流程。"
        "管理者可以用它決定本週主推課程、優先聯絡名單、投放通路與下一輪優化方向。",
    )


# -----------------------------
# 2. 高潛力推薦名單
# -----------------------------
elif page == "高潛力推薦名單":
    page_header(
        "高潛力推薦名單",
        "這頁是業務與行銷最常用的名單頁。你可以直接篩出某門課、某個客群、某個機率以上的會員。",
        "下載篩選後名單，拿去做 LINE、EDM、電話或再行銷。",
    )
    source = st.radio("名單類型", ["Top1：每位會員最推薦一門課", "Top3：每位會員前三推薦課"], horizontal=True)
    data = top1_exec if source.startswith("Top1") else top3_exec
    if data.empty:
        st.warning("找不到推薦名單，請確認 outputs 裡有 topic1_recommendation_top1.csv 或 topic1_recommendation_top3.csv。")
        st.stop()

    f1, f2, f3, f4 = st.columns(4)
    course_opts = sorted(data["推薦課程名稱"].dropna().unique().tolist()) if "推薦課程名稱" in data.columns else []
    selected_courses = f1.multiselect("課程名稱", course_opts, default=course_opts)
    threshold = f2.slider("預測機率門檻", 0.0, 1.0, 0.0, 0.01)
    selected_priority = f3.multiselect("優先級", ["高", "中", "低"], default=["高", "中", "低"])
    selected_segments = f4.multiselect("客群", SEGMENTS["客群"].tolist(), default=SEGMENTS["客群"].tolist())

    filtered = data.copy()
    filtered = filtered[filtered["預測報名機率"] >= threshold]
    if selected_courses:
        filtered = filtered[filtered["推薦課程名稱"].isin(selected_courses)]
    if selected_priority:
        filtered = filtered[filtered["優先級"].isin(selected_priority)]
    if selected_segments:
        filtered = filtered[filtered["客群"].isin(selected_segments)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("篩選後名單", len(filtered))
    k2.metric("高潛力", int((filtered["優先級"] == "高").sum()))
    k3.metric("中潛力", int((filtered["優先級"] == "中").sum()))
    k4.metric("低潛力", int((filtered["優先級"] == "低").sum()))

    if filtered.empty:
        st.warning("目前篩選條件下沒有名單。請降低預測機率門檻，或放寬課程、優先級、客群篩選。")
        st.stop()

    cols = [c for c in ["學員ID", "推薦課程名稱", "預測報名機率", "優先級", "客群", "推薦原因", "建議通路", "建議文案"] if c in filtered.columns]
    dataframe_with_progress(filtered[cols], height=420)

    left, right = st.columns(2)
    with left:
        course_counts = filtered["推薦課程名稱"].value_counts().rename_axis("推薦課程名稱").reset_index(name="推薦次數")
        fig = px.bar(course_counts, x="推薦課程名稱", y="推薦次數", title="各課程被推薦次數")
        st.plotly_chart(fig, use_container_width=True)
        explain_box("柱子越高，代表越多會員被推薦該課程。", "可快速找出本週主推課程。", "推薦次數最高的課程適合做 EDM 主推。")
    with right:
        fig = px.histogram(filtered, x="預測報名機率", nbins=20, title="推薦機率分布")
        st.plotly_chart(fig, use_container_width=True)
        explain_box("越靠右代表會員越可能報名。", "高機率名單適合優先聯絡。", "可把門檻調到 0.5 或 0.7 做分層投放。")

    csv = filtered[cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載目前篩選後名單 CSV", csv, "filtered_marketing_leads.csv", "text/csv")


# -----------------------------
# 3. 課程推薦系統
# -----------------------------
elif page == "課程推薦系統":
    page_header(
        "課程推薦系統",
        "選一門課後，系統會整理這門課最適合打哪些會員、主打什麼賣點、用什麼通路。",
        "適合課程開賣前做投放簡報，也適合每週行銷會議使用。",
    )

    if top3_exec.empty:
        st.warning("找不到 Top3 推薦名單。")
        st.stop()

    course_list = sorted(top3_exec["推薦課程名稱"].dropna().unique().tolist())
    course = st.selectbox("選擇課程", course_list)
    d = top3_exec[top3_exec["推薦課程名稱"] == course].copy()
    high = d[d["優先級"] == "高"]
    target = high if not high.empty else d

    main_segment = target["客群"].mode().iloc[0] if not target.empty else "N/A"
    avg_prob = target["預測報名機率"].mean() if not target.empty else np.nan
    main_channel = target["建議通路"].mode().iloc[0] if not target.empty else "N/A"
    main_copy = target["建議文案"].mode().iloc[0] if not target.empty else "N/A"
    segment_row = SEGMENTS[SEGMENTS["客群"] == main_segment]
    main_interest = segment_row["主要興趣"].iloc[0] if not segment_row.empty else "N/A"
    main_appeal = segment_row["建議賣點"].iloc[0] if not segment_row.empty else "N/A"

    executive_kpi_cards(
        [
            ("高潛力學員數", f"<span class='red-strong'>{len(high)}</span>", "建議今天優先聯絡的人數", "coral", "模型判斷比較可能報名的人，適合優先聯絡。"),
            ("平均預測機率", f"<span class='red-strong'>{avg_prob:.2f}</span>" if not pd.isna(avg_prob) else "N/A", "越高代表越值得優先投放", "dark", "模型估計會員對這門課的平均報名可能性。"),
            ("主要客群", f"<span class='red-strong'>{main_segment}</span>", "這門課最適合的會員類型", "soft-coral", "依推薦課程與會員興趣整理出的分眾。"),
            ("建議通路", f"<span class='red-strong'>{main_channel}</span>", "最適合啟動的溝通方式", "dark", "建議優先使用的會員觸及管道。"),
        ]
    )

    st.markdown("### 課程行銷策略摘要")
    ai_banner(
        "AI 策略摘要",
        f"這門課目前最適合先投放給 <b>{main_segment}</b>。主要興趣是 <b>{main_interest}</b>，"
        f"建議賣點是 <b>{main_appeal}</b>。通路可先使用 <b>{main_channel}</b>。"
        f"<br>文案方向可用：「{main_copy}」",
    )

    st.markdown("### 該課程高潛力學員表格")
    show_cols = [c for c in ["學員ID", "推薦課程名稱", "預測報名機率", "優先級", "客群", "建議通路", "建議文案"] if c in target.columns]
    dataframe_with_progress(target.sort_values("預測報名機率", ascending=False)[show_cols], height=420)

    fig = px.histogram(d, x="客群", color="優先級", title="該課程推薦客群分布")
    st.plotly_chart(fig, use_container_width=True)
    explain_box("看這門課主要被推薦給哪些客群。", "可判斷這門課要用哪一種行銷主軸。", "最大客群先做主文案，其餘客群做分眾素材。")


# -----------------------------
# 4. 客群洞察與市場區隔
# -----------------------------
elif page == "客群洞察與市場區隔":
    page_header(
        "客群洞察與市場區隔",
        "這頁把分群結果翻成行銷人看得懂的客群輪廓：誰是主要客群、在意什麼、該用什麼文案。",
        "不要用同一套文案打所有人，先依客群改主打賣點。",
    )
    st.markdown("### 三個可執行客群輪廓")
    segment_profile_cards()
    with st.expander("查看客群輪廓表格"):
        dataframe_with_progress(SEGMENTS)

    if cluster_df is not None:
        raw = cluster_df.copy()
        if "cluster" in raw.columns:
            raw["行銷命名"] = raw["cluster"].map({0: "商業語文／運務導向型", 1: "多元進修／一般潛力型"}).fillna("其他客群")
        st.markdown("### 原始分群結果")
        dataframe_with_progress(raw)

    strategy_plot = SEGMENTS.copy()
    strategy_plot["投放優先順序"] = [3, 2, 1]
    fig = px.bar(
        strategy_plot,
        x="客群",
        y="投放優先順序",
        color="建議通路",
        text="主要興趣",
        title="客群 × 行銷策略矩陣",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=True, yaxis_title="建議優先處理程度")
    st.plotly_chart(fig, use_container_width=True)
    explain_box("這裡不是看分數，而是把客群轉成行銷打法。", "每一群都有自己的賣點與通路。", "行銷素材至少準備三套，不要所有人都用同一段文案。")

    matrix = pd.DataFrame(
        [
            ["商業語文／運務導向型", "商務溝通更順，跨國工作更有底氣", "EDM", "商業語文、職場應用"],
            ["國貿實務導向型", "用案例學會進出口流程與文件應用", "LINE＋電話", "國貿實務、案例實戰"],
            ["證照／多元進修型", "把證照變成履歷亮點與職涯籌碼", "社群廣告", "證照、職涯加值"],
        ],
        columns=["客群", "文案主軸", "優先通路", "推薦課程方向"],
    )
    st.markdown("### 客群 × 文案矩陣")
    dataframe_with_progress(matrix)

    img_cols = st.columns(3)
    with img_cols[0]:
        show_image("kmeans_elbow_plot.png", "分群數參考圖", "看分群數增加後是否還有明顯改善。", "目前結果可支援客群分眾。", "用於判斷分群是否需要再細切。")
    with img_cols[1]:
        show_image("kmeans_silhouette_plot.png", "分群清楚度", "分數越高代表客群越分得開。", "原始分群以 K=2 較穩。", "企業端再整理成三個行銷輪廓，方便執行。")
    with img_cols[2]:
        show_image("cluster_profile_heatmap.png", "客群特徵熱圖", "顏色越深代表特徵越明顯。", "可快速看每群代表特徵。", "用來決定文案要強調哪個需求。")

    if chi_df is not None:
        display = chi_df.rename(columns={"variable": "檢定項目", "p_value": "p-value", "significant(p<0.05)": "是否顯著"}).copy()
        display["是否顯著"] = display["是否顯著"].map({1: "顯著", 0: "不顯著"})
        st.markdown("### 客群差異來源")
        dataframe_with_progress(display[[c for c in ["檢定項目", "p-value", "是否顯著"] if c in display.columns]])
        explain_box("p-value 低於 0.05 代表該特徵能區分客群。", "主要興趣與國貿實務興趣較有區分力。", "分眾行銷先看興趣，再看職稱。")


# -----------------------------
# 5. AI 文案與素材產生器
# -----------------------------
elif page == "AI 文案與素材產生器":
    page_header(
        "AI 文案與素材產生器",
        "根據課程、客群、賣點、平台和語氣，自動產生可修改的文案初稿。",
        "先用這裡產生第一版，再由行銷人員調整品牌語氣。",
    )

    c1, c2, c3 = st.columns(3)
    courses = sorted(top3_exec["推薦課程名稱"].dropna().unique().tolist()) if not top3_exec.empty else ["課程 2｜線上證照實務班"]
    course = c1.selectbox("課程", courses)
    segment = c2.selectbox("客群", SEGMENTS["客群"].tolist())
    appeal = c3.selectbox("主打賣點", ["證照", "職涯加值", "國貿實務", "商業語文", "高 CP 值", "線上彈性", "實務案例"])
    c4, c5 = st.columns(2)
    platform = c4.selectbox("投放平台", ["Facebook", "Instagram", "LINE", "EDM", "YouTube Shorts"])
    tone = c5.selectbox("文案語氣", ["專業", "親切", "急迫", "職涯導向", "年輕活潑"])

    output = generate_copy(course, segment, appeal, platform, tone)
    for title, text in output.items():
        st.text_area(title, text, height=110 if "腳本" not in title else 150)
    st.info("此文案是初稿，正式投放前建議依品牌語氣調整，並用 A/B Test 驗證成效。")

    st.markdown("### AI 圖像 Prompt 生成範例")
    st.caption("以下範例可直接複製到 ChatGPT、Canva、Midjourney 或其他圖像生成工具，再依課程名稱與客群調整。")

    prompt_1 = """請生成一張 1:1 方形課程廣告圖，風格為高質感深藍金色商務教育廣告。
主題：課程 1｜實體證照實務班
客群：商業語文／運務導向型
主視覺：右側為專業講師在實體教室授課，學員坐在會議桌前聽講、筆電與教材可見。
版面：左側深藍漸層大標題，右側實拍感教室場景，下方白色資訊卡區塊放 4 個 icon 賣點。
標題文字：課程 1｜實體證照實務班
副標：實務導向學習，提升職場專業力
賣點 icon：考照準備、專業認證、履歷加分、職涯競爭力
CTA 按鈕：立即報名
色彩：深海軍藍、金色、白色，高級商務感，字體清楚、排版整齊、留白乾淨。"""

    prompt_2 = """請生成一張 1:1 方形線上課程廣告圖，風格為高質感深藍金色 SaaS 教育廣告。
主題：課程 3｜線上國貿實務班
客群：國貿實務導向型
主視覺：右側是一位學員戴耳機在電腦前上線上課，螢幕中有講師與國際貿易實務簡報。
版面：左側深藍漸層大標題，右側線上學習情境，下方白色資訊卡呈現核心學習內容。
標題文字：課程 3｜線上國貿實務班
副標：掌握國貿核心實務，強化職場即戰力
賣點 icon：進出口流程、貿易文件、報關應用、案例實戰
CTA 按鈕：立即報名
色彩：深海軍藍、金色、白色，專業、可信任、線上彈性學習，文字清楚可讀。"""

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        img = safe_load_asset("prompt_example_certificate.png")
        if img is not None:
            st.image(img, caption="範例 1：實體證照實務班視覺", use_container_width=True)
        st.text_area("範例 Prompt 1：實體證照實務班", prompt_1, height=260)
    with img_col2:
        img = safe_load_asset("prompt_example_online_trade.png")
        if img is not None:
            st.image(img, caption="範例 2：線上國貿實務班視覺", use_container_width=True)
        st.text_area("範例 Prompt 2：線上國貿實務班", prompt_2, height=260)


# -----------------------------
# 6. 成效追蹤與 A/B Test
# -----------------------------
elif page == "成效追蹤與 A/B Test":
    page_header(
        "成效追蹤與 A/B Test",
        "這頁用來追蹤投放後的結果，看看哪個通路、哪個客群、哪種文案比較有效。",
        "每次活動結束後上傳 campaign_result.csv，累積下一次優化依據。",
    )
    with st.expander("成效指標怎麼看"):
        st.markdown(
            """
            - **開信率**：收到訊息的人裡，有多少人打開來看。
            - **點擊率**：收到訊息的人裡，有多少人點了連結。
            - **報名率**：收到訊息的人裡，有多少人完成報名。
            - **轉換率**：點擊連結的人裡，有多少人最後報名。
            """
        )

    sample = pd.DataFrame(
        [
            ["C001", 2, "證照／多元進修型", "LINE", 100, 78, 31, 8],
            ["C002", 2, "國貿實務導向型", "EDM", 120, 72, 25, 6],
        ],
        columns=["campaign_id", "課程ID", "客群", "通路", "發送人數", "開信數", "點擊數", "報名數"],
    )
    upload = st.file_uploader("上傳 campaign_result.csv", type=["csv"])
    if upload is None:
        st.info("尚未上傳成效資料。企業可先依照下方 Sample 欄位建立 campaign_result.csv，再上傳到此頁。")
        st.markdown("### campaign_result.csv 上傳格式 Sample")
        st.caption("欄位請維持相同名稱：campaign_id、課程ID、客群、通路、發送人數、開信數、點擊數、報名數。")
        result = sample.copy()
        dataframe_with_progress(sample)
    else:
        result = pd.read_csv(upload)

    required = ["campaign_id", "課程ID", "客群", "通路", "發送人數", "開信數", "點擊數", "報名數"]
    missing = [c for c in required if c not in result.columns]
    if missing:
        st.warning(f"缺少欄位：{', '.join(missing)}")
        st.stop()

    for col in ["發送人數", "開信數", "點擊數", "報名數"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    result["開信率"] = result["開信數"] / result["發送人數"].replace(0, np.nan)
    result["點擊率"] = result["點擊數"] / result["發送人數"].replace(0, np.nan)
    result["報名率"] = result["報名數"] / result["發送人數"].replace(0, np.nan)
    result["轉換率"] = result["報名數"] / result["點擊數"].replace(0, np.nan)

    st.markdown("### 成效表")
    dataframe_with_progress(result)
    fig = px.bar(result, x="campaign_id", y=["開信率", "點擊率", "報名率", "轉換率"], barmode="group", title="不同活動成效比較")
    st.plotly_chart(fig, use_container_width=True)
    explain_box("比較不同 campaign 的開信、點擊、報名與轉換。", "可找出哪個通路或文案更有效。", "未來可把結果回填，讓推薦模型和分眾規則更準。")


# -----------------------------
# 7. 模型解釋與預測方法
# -----------------------------
elif page == "模型解釋與預測方法":
    page_header(
        "模型解釋與預測方法",
        "這頁保留模型圖表，但只用來回答：哪些因素會影響推薦、要怎麼翻成行銷賣點。",
        "模型是輔助工具，不是自動決策工具。",
    )

    section("指標怎麼用")
    st.markdown(
        """
        - **預測準確度**：整體猜對比例，但不適合作為唯一判斷。
        - **找回率**：真正會報名的人裡，模型抓到多少。找名單時很重要。
        - **ROC-AUC**：排序能力，代表模型能不能把比較可能報名的人排前面。
        - **SHAP**：解釋每個因素如何影響推薦結果。
        - **Feature Importance**：模型最常使用哪些因素做判斷。
        """
    )

    if shap_df is not None:
        top = shap_df.sort_values("mean_abs_shap", ascending=False).head(20)
        fig = px.bar(top.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", color="model", orientation="h", title="影響推薦的前 20 個因素")
        st.plotly_chart(fig, use_container_width=True)
        explain_box("越長代表越常影響推薦結果。", "證照、價格區間、課程型態和興趣是主要訊號。", "把它們翻成職涯加值、案例實戰、彈性學習等行銷賣點。")

    st.markdown("### 行銷翻譯表")
    dataframe_with_progress(marketing_points())

    tab1, tab2 = st.tabs(["課程條件與個人偏好", "推薦模型與成效"])
    with tab1:
        show_image("shap_bar_attribute_model.png", "SHAP Bar：課程條件", "看哪些課程條件影響最大。", "證照與價格是重要因素。", "課程頁要優先說清楚證照價值與價格理由。")
        show_image("shap_summary_attribute_model.png", "SHAP Summary：課程條件", "右側通常代表推高報名機率，左側代表降低。", "可判斷哪些特徵是加分或扣分。", "用於調整課程包裝與廣告訊息。")
        show_image("xgboost_feature_importance_attribute_model.png", "XGBoost 重要因素：課程條件", "看模型常用哪些欄位預測。", "證照、價格、時段與課程形式被頻繁使用。", "把這些資訊放在課程頁首屏。")
    with tab2:
        show_image("shap_bar_course_dummy_model.png", "SHAP Bar：推薦模型", "看推薦模型最在意哪些因素。", "課程別與課程屬性都會影響推薦。", "可用來解釋推薦原因。")
        show_image("shap_summary_course_dummy_model.png", "SHAP Summary：推薦模型", "看特徵值高低如何推動推薦機率。", "不同課程對不同會員吸引力不同。", "推薦名單可搭配不同客群文案。")
        show_image("xgboost_feature_importance_course_dummy_model.png", "XGBoost 重要因素：推薦模型", "看推薦模型最常使用哪些因素。", "證照、價格帶、課程別是核心。", "先投放高分課程，再針對客群調整話術。")

    st.warning("模型只作為行銷輔助，不應完全自動化決策。正式投放前仍需搭配預算、課程檔期、會員互動紀錄與品牌語氣判斷。")
