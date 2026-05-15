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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="課程精準行銷與會員推薦儀表板",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
outputs_path = APP_DIR / "outputs"
if not outputs_path.exists():
    outputs_path = Path.cwd() / "outputs"


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; max-width: 1360px;}
    h1, h2, h3 {letter-spacing: 0;}
    .action-card {
        border: 1px solid #dbe3ef;
        border-left: 5px solid #2563eb;
        background: #f8fbff;
        color: #111827 !important;
        border-radius: 8px;
        padding: .95rem 1.1rem;
        margin-bottom: .75rem;
    }
    .action-card * {color: #111827 !important;}
    .explain-box {
        border: 1px solid #dbe3ef;
        background: #ffffff;
        color: #111827 !important;
        border-radius: 8px;
        padding: .9rem 1rem;
        margin: .65rem 0 1rem 0;
    }
    .explain-box * {color: #111827 !important;}
    .small-note {font-size: .92rem; color: #64748b; line-height: 1.55;}
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


def explain_box(how: str, finding: str, action: str):
    st.markdown(
        f"""
        <div class="explain-box">
        <b>怎麼看：</b>{how}<br>
        <b>目前看到：</b>{finding}<br>
        <b>下一步：</b>{action}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_image(file_name: str, title: str, how: str, finding: str, action: str):
    st.markdown(f"### {title}")
    img = safe_load_image(outputs_path / file_name)
    if img is not None:
        st.image(img, use_container_width=True)
    explain_box(how, finding, action)


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
    if "證照" in reason:
        return "證照／多元進修型"
    if course_id in {"2", "3"}:
        return "國貿實務導向型"
    if course_id in {"4", "7"}:
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
    course = row.get("推薦課程ID", "")
    if "國貿" in segment:
        return f"推薦課程 {course}：用案例快速補強進出口流程與職場即戰力。"
    if "商業語文" in segment:
        return f"推薦課程 {course}：強化商務溝通與客戶應對，讓工作場景更好用。"
    return f"推薦課程 {course}：主打證照與職涯加值，適合想提升履歷競爭力的會員。"


def prepare_recommendations(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    d = df.copy()
    if "預測報名機率" in d.columns:
        d["預測報名機率"] = pd.to_numeric(d["預測報名機率"], errors="coerce")
    else:
        d["預測報名機率"] = np.nan
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
        "行銷總覽",
        "高潛力推薦名單",
        "單一課程作戰室",
        "客群洞察與市場區隔",
        "AI 文案與素材產生器",
        "成效追蹤與 A/B Test",
        "模型解釋與安全使用",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"資料來源：{outputs_path.resolve()}")


# -----------------------------
# 1. 行銷總覽
# -----------------------------
if page == "行銷總覽":
    st.title("行銷總覽")
    st.caption("把模型結果轉成今天可以執行的會員名單、課程主打與通路建議。")

    learners = top1_exec["學員ID"].nunique() if "學員ID" in top1_exec.columns else 0
    courses = top3_exec["推薦課程ID"].nunique() if "推薦課程ID" in top3_exec.columns else 0
    high_count = int((top1_exec["優先級"] == "高").sum()) if not top1_exec.empty else 0
    mid_count = int((top1_exec["優先級"] == "中").sum()) if not top1_exec.empty else 0
    top_course = top1_exec["推薦課程ID"].mode().iloc[0] if "推薦課程ID" in top1_exec.columns and not top1_exec.empty else "N/A"
    roc = metrics.get("course", {}).get("ROC-AUC", np.nan)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("總會員數", learners)
    c2.metric("課程數", courses)
    c3.metric("高潛力名單", high_count)
    c4.metric("中潛力名單", mid_count)
    c5.metric("Top 推薦課程", top_course)
    c6.metric("排序能力 ROC-AUC", f"{roc:.3f}" if not pd.isna(roc) else "N/A")

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
        st.dataframe(points, use_container_width=True, hide_index=True)

    if not top1_exec.empty:
        fig = px.histogram(top1_exec, x="優先級", color="優先級", title="推薦名單優先級分布")
        st.plotly_chart(fig, use_container_width=True)
        explain_box("看高、中、低潛力名單各有多少人。", "這能決定本週行銷資源要放在哪一層會員。", "先處理高潛力，再用 EDM 培養中低潛力。")


# -----------------------------
# 2. 高潛力推薦名單
# -----------------------------
elif page == "高潛力推薦名單":
    st.title("高潛力推薦名單")
    st.caption("把推薦機率轉成可執行的會員聯絡清單。")

    source = st.radio("名單類型", ["Top1：每位會員最推薦一門課", "Top3：每位會員前三推薦課"], horizontal=True)
    data = top1_exec if source.startswith("Top1") else top3_exec
    if data.empty:
        st.warning("找不到推薦名單，請確認 outputs 裡有 topic1_recommendation_top1.csv 或 topic1_recommendation_top3.csv。")
        st.stop()

    f1, f2, f3, f4 = st.columns(4)
    course_opts = sorted(data["推薦課程ID"].dropna().unique().tolist()) if "推薦課程ID" in data.columns else []
    selected_courses = f1.multiselect("課程", course_opts, default=course_opts)
    threshold = f2.slider("預測機率門檻", 0.0, 1.0, 0.0, 0.01)
    selected_priority = f3.multiselect("優先級", ["高", "中", "低"], default=["高", "中", "低"])
    selected_segments = f4.multiselect("客群", SEGMENTS["客群"].tolist(), default=SEGMENTS["客群"].tolist())

    filtered = data.copy()
    filtered = filtered[filtered["預測報名機率"] >= threshold]
    if selected_courses:
        filtered = filtered[filtered["推薦課程ID"].isin(selected_courses)]
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

    cols = [c for c in ["學員ID", "推薦課程ID", "預測報名機率", "優先級", "客群", "推薦原因", "建議通路", "建議文案"] if c in filtered.columns]
    st.dataframe(filtered[cols], use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        course_counts = filtered["推薦課程ID"].value_counts().rename_axis("推薦課程ID").reset_index(name="推薦次數")
        fig = px.bar(course_counts, x="推薦課程ID", y="推薦次數", title="各課程被推薦次數")
        st.plotly_chart(fig, use_container_width=True)
        explain_box("柱子越高，代表越多會員被推薦該課程。", "可快速找出本週主推課程。", "推薦次數最高的課程適合做 EDM 主推。")
    with right:
        fig = px.histogram(filtered, x="預測報名機率", nbins=20, title="推薦機率分布")
        st.plotly_chart(fig, use_container_width=True)
        explain_box("越靠右代表會員越可能報名。", "高機率名單適合優先聯絡。", "可把門檻調到 0.5 或 0.7 做分層投放。")

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載目前篩選後名單 CSV", csv, "filtered_marketing_leads.csv", "text/csv")


# -----------------------------
# 3. 單一課程作戰室
# -----------------------------
elif page == "單一課程作戰室":
    st.title("單一課程作戰室")
    st.caption("選一門課，立刻看目標會員、主打賣點與建議通路。")

    if top3_exec.empty:
        st.warning("找不到 Top3 推薦名單。")
        st.stop()

    course_list = sorted(top3_exec["推薦課程ID"].dropna().unique().tolist())
    course = st.selectbox("選擇課程", course_list)
    d = top3_exec[top3_exec["推薦課程ID"] == course].copy()
    high = d[d["優先級"] == "高"]
    target = high if not high.empty else d

    main_segment = target["客群"].mode().iloc[0] if not target.empty else "N/A"
    avg_prob = target["預測報名機率"].mean() if not target.empty else np.nan
    main_channel = target["建議通路"].mode().iloc[0] if not target.empty else "N/A"
    main_copy = target["建議文案"].mode().iloc[0] if not target.empty else "N/A"
    segment_row = SEGMENTS[SEGMENTS["客群"] == main_segment]
    main_interest = segment_row["主要興趣"].iloc[0] if not segment_row.empty else "N/A"
    main_appeal = segment_row["建議賣點"].iloc[0] if not segment_row.empty else "N/A"

    a, b, c, e = st.columns(4)
    a.metric("高潛力學員數", len(high))
    b.metric("平均預測機率", f"{avg_prob:.2f}" if not pd.isna(avg_prob) else "N/A")
    c.metric("主要客群", main_segment)
    e.metric("建議通路", main_channel)

    st.markdown("### 課程行銷策略摘要")
    st.markdown(
        f"""
        <div class="action-card">
        這門課目前最適合先投放給 <b>{main_segment}</b>。主要興趣是 <b>{main_interest}</b>，
        建議賣點是 <b>{main_appeal}</b>。通路可先使用 <b>{main_channel}</b>，
        文案方向可用：「{main_copy}」
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 該課程高潛力學員表格")
    show_cols = [c for c in ["學員ID", "推薦課程ID", "預測報名機率", "優先級", "客群", "推薦原因", "建議通路", "建議文案"] if c in target.columns]
    st.dataframe(target.sort_values("預測報名機率", ascending=False)[show_cols], use_container_width=True, hide_index=True)

    fig = px.histogram(d, x="客群", color="優先級", title="該課程推薦客群分布")
    st.plotly_chart(fig, use_container_width=True)
    explain_box("看這門課主要被推薦給哪些客群。", "可判斷這門課要用哪一種行銷主軸。", "最大客群先做主文案，其餘客群做分眾素材。")


# -----------------------------
# 4. 客群洞察與市場區隔
# -----------------------------
elif page == "客群洞察與市場區隔":
    st.title("客群洞察與市場區隔")
    st.caption("把 Cluster 轉成企業能使用的客群名稱與文案方向。")

    st.markdown("### 三個可執行客群")
    st.dataframe(SEGMENTS, use_container_width=True, hide_index=True)

    if cluster_df is not None:
        raw = cluster_df.copy()
        if "cluster" in raw.columns:
            raw["行銷命名"] = raw["cluster"].map({0: "商業語文／運務導向型", 1: "多元進修／一般潛力型"}).fillna("其他客群")
        st.markdown("### 原始分群結果")
        st.dataframe(raw, use_container_width=True, hide_index=True)

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
    st.dataframe(matrix, use_container_width=True, hide_index=True)

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
        st.dataframe(display[[c for c in ["檢定項目", "p-value", "是否顯著"] if c in display.columns]], use_container_width=True, hide_index=True)
        explain_box("p-value 低於 0.05 代表該特徵能區分客群。", "主要興趣與國貿實務興趣較有區分力。", "分眾行銷先看興趣，再看職稱。")


# -----------------------------
# 5. AI 文案與素材產生器
# -----------------------------
elif page == "AI 文案與素材產生器":
    st.title("AI 文案與素材產生器")
    st.caption("不用串 API，先用規則式模板產生可交給行銷人員修改的初稿。")

    c1, c2, c3 = st.columns(3)
    courses = sorted(top3_exec["推薦課程ID"].dropna().unique().tolist()) if not top3_exec.empty else ["課程_2"]
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


# -----------------------------
# 6. 成效追蹤與 A/B Test
# -----------------------------
elif page == "成效追蹤與 A/B Test":
    st.title("成效追蹤與 A/B Test")
    st.caption("上傳投放結果，追蹤不同客群、通路和文案方向的實際表現。")

    sample = pd.DataFrame(
        [
            ["C001", 2, "證照／多元進修型", "LINE", 100, 78, 31, 8],
            ["C002", 2, "國貿實務導向型", "EDM", 120, 72, 25, 6],
        ],
        columns=["campaign_id", "課程ID", "客群", "通路", "發送人數", "開信數", "點擊數", "報名數"],
    )
    upload = st.file_uploader("上傳 campaign_result.csv", type=["csv"])
    if upload is None:
        st.info("尚未上傳成效資料。請使用下方格式建立 campaign_result.csv。")
        result = sample.copy()
        st.dataframe(sample, use_container_width=True, hide_index=True)
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
    st.dataframe(result, use_container_width=True, hide_index=True)
    fig = px.bar(result, x="campaign_id", y=["開信率", "點擊率", "報名率", "轉換率"], barmode="group", title="不同活動成效比較")
    st.plotly_chart(fig, use_container_width=True)
    explain_box("比較不同 campaign 的開信、點擊、報名與轉換。", "可找出哪個通路或文案更有效。", "未來可把結果回填，讓推薦模型和分眾規則更準。")


# -----------------------------
# 7. 模型解釋與安全使用
# -----------------------------
elif page == "模型解釋與安全使用":
    st.title("模型解釋與安全使用")
    st.caption("保留模型圖表，但把技術結果翻成行銷可以使用的語言。")

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
    st.dataframe(marketing_points(), use_container_width=True, hide_index=True)

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
