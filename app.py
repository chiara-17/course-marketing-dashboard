#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
套件安裝：
pip install streamlit pandas numpy matplotlib plotly pillow openpyxl seaborn altair

啟動：
streamlit run app.py
"""

from __future__ import annotations

import io
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

# 優先使用 app.py 同層的 outputs，避免從其他目錄啟動時找不到檔案
APP_DIR = Path(__file__).resolve().parent
outputs_path = APP_DIR / "outputs"

# 若同層 outputs 不存在，才 fallback 到目前工作目錄 outputs
if not outputs_path.exists():
    outputs_path = Path.cwd() / "outputs"


# -------------------------
# Helper Functions
# -------------------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        st.warning(f"尚未找到此檔案：{path.name}，請確認 outputs 資料夾是否已產生分析結果。")
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    st.warning(f"檔案讀取失敗：{path.name}")
    return None


def safe_read_txt(path: Path) -> Optional[str]:
    if not path.exists():
        st.warning(f"尚未找到此檔案：{path.name}，請確認 outputs 資料夾是否已產生分析結果。")
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    st.warning(f"文字檔讀取失敗：{path.name}")
    return None


def safe_load_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        st.warning(f"尚未找到此圖檔：{path.name}，請確認 outputs 資料夾是否已產生分析結果。")
        return None
    try:
        return Image.open(path)
    except Exception:
        st.warning(f"圖檔載入失敗：{path.name}")
        return None


def find_file(candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = outputs_path / name
        if p.exists():
            return p
    return None


def parse_model_report(report_text: str) -> dict:
    out = {
        "xgb_1a": {},
        "xgb_1b": {},
        "cm_1a": None,
        "cm_1b": None,
    }
    if not report_text:
        return out

    block_1a = re.search(r"\[XGBoost_1A.*?\](.*?)(?:\n\[XGBoost_1B|\Z)", report_text, re.S)
    block_1b = re.search(r"\[XGBoost_1B.*?\](.*?)(?:\n\[MNL|\Z)", report_text, re.S)
    key_map = {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1",
        "ROC_AUC": "ROC-AUC",
        "PR_AUC": "PR-AUC",
    }

    def _extract_metrics(block_text: str) -> dict:
        d = {}
        for raw_k, show_k in key_map.items():
            m = re.search(rf"{re.escape(raw_k)}\s*:\s*([0-9.]+)", block_text)
            if m:
                d[show_k] = float(m.group(1))
        return d

    def _extract_cm(block_text: str):
        m = re.search(r"Confusion Matrix:\s*\[\[([0-9\s]+)\]\s*\[([0-9\s]+)\]\]", block_text, re.S)
        if not m:
            return None
        row1 = [int(x) for x in m.group(1).split()]
        row2 = [int(x) for x in m.group(2).split()]
        return np.array([row1, row2])

    if block_1a:
        t = block_1a.group(1)
        out["xgb_1a"] = _extract_metrics(t)
        out["cm_1a"] = _extract_cm(t)
    if block_1b:
        t = block_1b.group(1)
        out["xgb_1b"] = _extract_metrics(t)
        out["cm_1b"] = _extract_cm(t)
    return out


def add_priority(df: pd.DataFrame, prob_col: str = "預測報名機率") -> pd.DataFrame:
    if prob_col not in df.columns:
        return df
    d = df.copy()
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce")
    d["推薦優先級"] = pd.cut(
        d[prob_col],
        bins=[-0.001, 0.5, 0.7, 1.0],
        labels=["低", "中", "高"],
    )
    return d


def section_header(title: str, desc: str = ""):
    st.markdown(f"## {title}")
    if desc:
        st.caption(desc)


# -------------------------
# Load Data
# -------------------------
model_report_path = outputs_path / "topic1_model_performance_report.txt"
coef_path = find_file(["topic1_logistic_coefficients.csv", "topic1_logistic_or_mnl_coefficients.csv"])
xgb_imp_path = outputs_path / "topic1_xgboost_feature_importance.csv"
shap_imp_path = outputs_path / "topic1_shap_importance.csv"
top1_path = outputs_path / "topic1_recommendation_top1.csv"
top3_path = outputs_path / "topic1_recommendation_top3.csv"
cluster_profile_path = outputs_path / "topic2_cluster_profile.csv"
chi_path = outputs_path / "topic2_chi_square_results.csv"
final_report_path = outputs_path / "final_analysis_report.md"

model_report_text = safe_read_txt(model_report_path)
coef_df = safe_read_csv(coef_path) if coef_path else None
xgb_imp_df = safe_read_csv(xgb_imp_path)
shap_imp_df = safe_read_csv(shap_imp_path)
top1_df = safe_read_csv(top1_path)
top3_df = safe_read_csv(top3_path)
cluster_df = safe_read_csv(cluster_profile_path)
chi_df = safe_read_csv(chi_path)
final_report_text = safe_read_txt(final_report_path)
model_metrics = parse_model_report(model_report_text or "")


# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("📊 導覽選單")
page = st.sidebar.radio(
    "請選擇頁面",
    [
        "總覽",
        "報名因素分析",
        "課程推薦名單",
        "市場區隔分析",
        "行銷策略產生器",
        "模型表現與研究限制",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"資料來源：{outputs_path.resolve()}")


# -------------------------
# Page 1: 總覽
# -------------------------
if page == "總覽":
    section_header("總覽", "課程精準行銷與會員推薦儀表板")
    st.info("本儀表板整合報名機率預測、SHAP 重要因素、會員課程推薦與市場區隔分析，可協助內部人員進行精準行銷、課程規劃與廣告文案發想。")

    # KPI 推算
    learner_count = None
    if top1_df is not None and "學員ID" in top1_df.columns:
        learner_count = int(top1_df["學員ID"].nunique())
    elif top3_df is not None and "學員ID" in top3_df.columns:
        learner_count = int(top3_df["學員ID"].nunique())

    course_count = None
    if top3_df is not None and "推薦課程ID" in top3_df.columns:
        course_count = int(top3_df["推薦課程ID"].nunique())
    elif top1_df is not None and "推薦課程ID" in top1_df.columns:
        course_count = int(top1_df["推薦課程ID"].nunique())

    rec_count = int(len(top3_df)) if top3_df is not None else (int(len(top1_df)) if top1_df is not None else None)
    top1_count = int(len(top1_df)) if top1_df is not None else None
    top3_count = int(len(top3_df)) if top3_df is not None else None
    roc_auc = model_metrics.get("xgb_1b", {}).get("ROC-AUC", model_metrics.get("xgb_1a", {}).get("ROC-AUC"))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("學員數", learner_count if learner_count is not None else "N/A")
    c2.metric("課程數", course_count if course_count is not None else "N/A")
    c3.metric("推薦名單數", rec_count if rec_count is not None else "N/A")
    c4.metric("模型 ROC-AUC", f"{roc_auc:.3f}" if isinstance(roc_auc, float) else "N/A")
    c5.metric("Top1 推薦人數", top1_count if top1_count is not None else "N/A")
    c6.metric("Top3 推薦筆數", top3_count if top3_count is not None else "N/A")

    st.markdown("### 主要研究發現")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.success("證照屬性是重要報名因素")
    f2.success("價格區間會影響報名機率")
    f3.success("興趣特徵比性別或職稱更能區分客群")
    f4.success("XGBoost 可用於會員課程推薦")
    f5.success("分群結果可支援行銷文案與課程包設計")

    with st.expander("查看 final_analysis_report.md 摘要"):
        if final_report_text:
            st.text(final_report_text[:4000])


# -------------------------
# Page 2: 報名因素分析
# -------------------------
elif page == "報名因素分析":
    section_header("報名因素分析", "回答：哪些因素會影響學員報名？")

    left, right = st.columns(2)

    images = [
        ("shap_bar_attribute_model.png", "SHAP Bar Plot（屬性模型）"),
        ("shap_summary_attribute_model.png", "SHAP Summary Plot（屬性模型）"),
        ("xgboost_feature_importance_attribute_model.png", "XGBoost Feature Importance（屬性模型）"),
        ("shap_bar_course_dummy_model.png", "SHAP Bar Plot（課程 Dummy 模型）"),
        ("shap_summary_course_dummy_model.png", "SHAP Summary Plot（課程 Dummy 模型）"),
        ("xgboost_feature_importance_course_dummy_model.png", "XGBoost Feature Importance（課程 Dummy 模型）"),
    ]

    for i, (fname, title) in enumerate(images):
        img = safe_load_image(outputs_path / fname)
        col = left if i % 2 == 0 else right
        with col:
            st.markdown(f"### {title}")
            if img is not None:
                st.image(img, use_container_width=True)

    st.info("SHAP bar plot：代表變數平均影響力大小，越上方越重要。")
    st.info("SHAP summary plot：右邊代表推高報名機率，左邊代表降低報名機率；紅色代表該變數值高，藍色代表該變數值低。")
    st.info("Feature importance：代表 XGBoost 在預測時常使用哪些變數。")

    st.markdown("### SHAP Top 20 重要變數")
    if shap_imp_df is not None:
        d = shap_imp_df.copy()
        val_col = "mean_abs_shap" if "mean_abs_shap" in d.columns else d.columns[1]
        d = d.sort_values(val_col, ascending=False).head(20)
        st.dataframe(d, use_container_width=True)

    st.markdown("### 行銷解釋表")
    marketing_map = pd.DataFrame(
        [
            ["證照", "學員重視考照、專業認證與職涯加值，廣告文案可強調證照價值與職涯競爭力。"],
            ["3000-4000", "此價格帶可能較符合學員可接受的進修預算，可主打高 CP 值與短期職涯投資。"],
            ["線上", "若線上課程影響偏低，行銷應補強互動性、回放、教材與彈性學習優勢。"],
            ["興趣_國貿實務", "對國貿實務有興趣者可推實務案例型、進出口流程、職場應用課程。"],
            ["興趣_專業證照", "可推考照班、證照輔導班、職涯升級課程。"],
        ],
        columns=["變數", "行銷解釋"],
    )
    st.dataframe(marketing_map, use_container_width=True)


# -------------------------
# Page 3: 課程推薦名單
# -------------------------
elif page == "課程推薦名單":
    section_header("課程推薦名單", "最重要的落地應用頁面")

    source_choice = st.radio("推薦類型", ["Top1", "Top3"], horizontal=True)
    base_df = top1_df if source_choice == "Top1" else top3_df
    if base_df is None:
        st.warning("推薦名單檔案不存在，請先確認 outputs 是否有 topic1_recommendation_top1.csv / top3.csv")
        st.stop()

    df = add_priority(base_df)
    if "預測報名機率" in df.columns:
        df["預測報名機率"] = pd.to_numeric(df["預測報名機率"], errors="coerce")

    c1, c2, c3 = st.columns([1, 1, 1])
    threshold = c1.slider("預測機率門檻", 0.0, 1.0, 0.0, 0.01)
    course_opts = sorted(df["推薦課程ID"].dropna().unique().tolist()) if "推薦課程ID" in df.columns else []
    selected_courses = c2.multiselect("推薦課程ID 篩選", options=course_opts, default=course_opts)
    selected_priority = c3.multiselect("推薦優先級篩選", options=["高", "中", "低"], default=["高", "中", "低"])

    filtered = df.copy()
    if "預測報名機率" in filtered.columns:
        filtered = filtered[filtered["預測報名機率"] >= threshold]
    if "推薦課程ID" in filtered.columns and selected_courses:
        filtered = filtered[filtered["推薦課程ID"].isin(selected_courses)]
    if "推薦優先級" in filtered.columns and selected_priority:
        filtered = filtered[filtered["推薦優先級"].isin(selected_priority)]

    k1, k2, k3 = st.columns(3)
    high_n = int((filtered.get("推薦優先級") == "高").sum()) if "推薦優先級" in filtered.columns else 0
    mid_n = int((filtered.get("推薦優先級") == "中").sum()) if "推薦優先級" in filtered.columns else 0
    low_n = int((filtered.get("推薦優先級") == "低").sum()) if "推薦優先級" in filtered.columns else 0
    k1.metric("高優先級人數", high_n)
    k2.metric("中優先級人數", mid_n)
    k3.metric("低優先級人數", low_n)

    st.markdown("### 推薦名單表格")
    st.dataframe(filtered, use_container_width=True)

    if "推薦課程ID" in filtered.columns:
        cnt = filtered["推薦課程ID"].value_counts().reset_index()
        cnt.columns = ["推薦課程ID", "推薦次數"]
        fig = px.bar(cnt, x="推薦課程ID", y="推薦次數", title="每門課被推薦次數")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 高潛力名單（推薦優先級=高）")
    high_df = filtered[filtered["推薦優先級"] == "高"] if "推薦優先級" in filtered.columns else pd.DataFrame()
    st.dataframe(high_df, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載目前篩選後推薦名單 CSV",
        data=csv_bytes,
        file_name=f"filtered_recommendation_{source_choice}.csv",
        mime="text/csv",
    )

    st.info("此推薦名單可用於 EDM、LINE 推播、電話行銷與廣告再行銷。由於模型 precision 中等，建議先用模型篩選高潛力名單，再搭配商業規則進行二次篩選。")


# -------------------------
# Page 4: 市場區隔分析
# -------------------------
elif page == "市場區隔分析":
    section_header("市場區隔分析", "回答：學員可以分成哪些市場區隔？每群適合什麼行銷策略？")
    if cluster_df is None:
        st.warning("尚未找到 topic2_cluster_profile.csv")
        st.stop()

    d = cluster_df.copy()
    if "cluster" in d.columns:
        d["群別名稱"] = d["cluster"].map({
            0: "商業語文／運務導向型",
            1: "多元進修／一般潛力型",
        }).fillna(d.get("segment_name", "未命名群"))
    else:
        d["群別名稱"] = d.get("segment_name", "未命名群")

    st.markdown("### 分群輪廓表")
    st.dataframe(d, use_container_width=True)

    if "群體人數" in d.columns:
        fig1 = px.bar(d, x="群別名稱", y="群體人數", title="各群人數")
        st.plotly_chart(fig1, use_container_width=True)
    if "群體比例" in d.columns:
        fig2 = px.pie(d, names="群別名稱", values="群體比例", title="各群比例")
        st.plotly_chart(fig2, use_container_width=True)

    for fname, caption in [
        ("cluster_profile_heatmap.png", "群輪廓熱圖"),
        ("kmeans_elbow_plot.png", "K-means Elbow Plot"),
        ("kmeans_silhouette_plot.png", "K-means Silhouette Plot"),
    ]:
        img = safe_load_image(outputs_path / fname)
        if img is not None:
            st.markdown(f"### {caption}")
            st.image(img, use_container_width=True)

    st.info("Elbow plot：用來觀察分群數增加後，群內誤差是否明顯下降。")
    st.info("Silhouette score：越高代表分群越清楚。本研究 K=2 較合適，但分群邊界仍需保守解讀。")

    st.markdown("### 卡方檢定結果")
    if chi_df is not None:
        show = chi_df.copy()
        if "significant(p<0.05)" in show.columns:
            show["是否顯著"] = show["significant(p<0.05)"].map({1: "是", 0: "否"})
        if "variable" in show.columns:
            show = show.rename(columns={"variable": "檢定項目", "p_value": "p-value"})
        show["解釋"] = show.apply(
            lambda r: "與分群顯著關聯" if str(r.get("是否顯著", "")) == "是" else "與分群關聯不顯著",
            axis=1,
        )
        keep = [c for c in ["檢定項目", "p-value", "是否顯著", "解釋"] if c in show.columns]
        st.dataframe(show[keep], use_container_width=True)

    st.success("cluster × 主要興趣：顯著")
    st.success("cluster × 興趣_國貿實務：顯著")
    st.warning("cluster × 性別：不顯著")
    st.warning("cluster × 職稱：不顯著")

    st.info("分群結果顯示，市場差異主要不是由性別或職稱造成，而是由興趣結構造成，尤其國貿實務興趣具有明顯區辨力。因此行銷分眾應以興趣為主，而非只依賴人口統計變數。")


# -------------------------
# Page 5: 行銷策略產生器
# -------------------------
elif page == "行銷策略產生器":
    section_header("行銷策略產生器", "根據模型洞察快速產生文案初稿（規則式）")

    col1, col2, col3 = st.columns(3)
    target_course = col1.text_input("目標課程", value="國貿實務應用班")
    audience = col2.selectbox(
        "目標客群",
        ["證照導向型", "國貿實務導向型", "商業語文／運務導向型", "多元進修／一般潛力型", "線上學習型"],
    )
    tone = col3.selectbox("文案語氣", ["專業", "親切", "急迫", "職涯導向", "年輕活潑"])

    col4, col5 = st.columns(2)
    selling_points = col4.multiselect(
        "主打賣點",
        ["證照", "職涯加值", "國貿實務", "商業語文", "高 CP 值", "線上彈性", "實務案例"],
        default=["證照", "職涯加值"],
    )
    platform = col5.selectbox("投放平台", ["Facebook", "Instagram", "LINE", "EDM", "YouTube Shorts"])

    msg_parts = []
    if "證照" in selling_points:
        msg_parts.append("強調考照、專業認證、履歷加分與職涯競爭力")
    if "國貿實務" in selling_points:
        msg_parts.append("聚焦進出口流程、實務案例、職場即戰力與貿易文件報關應用")
    if "商業語文" in selling_points:
        msg_parts.append("凸顯國際溝通、商務情境、客戶應對與跨國工作能力")
    if "線上彈性" in selling_points:
        msg_parts.append("主打可回放、不受地點限制、下班後學習與彈性進修")
    if "高 CP 值" in selling_points:
        msg_parts.append("訴求小額投資、高價值回報、短期進修與職涯升級")
    if "實務案例" in selling_points:
        msg_parts.append("以真實案例帶動學習動機，提升可落地應用")
    if "職涯加值" in selling_points:
        msg_parts.append("連結升遷、轉職、加薪等職涯成果")
    core_message = "；".join(msg_parts) if msg_parts else "強調課程價值與實務應用"

    title = f"【{target_course}】{audience}必看：現在開始打造你的職場競爭力"
    social = (
        f"想讓進修真正帶來改變？\n"
        f"這門《{target_course}》專為「{audience}」設計，{core_message}。\n"
        f"平台：{platform}｜語氣：{tone}\n"
        f"立即了解課程內容，為下一次職涯機會做好準備。"
    )
    line_push = f"📌 {target_course} 熱門開課中！\n{core_message}。\n回覆「我要報名」立即索取課程資訊。"
    edm_subject = f"【{target_course}】給{audience}的進修方案：{selling_points[0] if selling_points else '職涯升級'}現在開始"
    short_script = (
        "0-3秒：痛點開場（升遷卡關/技能不足）\n"
        f"3-8秒：介紹《{target_course}》與客群「{audience}」\n"
        f"8-12秒：亮點訴求：{core_message}\n"
        "12-15秒：CTA（立即報名/私訊了解）"
    )
    prompt = (
        f"請生成一張 {platform} 廣告視覺，主題為「{target_course}」，客群「{audience}」，"
        f"風格「{tone}」，重點包含：{', '.join(selling_points) if selling_points else '職涯加值'}，"
        "畫面需有課程標題、報名按鈕與專業學習場景。"
    )

    st.markdown("### 產生結果")
    st.text_area("1. 廣告標題", title, height=80)
    st.text_area("2. 社群貼文文案", social, height=160)
    st.text_area("3. LINE 推播文案", line_push, height=120)
    st.text_area("4. EDM 標題", edm_subject, height=80)
    st.text_area("5. 15 秒短影片腳本", short_script, height=150)
    st.text_area("6. AI 圖像 / 影片生成 Prompt", prompt, height=120)

    st.warning("此文案為根據模型結果與市場區隔規則產生的初稿，實際投放前建議再由行銷人員依品牌語氣調整，並透過 A/B Test 驗證成效。")


# -------------------------
# Page 6: 模型表現與研究限制
# -------------------------
elif page == "模型表現與研究限制":
    section_header("模型表現與研究限制")

    if model_report_text:
        st.markdown("### 模型摘要")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### XGBoost 1A（屬性模型）")
            for k, v in model_metrics.get("xgb_1a", {}).items():
                st.metric(k, f"{v:.4f}")
            cm1 = model_metrics.get("cm_1a")
            if cm1 is not None:
                st.write("Confusion Matrix")
                st.dataframe(pd.DataFrame(cm1, index=["實際0", "實際1"], columns=["預測0", "預測1"]))
        with c2:
            st.markdown("#### XGBoost 1B（課程Dummy模型）")
            for k, v in model_metrics.get("xgb_1b", {}).items():
                st.metric(k, f"{v:.4f}")
            cm2 = model_metrics.get("cm_1b")
            if cm2 is not None:
                st.write("Confusion Matrix")
                st.dataframe(pd.DataFrame(cm2, index=["實際0", "實際1"], columns=["預測0", "預測1"]))

        with st.expander("查看原始模型報告文字"):
            st.text(model_report_text)

    st.markdown("### 指標白話解釋")
    st.info("Accuracy：整體預測正確率")
    st.info("Precision：模型說會報名的人裡面，真的會報名的比例")
    st.info("Recall：真正會報名的人裡面，模型抓到多少")
    st.info("F1：Precision 和 Recall 的平衡")
    st.info("ROC-AUC：模型排序能力")
    st.info("PR-AUC：不平衡資料下的重要評估指標")
    st.success("本研究目標是找出高潛力學員，因此 Recall 與排序能力比單看 Accuracy 更重要。")

    st.markdown("### 研究限制")
    limitations = [
        "本資料來自實際報名名單，未必能完整觀察學員真實選擇集合。",
        "MNL 部分使用近似 Logit 模型，結果應以趨勢解讀，不宜過度解釋 p-value。",
        "樣本數較小，且 Y=1 與 Y=0 類別不平衡。",
        "部分課程屬性與課程本身高度重疊，例如證照課可能同時代表某特定課程。",
        "推薦結果應搭配商業規則二次篩選，不宜完全自動化決策。",
    ]
    for i, t in enumerate(limitations, start=1):
        st.write(f"{i}. {t}")
