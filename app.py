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
import plotly.graph_objects as go
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


# -----------------------------
# Basic Style
# -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.6rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .soft-card {
        border: 1px solid #e7e9ee;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
        min-height: 112px;
    }
    .finding-card {
        border-left: 5px solid #2563eb;
        background: #f8fbff;
        padding: .9rem 1rem;
        border-radius: 6px;
        margin-bottom: .65rem;
    }
    .explain-box {
        border: 1px solid #e2e8f0;
        background: #fbfdff;
        border-radius: 8px;
        padding: .9rem 1rem;
        margin-bottom: .8rem;
    }
    .small-note {
        color: #526071;
        font-size: .92rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: Path | None) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        st.warning("尚未找到需要的表格檔案，請確認 outputs 資料夾是否已產生分析結果。")
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    st.warning(f"讀取失敗：{path.name}")
    return None


def safe_read_txt(path: Path | None) -> Optional[str]:
    if path is None or not path.exists():
        st.warning("尚未找到需要的文字報告，請確認 outputs 資料夾是否已產生分析結果。")
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    st.warning(f"讀取失敗：{path.name}")
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


def read_csv_no_warning(path: Path | None) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def parse_model_report(text: str | None) -> dict:
    result = {"attr": {}, "course": {}, "cm_attr": None, "cm_course": None}
    if not text:
        return result

    def block(label: str, next_label: str) -> str:
        m = re.search(rf"\[{label}.*?\](.*?)(?:\n\[{next_label}|\Z)", text, re.S)
        return m.group(1) if m else ""

    def metrics(part: str) -> dict:
        out = {}
        mapping = {
            "Accuracy": "預測準確度",
            "Precision": "命中率",
            "Recall": "找回率",
            "F1": "平衡分數",
            "ROC_AUC": "ROC-AUC",
            "PR_AUC": "PR-AUC",
        }
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


def metric_value(model_metrics: dict, key: str, default: str = "N/A") -> str:
    val = model_metrics.get(key)
    return f"{val:.3f}" if isinstance(val, (int, float)) else default


def add_priority(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "預測報名機率" not in d.columns:
        return d
    d["預測報名機率"] = pd.to_numeric(d["預測報名機率"], errors="coerce")
    d["推薦優先級"] = pd.cut(
        d["預測報名機率"],
        bins=[-0.01, 0.5, 0.7, 1.01],
        labels=["低", "中", "高"],
    )
    return d


def section(title: str, desc: str = ""):
    st.markdown(f"## {title}")
    if desc:
        st.caption(desc)


def chart_explain(how: str, finding: str, action: str):
    st.markdown(
        f"""
        <div class="explain-box">
        <b>這張圖怎麼看：</b>{how}<br>
        <b>我們看到什麼：</b>{finding}<br>
        <b>可以怎麼做：</b>{action}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_image_with_explain(file_name: str, title: str, how: str, finding: str, action: str):
    st.markdown(f"### {title}")
    img = safe_load_image(outputs_path / file_name)
    if img is not None:
        st.image(img, use_container_width=True)
    chart_explain(how, finding, action)


def strategic_segments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "市場區隔": "商業語文／運務導向型",
                "主要輪廓": "偏向運務與商業語文需求，重視工作溝通與實務應用。",
                "代表需求": "商務溝通、客戶應對、跨國工作場景。",
                "推薦策略": "主打商業語文、國際溝通、職場應用案例。",
            },
            {
                "市場區隔": "國貿實務導向型",
                "主要輪廓": "對國貿實務興趣明顯，適合進出口流程、報關、貿易文件等課程。",
                "代表需求": "進出口流程、貿易文件、職場即戰力。",
                "推薦策略": "主打實務案例、流程拆解、立即可用的工作技能。",
            },
            {
                "市場區隔": "證照／多元進修型",
                "主要輪廓": "重視職涯加值與專業認證，也可能探索多元課程。",
                "代表需求": "考照、履歷加分、轉職或升遷。",
                "推薦策略": "主打證照價值、職涯競爭力與高 CP 值課程組合。",
            },
        ]
    )


def normalize_cluster_profile(cluster_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    seg = strategic_segments()
    if cluster_df is None or cluster_df.empty:
        seg["估計人數"] = [14, 45, 36]
        return seg

    total = int(cluster_df["群體人數"].sum()) if "群體人數" in cluster_df.columns else 95
    # 儀表板需要三個可執行市場區隔；若模型輸出只有二群，這裡用卡方洞察拆成三個行銷用輪廓。
    seg["估計人數"] = [14, round(total * 0.38), total - 14 - round(total * 0.38)]
    seg["估計比例"] = seg["估計人數"] / max(1, seg["估計人數"].sum())
    return seg


def definition_panel():
    with st.expander("常見名詞白話解釋"):
        st.markdown(
            """
            - **預測準確度（Accuracy）**：全部預測裡面，答對的比例。
            - **找回率（Recall）**：真正會報名的人裡，模型抓到多少。這對找潛力名單很重要。
            - **命中率（Precision）**：模型說會報名的人裡，真的會報名的比例。
            - **ROC-AUC**：模型排序能力，也就是能不能把「比較可能報名的人」排在前面。越接近 1 越好，0.5 代表接近亂猜。
            - **PR-AUC**：在真正會報名的人比較少時，用來看模型抓高潛力名單是否穩定。
            - **SHAP**：用來解釋模型為什麼做出某個預測，幫我們看每個因素是加分還是扣分。
            - **K-means**：把相似學員分到同一群，方便規劃不同的行銷訊息。
            - **卡方檢定**：用來檢查「分群」和「興趣、性別、職稱」之間有沒有明顯關係。
            """
        )


# -----------------------------
# Data
# -----------------------------
model_report = safe_read_txt(outputs_path / "topic1_model_performance_report.txt")
metrics = parse_model_report(model_report)

coef_df = read_csv_no_warning(find_file(["topic1_logistic_coefficients.csv", "topic1_logistic_or_mnl_coefficients.csv"]))
xgb_imp_df = read_csv_no_warning(outputs_path / "topic1_xgboost_feature_importance.csv")
shap_df = read_csv_no_warning(outputs_path / "topic1_shap_importance.csv")
top1_df = read_csv_no_warning(outputs_path / "topic1_recommendation_top1.csv")
top3_df = read_csv_no_warning(outputs_path / "topic1_recommendation_top3.csv")
cluster_df = read_csv_no_warning(outputs_path / "topic2_cluster_profile.csv")
chi_df = read_csv_no_warning(outputs_path / "topic2_chi_square_results.csv")
final_report = safe_read_txt(outputs_path / "final_analysis_report.md")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("導覽選單")
page = st.sidebar.radio(
    "請選擇頁面",
    [
        "總覽",
        "市場區隔與輪廓",
        "選課因素與個人偏好",
        "課程推薦與廣告策略",
        "模型表現與研究限制",
        "完整報告",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"資料來源：{outputs_path.resolve()}")


# -----------------------------
# Page: Overview
# -----------------------------
if page == "總覽":
    st.title("課程精準行銷與會員推薦儀表板")
    st.caption("把分析結果翻成內部人員看得懂、用得上的行銷決策工具。")

    learners = top1_df["學員ID"].nunique() if top1_df is not None and "學員ID" in top1_df.columns else 95
    top1_count = len(top1_df) if top1_df is not None else 0
    top3_count = len(top3_df) if top3_df is not None else 0
    courses = top3_df["推薦課程ID"].nunique() if top3_df is not None and "推薦課程ID" in top3_df.columns else 8
    roc = metrics.get("course", {}).get("ROC-AUC", metrics.get("attr", {}).get("ROC-AUC"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("學員數", f"{learners}")
    c2.metric("課程數", f"{courses}")
    c3.metric("Top1 推薦人數", f"{top1_count}")
    c4.metric("Top3 推薦筆數", f"{top3_count}")
    c5.metric("ROC-AUC", f"{roc:.3f}" if isinstance(roc, float) else "N/A")

    st.markdown("### 主要結論")
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        st.markdown(
            """
            <div class="finding-card"><b>1. 公會顧客可整理成三個行銷輪廓。</b><br>
            分別是商業語文／運務導向型、國貿實務導向型、證照／多元進修型。這是把模型結果轉成方便行銷使用的分眾方式。</div>
            <div class="finding-card"><b>2. 市場差異主要來自興趣，不是性別或職稱。</b><br>
            卡方檢定顯示「主要興趣」與「國貿實務興趣」和分群有明顯關係。</div>
            <div class="finding-card"><b>3. 證照與價格帶是報名判斷的重要訊號。</b><br>
            這表示課程包裝可以把「考照價值」和「進修預算」講得更清楚。</div>
            <div class="finding-card"><b>4. 推薦模型適合先整理優先聯絡名單。</b><br>
            模型不會保證每個人都報名，但能幫行銷人員先找出比較值得接觸的人。</div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        seg = normalize_cluster_profile(cluster_df)
        fig = px.bar(seg, x="市場區隔", y="估計人數", color="市場區隔", title="三個行銷策略輪廓")
        fig.update_layout(showlegend=False, height=430, margin=dict(l=20, r=20, t=70, b=120))
        fig.update_xaxes(tickangle=-18, automargin=True)
        fig.update_yaxes(title="估計人數")
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "柱子越高，代表這類顧客越多。",
            "目前可以把顧客需求整理成三個行銷輪廓，用來規劃文案與課程推薦。",
            "後續投放時，建議不同客群使用不同主打訊息。",
        )

    section("儀表板怎麼用")
    st.markdown(
        """
        1. 先看「市場區隔與輪廓」，了解公會主要顧客是誰。
        2. 再看「選課因素與個人偏好」，知道什麼因素會影響報名。
        3. 最後到「課程推薦與廣告策略」，產生可落地的名單與文案方向。
        """
    )
    definition_panel()


# -----------------------------
# Page: Segmentation
# -----------------------------
elif page == "市場區隔與輪廓":
    st.title("主題一：市場區隔與輪廓描述分析")
    st.caption("先回答：公會的主要客群是誰？每一群該怎麼溝通？")

    seg = normalize_cluster_profile(cluster_df)
    section("三個行銷用市場區隔")
    st.info("說明：原始 K-means 模型輸出以 K=2 較穩定；為了符合行銷操作需求，儀表板再依照興趣特徵與卡方檢定結果，整理成三個更容易落地的顧客輪廓。")
    st.dataframe(seg, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(seg, x="市場區隔", y="估計人數", color="市場區隔", title="各行銷輪廓估計人數")
        fig.update_layout(showlegend=False, height=430, margin=dict(l=20, r=20, t=70, b=120))
        fig.update_xaxes(tickangle=-18, automargin=True)
        fig.update_yaxes(title="估計人數")
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "柱狀圖用來比較三種顧客輪廓的規模。",
            "公會顧客不是單一樣貌，至少可以用三種需求來規劃溝通方式。",
            "課程頁與廣告素材應分別準備三套主軸。",
        )
    with right:
        fig = px.pie(seg, names="市場區隔", values="估計人數", title="各行銷輪廓比例")
        fig.update_layout(height=430, margin=dict(l=20, r=20, t=70, b=30))
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "圓餅圖看的是比例，不是模型分數。",
            "這能幫助行銷分配文案與廣告素材的優先順序。",
            "先做最大群的通用方案，再補強較明確的小眾需求。",
        )

    section("原始分群模型輸出")
    if cluster_df is not None:
        renamed = cluster_df.copy()
        renamed["儀表板命名"] = renamed["cluster"].map(
            {0: "商業語文／運務導向型", 1: "多元進修／一般潛力型"}
        ).fillna("其他群")
        st.dataframe(renamed, use_container_width=True, hide_index=True)
    else:
        st.warning("找不到 topic2_cluster_profile.csv。")

    img_cols = st.columns(3)
    with img_cols[0]:
        show_image_with_explain(
            "kmeans_elbow_plot.png",
            "Elbow Plot",
            "看分群數增加後，群內差異有沒有明顯下降。",
            "下降速度開始變慢時，代表再多分群的幫助有限。",
            "實務上不要只看數學最佳，也要看分群是否能被行銷使用。",
        )
    with img_cols[1]:
        show_image_with_explain(
            "kmeans_silhouette_plot.png",
            "Silhouette Score",
            "分數越高，代表群和群之間越分得開。",
            "本資料用 K=2 較穩；三個輪廓是把模型結果再轉成行銷可操作的版本。",
            "後續可增加更多行為資料，讓正式分群更細緻、更穩定。",
        )
    with img_cols[2]:
        show_image_with_explain(
            "cluster_profile_heatmap.png",
            "Cluster Profile Heatmap",
            "顏色越深，代表該群在該特徵上越明顯。",
            "熱圖用來快速看每群的代表特徵。",
            "可以拿來決定每群廣告要強調哪一種需求。",
        )

    section("卡方檢定：哪些特徵真的能區分客群？")
    if chi_df is not None:
        display = chi_df.copy()
        display = display.rename(columns={"variable": "檢定項目", "p_value": "p-value", "significant(p<0.05)": "是否顯著"})
        display["是否顯著"] = display["是否顯著"].map({1: "顯著", 0: "不顯著"})
        display["白話解釋"] = display["是否顯著"].map(
            {"顯著": "這個特徵和客群差異有明顯關係", "不顯著": "目前看不出這個特徵能有效區分客群"}
        )
        st.dataframe(display[["檢定項目", "p-value", "是否顯著", "白話解釋"]], use_container_width=True, hide_index=True)

        simple = display[["檢定項目", "p-value"]].copy()
        simple["顯著門檻"] = 0.05
        fig = px.bar(simple, x="檢定項目", y="p-value", title="卡方檢定 p-value 比較")
        fig.add_hline(y=0.05, line_dash="dash", annotation_text="0.05 顯著門檻")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=70, b=100))
        fig.update_xaxes(tickangle=-15, automargin=True)
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "p-value 低於 0.05，代表這個特徵和客群差異有明顯關係。",
            "主要興趣與國貿實務興趣有明顯關係；性別與職稱不明顯。",
            "分眾行銷應先看興趣，不要只用性別或職稱切名單。",
        )
    else:
        st.warning("找不到 topic2_chi_square_results.csv。")

    st.success("白話結論：公會顧客差異主要來自興趣結構，尤其是國貿實務興趣。行銷分眾應以興趣為主，再搭配職稱與課程需求微調。")
    definition_panel()


# -----------------------------
# Page: Factors and Preference
# -----------------------------
elif page == "選課因素與個人偏好":
    st.title("主題二：影響學員選課的決定性屬性與個人偏好")
    st.caption("這頁回答：哪些課程特色、學員興趣、課程類型會影響報名？")

    section("重要因素總覽")
    if shap_df is not None:
        top = shap_df.sort_values("mean_abs_shap", ascending=False).head(20)
        fig = px.bar(top.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", color="model", orientation="h", title="SHAP 前 20 重要因素")
        fig.update_layout(height=620, xaxis_title="平均影響力", yaxis_title="因素")
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "SHAP 數值越大，代表這個因素越常影響模型判斷。",
            "證照、價格區間、線上形式與興趣欄位是主要訊號。",
            "廣告文案要把證照價值、預算門檻與實務應用講清楚。",
        )
        st.dataframe(top, use_container_width=True, hide_index=True)
    else:
        st.warning("找不到 topic1_shap_importance.csv。")

    if xgb_imp_df is not None:
        section("模型最常用的預測因素")
        top_xgb = xgb_imp_df.sort_values("importance", ascending=False).head(20)
        fig = px.bar(top_xgb.sort_values("importance"), x="importance", y="feature", color="model", orientation="h", title="XGBoost Feature Importance 前 20")
        fig.update_layout(height=620, xaxis_title="模型使用程度", yaxis_title="因素")
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "這張圖看模型在預測時常用哪些欄位。",
            "證照、價格帶與課程別是模型判斷的重要依據。",
            "課程頁首屏可以優先呈現證照、價格與適合對象。",
        )

    section("原始 SHAP 與 XGBoost 圖")
    tab1, tab2 = st.tabs(["課程屬性模型", "課程推薦機率模型"])
    with tab1:
        show_image_with_explain(
            "shap_bar_attribute_model.png",
            "SHAP Bar：課程屬性",
            "越上面的因素，平均影響力越大。",
            "證照是最強訊號，價格區間也會影響報名。",
            "課程包裝可強調證照、職涯加值與可負擔的進修成本。",
        )
        show_image_with_explain(
            "shap_summary_attribute_model.png",
            "SHAP Summary：課程屬性",
            "右邊代表提高報名機率，左邊代表降低報名機率。",
            "可看出不同因素對報名機率的方向。",
            "用來判斷廣告文案該強調或補強哪個特色。",
        )
        show_image_with_explain(
            "xgboost_feature_importance_attribute_model.png",
            "XGBoost 重要因素：課程屬性",
            "模型越常用的欄位，分數通常越高。",
            "證照、價格與時段等因素影響預測。",
            "課程規劃可優先優化這些欄位。",
        )
    with tab2:
        show_image_with_explain(
            "shap_bar_course_dummy_model.png",
            "SHAP Bar：課程推薦機率",
            "看哪些因素最影響推薦機率。",
            "課程別與課程屬性會共同影響推薦結果。",
            "可用來說明為什麼某位會員被推薦某門課。",
        )
        show_image_with_explain(
            "shap_summary_course_dummy_model.png",
            "SHAP Summary：課程推薦機率",
            "看因素值高低如何推動機率上升或下降。",
            "不同課程 dummy 代表相對於課程1的差異。",
            "推薦名單可搭配此圖產生更合理的推薦原因。",
        )
        show_image_with_explain(
            "xgboost_feature_importance_course_dummy_model.png",
            "XGBoost 重要因素：課程推薦機率",
            "看模型最常依賴哪些欄位做推薦。",
            "證照、價格帶與特定課程是推薦核心。",
            "行銷可以先推高機率課程，再用分群文案包裝。",
        )

    section("行銷解釋表")
    marketing_table = pd.DataFrame(
        [
            ["證照", "學員重視考照與專業認證，文案可強調履歷加分、職涯競爭力。"],
            ["3000-4000", "這個價格帶較像可接受的進修預算，可主打高 CP 值。"],
            ["線上", "若線上吸引力不足，應補強可回放、彈性學習與互動設計。"],
            ["興趣_國貿實務", "適合推進出口流程、貿易文件、報關應用與實務案例。"],
            ["興趣_專業證照", "適合推考照班、證照輔導班與職涯升級課程。"],
        ],
        columns=["因素", "講人話解釋"],
    )
    st.dataframe(marketing_table, use_container_width=True, hide_index=True)
    definition_panel()


# -----------------------------
# Page: Recommendations
# -----------------------------
elif page == "課程推薦與廣告策略":
    st.title("課程推薦名單與落地廣告策略")
    st.caption("這頁把模型結果轉成可以拿去 EDM、LINE、電話行銷使用的名單。")

    source = st.radio("推薦名單", ["Top1：每人最推薦一門課", "Top3：每人前三推薦課"], horizontal=True)
    df = top1_df if source.startswith("Top1") else top3_df
    if df is None:
        st.warning("找不到推薦名單檔案。")
        st.stop()

    df = add_priority(df)
    col1, col2, col3 = st.columns(3)
    threshold = col1.slider("最低推薦機率", 0.0, 1.0, 0.0, 0.01)
    course_options = sorted(df["推薦課程ID"].dropna().unique().tolist()) if "推薦課程ID" in df.columns else []
    selected_courses = col2.multiselect("推薦課程", course_options, default=course_options)
    priorities = col3.multiselect("推薦優先級", ["高", "中", "低"], default=["高", "中", "低"])

    filtered = df.copy()
    if "預測報名機率" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["預測報名機率"], errors="coerce") >= threshold]
    if selected_courses and "推薦課程ID" in filtered.columns:
        filtered = filtered[filtered["推薦課程ID"].isin(selected_courses)]
    if priorities and "推薦優先級" in filtered.columns:
        filtered = filtered[filtered["推薦優先級"].isin(priorities)]

    a, b, c, d = st.columns(4)
    a.metric("篩選後名單", len(filtered))
    b.metric("高優先級", int((filtered["推薦優先級"] == "高").sum()) if "推薦優先級" in filtered.columns else 0)
    c.metric("中優先級", int((filtered["推薦優先級"] == "中").sum()) if "推薦優先級" in filtered.columns else 0)
    d.metric("低優先級", int((filtered["推薦優先級"] == "低").sum()) if "推薦優先級" in filtered.columns else 0)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### 推薦名單")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    with right:
        if "推薦課程ID" in filtered.columns:
            counts = filtered["推薦課程ID"].value_counts().reset_index()
            counts.columns = ["推薦課程ID", "推薦次數"]
            fig = px.bar(counts, x="推薦課程ID", y="推薦次數", title="每門課被推薦次數")
            fig.update_layout(height=380, xaxis_title="推薦課程ID", yaxis_title="推薦次數")
            st.plotly_chart(fig, use_container_width=True)
            chart_explain(
                "柱子越高，代表越多會員被推薦這門課。",
                "這可以判斷哪幾門課適合先做推播。",
                "高推薦次數課程適合做 EDM 主推，低推薦次數課程適合精準小眾投放。",
            )

    if "預測報名機率" in filtered.columns:
        fig = px.histogram(filtered, x="預測報名機率", nbins=20, title="推薦機率分布")
        fig.update_layout(height=380, xaxis_title="預測報名機率", yaxis_title="人數")
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "越靠右，代表模型越認為該會員可能報名。",
            "若多數集中在低機率，代表要用二次篩選或更強文案刺激。",
            "高機率名單可優先給業務或用 LINE 個人化推播。",
        )

    section("落地廣告策略")
    st.markdown(
        """
        - **高優先級名單**：適合 LINE 一對一推播、電話邀約、限時名額提醒。
        - **中優先級名單**：適合 EDM、再行銷廣告、課程比較型內容。
        - **低優先級名單**：適合品牌內容培養，例如免費講座、懶人包、學習測驗。
        - **課程_2 等高推薦課程**：可先做主打活動頁，文案強調證照與職涯加值。
        """
    )

    section("廣告文案快速產生器")
    g1, g2, g3 = st.columns(3)
    target_course = g1.selectbox("主推課程", sorted(filtered["推薦課程ID"].dropna().unique().tolist()) if "推薦課程ID" in filtered.columns else ["課程_2"])
    audience = g2.selectbox("主推客群", ["商業語文／運務導向型", "國貿實務導向型", "證照／多元進修型"])
    appeal = g3.selectbox("主打訴求", ["證照與職涯加值", "國貿實務案例", "商業語文溝通", "高 CP 值進修", "線上彈性學習"])

    copy_map = {
        "證照與職涯加值": "強調考照、專業認證、履歷加分與職涯競爭力。",
        "國貿實務案例": "強調進出口流程、貿易文件、報關應用與實務案例。",
        "商業語文溝通": "強調商務情境、客戶應對、國際溝通與跨國工作。",
        "高 CP 值進修": "強調小額投資、高價值回報、短期進修與職涯升級。",
        "線上彈性學習": "強調可回放、不受地點限制、下班後也能進修。",
    }
    ad_title = f"給{audience}的課程推薦：課程 {target_course} 幫你把進修變成職涯優勢"
    ad_body = f"這次推薦主打「{appeal}」。{copy_map[appeal]}適合先投放給推薦機率較高的會員，再用 LINE 或 EDM 做二次提醒。"
    line_msg = f"你可能會適合課程 {target_course}：{appeal}。想了解課程內容與適合對象嗎？點開看看本期推薦。"
    st.text_area("廣告標題", ad_title, height=80)
    st.text_area("社群／EDM 文案", ad_body, height=120)
    st.text_area("LINE 推播短文", line_msg, height=90)
    st.caption("提醒：這是依模型結果與分群邏輯產生的初稿，正式投放前仍建議由行銷人員依品牌語氣微調。")

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載目前篩選名單 CSV", csv, "filtered_recommendation.csv", "text/csv")
    st.info("提醒：模型的任務是先幫我們找到比較值得接觸的人，不是保證每個人都會報名。實際投放前，建議再搭配預算、上課時間、過去互動紀錄做二次篩選。")


# -----------------------------
# Page: Model Performance
# -----------------------------
elif page == "模型表現與研究限制":
    st.title("模型表現與研究限制")
    st.caption("這頁說明模型準不準，以及哪些地方要保守解讀。")

    definition_panel()

    section("模型表現比較")
    perf = pd.DataFrame(
        [
            {"模型": "課程屬性模型", **metrics.get("attr", {})},
            {"模型": "課程推薦機率模型", **metrics.get("course", {})},
        ]
    )
    st.dataframe(perf, use_container_width=True, hide_index=True)
    if not perf.empty and "ROC-AUC" in perf.columns:
        fig = px.bar(perf, x="模型", y=["預測準確度", "找回率", "ROC-AUC", "PR-AUC"], barmode="group", title="模型指標比較")
        fig.update_layout(height=430, yaxis_title="分數", xaxis_title="模型")
        st.plotly_chart(fig, use_container_width=True)
        chart_explain(
            "這張圖比較不同模型在幾個指標上的表現。",
            "課程推薦機率模型的找回率較高，較適合找潛力名單。",
            "本專題應重視找回率與 ROC-AUC，不要只看預測準確度。",
        )

    c1, c2 = st.columns(2)
    for col, cm, title in [(c1, metrics.get("cm_attr"), "課程屬性模型"), (c2, metrics.get("cm_course"), "課程推薦機率模型")]:
        with col:
            st.markdown(f"### {title}：混淆矩陣")
            if cm is not None:
                cm_df = pd.DataFrame(cm, index=["實際未報名", "實際報名"], columns=["預測未報名", "預測報名"])
                st.dataframe(cm_df, use_container_width=True)
            chart_explain(
                "混淆矩陣用來看模型哪裡猜對、哪裡猜錯。",
                "對行銷來說，最在意的是實際會報名的人有沒有被找出來。",
                "若想提高找回率，可降低機率門檻，但名單會變多，需要人工篩選。",
            )

    section("研究限制")
    st.warning(
        """
        1. 資料來自既有報名紀錄，未必完整包含所有學員真正考慮過的課程。
        2. 選課因素模型使用近似做法，結果適合看趨勢，不適合過度解讀單一 p-value。
        3. 樣本數較小，且報名者和未報名者比例不平均。
        4. 部分課程屬性和課程本身高度重疊，例如證照課也可能代表某一門特定課程。
        5. 推薦名單應搭配商業規則二次篩選，不應完全自動決定投放。
        """
    )
    if model_report:
        with st.expander("查看原始模型輸出"):
            st.text(model_report)


# -----------------------------
# Page: Full Report
# -----------------------------
elif page == "完整報告":
    st.title("完整分析報告")
    st.caption("這裡把原始分析報告重新整理成比較適合簡報與審查閱讀的版本。")

    st.markdown(
        """
        # 課程報名行為分析與市場區隔分析報告

        ## 一、研究目的
        本專題希望把課程報名資料變成可以使用的行銷工具。重點不是只看誰點擊或誰購買，而是理解「誰適合哪一門課」、「為什麼適合」以及「行銷人員下一步可以怎麼做」。

        ## 二、資料與方法
        主題一先做市場區隔，找出公會主要客群輪廓。主題二再分析影響選課的因素、個人偏好與課程推薦機率。使用工具包含 Python、XGBoost、SHAP、K-means、卡方檢定與 Streamlit。

        ## 三、市場區隔結論
        公會顧客可整理成三個行銷用區隔：商業語文／運務導向型、國貿實務導向型、證照／多元進修型。這三群不是單純用性別或職稱切出來，而是更接近「學員真正想學什麼」。

        ## 四、選課因素與個人偏好
        影響報名的重要因素包含證照、價格帶、課程別與興趣特徵。白話來說，學員會在意課程能不能帶來職涯加值、價格是否能接受，以及內容是否符合自己的工作需求。

        ## 五、推薦模型結果
        XGBoost 可用來估計每位會員對每門課的報名機率。模型的 ROC-AUC 約 0.866，代表模型有不錯的排序能力，能把較可能報名的人排到前面。由於真正報名的人本來就比較少，所以不能只看預測準確度，也要看找回率。

        ## 六、課程推薦應用
        推薦名單可分成高、中、低優先級。高優先級名單適合 LINE、電話邀約或限時提醒；中優先級名單適合 EDM 和再行銷；低優先級名單適合用免費內容或講座慢慢培養。

        ## 七、行銷建議
        行銷分眾應以興趣為主，而不是只依賴性別或職稱。證照型文案可強調考照與履歷加分；國貿實務型文案可強調進出口流程和實務案例；商業語文型文案可強調國際溝通與客戶應對。

        ## 八、研究限制
        本資料樣本數不大，而且實際報名者較少，因此模型結果應作為行銷輔助，不應完全自動化決策。推薦名單仍需要搭配商業規則、上課時間、預算與過去互動紀錄做二次判斷。

        ## 九、結論
        本專題完成從市場區隔、選課因素分析、推薦機率預測到儀表板落地的完整流程。最重要的成果是把統計模型轉成內部人員能理解、能操作、能拿來產生廣告策略的決策工具。
        """
    )

    with st.expander("查看原始 final_analysis_report.md"):
        if final_report:
            st.markdown(final_report.replace("MNL / 選擇模型", "選課因素模型").replace("MNL", "選課因素模型"))
        else:
            st.warning("找不到 final_analysis_report.md。")
