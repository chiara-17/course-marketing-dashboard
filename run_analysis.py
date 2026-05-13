#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
課程報名行為分析與市場區隔分析

用途：
1) 從頭讀取 Excel
2) 完成資料檢查
3) 主題一：MNL/Logit近似 + XGBoost + SHAP + 推薦
4) 主題二：K-means + 輪廓 + 卡方檢定
5) 輸出所有指定檔案與圖表

執行方式（Windows PowerShell）：
  $env:PYTHONIOENCODING="utf-8"
  python run_analysis.py
"""

from __future__ import annotations

import os
import glob
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.stats import chi2_contingency
from xgboost import XGBClassifier
import shap
import statsmodels.api as sm

try:
    from statsmodels.discrete.conditional_models import ConditionalLogit
    HAS_CLOGIT = True
except Exception:
    HAS_CLOGIT = False


# ---------- 全域設定 ----------
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = r"C:\Users\Lin17\Documents\Codex\2026-05-13\files-mentioned-by-the-user-dummy"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文欄位常數，避免手動輸入造成編碼風險
COL_SID = "學員ID"
COL_Y = "Y_是否選擇"
COL_GENDER = "性別編"
COL_COURSE_PREFIX = "課程_"
COL_COURSE_RAW = "課程ID_原始_查核"
COL_COURSE_DUMMY_CHECK = "課程dummy檢查"
COL_TITLE_PREFIX = "職稱_"
COL_INTEREST_PREFIX = "興趣_"


@dataclass
class ChoiceModelResult:
    model_type: str
    coef: pd.Series
    pvalue: pd.Series
    odds_ratio: pd.Series
    dropped_ref: List[str]


def find_excel_file() -> str:
    files = [
        f
        for f in glob.glob(r"C:\Users\Lin17\Downloads\*Dummy*.xlsx")
        if not os.path.basename(f).startswith("~$")
    ]
    if not files:
        raise FileNotFoundError("找不到目標 Excel 檔案（Downloads 下 *Dummy*.xlsx）")
    return files[0]


def select_numeric_features(df: pd.DataFrame, exclude_extra: List[str] | None = None) -> List[str]:
    exclude = {
        COL_Y,
        COL_SID,
        "row_index",
        "學員列號",
        "課程版本",
        "有興趣課程",
        "課程ID",
    }
    if exclude_extra:
        exclude |= set(exclude_extra)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def drop_collinear_reference_cols(x: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # 針對已知互斥二元欄位，自動刪除一欄作參照組
    pairs = [("晚上", "白天"), ("線上", "實體"), ("一週兩次", "一週一次")]
    dropped = [b for a, b in pairs if a in x.columns and b in x.columns]
    return x.drop(columns=dropped, errors="ignore"), dropped


def run_data_check(sheets: Dict[str, pd.DataFrame], course_df: pd.DataFrame, attr_df: pd.DataFrame) -> Dict[str, object]:
    lines: List[str] = []
    lines.append("=== 資料檢查報告 ===")
    lines.append("Sheet 名稱：" + ", ".join(sheets.keys()))
    lines.append("")

    for name, df in sheets.items():
        lines.append(f"[{name}] shape={df.shape}")
        lines.append("欄位：" + ", ".join(map(str, df.columns.tolist())))
        lines.append("前5列：")
        lines.append(df.head(5).to_string(index=False))
        lines.append(f"缺失值總數：{int(df.isna().sum().sum())}")
        lines.append("型態：" + ", ".join([f"{k}:{v}" for k, v in df.dtypes.astype(str).items()]))
        lines.append("")

    problems: List[str] = []
    fixes: List[str] = []

    for nm, df in [("主題一_課程Dummy模型", course_df), ("主題一_屬性模型", attr_df)]:
        cnt = df.groupby(COL_SID).size()
        if (cnt != 8).any():
            problems.append(f"{nm}：不是每位學員 8 筆")
        else:
            fixes.append(f"{nm}：每位學員均為 8 筆")

        y_uniq = set(pd.to_numeric(df[COL_Y], errors="coerce").dropna().unique().tolist())
        if not y_uniq.issubset({0, 1}):
            problems.append(f"{nm}：{COL_Y} 含非 0/1，已重編碼")
            df[COL_Y] = df[COL_Y].apply(lambda v: 1 if v == 1 else 0)
            fixes.append(f"{nm}：{COL_Y} 已重編碼為 0/1")

    y_sum = course_df.groupby(COL_SID)[COL_Y].sum()
    if (y_sum != 1).any():
        problems.append("主題一_課程Dummy模型：非每位學員剛好 1 筆 Y=1")
    else:
        fixes.append("主題一_課程Dummy模型：每位學員剛好 1 筆 Y=1")

    dcols = [c for c in course_df.columns if str(c).startswith(COL_COURSE_PREFIX)]
    for c in dcols:
        vals = set(pd.to_numeric(course_df[c], errors="coerce").dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            problems.append(f"{c} 含非 0/1")

    if COL_COURSE_RAW in course_df.columns and dcols:
        s = course_df[dcols].sum(axis=1)
        ok = ((course_df[COL_COURSE_RAW] == 1) & (s == 0)) | ((course_df[COL_COURSE_RAW] > 1) & (s == 1))
        if not ok.all():
            problems.append("課程 dummy 與原始課程ID 查核不一致")
        else:
            fixes.append("課程 dummy 與原始課程ID 查核一致")

    y0 = int((course_df[COL_Y] == 0).sum())
    y1 = int((course_df[COL_Y] == 1).sum())
    imbalance = max(y0, y1) / max(1, min(y0, y1))

    lines.append("=== 主題一 target 分布 ===")
    lines.append(f"Y=0：{y0}")
    lines.append(f"Y=1：{y1}")
    lines.append(f"不平衡比（大/小）：{imbalance:.2f}")
    lines.append(f"是否不平衡（>1.5）：{'是' if imbalance > 1.5 else '否'}")
    lines.append("")
    lines.append("=== 發現問題 ===")
    lines.extend(problems if problems else ["無重大問題"])
    lines.append("")
    lines.append("=== 修正 / 確認 ===")
    lines.extend(fixes if fixes else ["無"])

    with open(os.path.join(OUTPUT_DIR, "data_check_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "problems": problems,
        "fixes": fixes,
        "y0": y0,
        "y1": y1,
        "imbalance": imbalance,
    }


def fit_choice_model(df: pd.DataFrame, features: List[str]) -> ChoiceModelResult:
    y = df[COL_Y].astype(int)
    x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    x, dropped = drop_collinear_reference_cols(x)

    if HAS_CLOGIT:
        try:
            model = ConditionalLogit(y, x, groups=df[COL_SID])
            res = model.fit(disp=False)
            return ChoiceModelResult(
                model_type="ConditionalLogit",
                coef=res.params,
                pvalue=res.pvalues,
                odds_ratio=np.exp(res.params),
                dropped_ref=dropped,
            )
        except Exception:
            pass

    # fallback 1: 일반 Logit
    x2 = sm.add_constant(x, has_constant="add")
    model = sm.Logit(y, x2)
    try:
        res = model.fit(disp=False, maxiter=300)
        return ChoiceModelResult(
            model_type="Logit近似MNL",
            coef=res.params,
            pvalue=res.pvalues,
            odds_ratio=np.exp(res.params),
            dropped_ref=dropped,
        )
    except Exception:
        # fallback 2: regularized logit（處理奇異矩陣）
        res = model.fit_regularized(disp=False, maxiter=300)
        return ChoiceModelResult(
            model_type="Logit近似MNL_regularized",
            coef=res.params,
            pvalue=pd.Series(np.nan, index=res.params.index),
            odds_ratio=np.exp(res.params),
            dropped_ref=dropped,
        )


def eval_xgboost(df: pd.DataFrame, model_tag: str):
    feats = select_numeric_features(df, exclude_extra=[COL_COURSE_RAW, COL_COURSE_DUMMY_CHECK])
    x = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[COL_Y].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "PR_AUC": average_precision_score(y_test, prob),
    }
    cm = confusion_matrix(y_test, pred)

    imp = pd.DataFrame({"feature": feats, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    fig_name = (
        "xgboost_feature_importance_attribute_model.png"
        if model_tag == "attribute"
        else "xgboost_feature_importance_course_dummy_model.png"
    )
    plt.figure(figsize=(9, 6))
    top = imp.head(20).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title(f"XGBoost Feature Importance ({model_tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), dpi=200)
    plt.close()

    # SHAP
    explainer = shap.TreeExplainer(model)
    sample_x = x_test if len(x_test) <= 300 else x_test.sample(300, random_state=42)
    shap_values = explainer.shap_values(sample_x)
    shap_imp = pd.DataFrame(
        {"feature": sample_x.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    sum_name = (
        "shap_summary_attribute_model.png"
        if model_tag == "attribute"
        else "shap_summary_course_dummy_model.png"
    )
    bar_name = (
        "shap_bar_attribute_model.png"
        if model_tag == "attribute"
        else "shap_bar_course_dummy_model.png"
    )
    shap.summary_plot(shap_values, sample_x, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, sum_name), dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, sample_x, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, bar_name), dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "model": model,
        "features": feats,
        "metrics": metrics,
        "cm": cm,
        "importance": imp,
        "shap_importance": shap_imp,
        "x_all": x,
        "pred_all_prob": model.predict_proba(x)[:, 1],
    }


def build_recommendation(course_df: pd.DataFrame, course_xgb_result: Dict[str, object]):
    df = course_df.copy()
    df["pred_prob"] = course_xgb_result["pred_all_prob"]

    if COL_COURSE_RAW in df.columns:
        df["課程ID"] = df[COL_COURSE_RAW]
    else:
        dcols = [c for c in df.columns if str(c).startswith(COL_COURSE_PREFIX)]
        df["課程ID"] = 1
        if dcols:
            mapped = df[dcols].idxmax(axis=1).str.replace(COL_COURSE_PREFIX, "", regex=False)
            df.loc[df[dcols].sum(axis=1) > 0, "課程ID"] = mapped.astype(int)

    top_features = course_xgb_result["shap_importance"]["feature"].head(8).tolist()

    def gen_reason(row: pd.Series) -> str:
        reasons = []
        for f in top_features:
            if f in row.index and row[f] == 1:
                if str(f).startswith(COL_INTEREST_PREFIX):
                    reasons.append(f"對{str(f).replace(COL_INTEREST_PREFIX, '')}有興趣")
                elif str(f).startswith(COL_TITLE_PREFIX):
                    reasons.append(f"職稱屬於{str(f).replace(COL_TITLE_PREFIX, '')}族群")
                elif str(f).startswith(COL_COURSE_PREFIX):
                    reasons.append("該課程類型在模型中屬高潛力")
                else:
                    reasons.append(f"{f}特徵符合高報名樣態")
            if len(reasons) >= 2:
                break
        if not reasons:
            reasons.append("整體輪廓與高報名族群相近")
        return "；".join(reasons)

    cand = df[df[COL_Y] == 0].copy()
    cand["推薦原因"] = cand.apply(gen_reason, axis=1)

    top1 = (
        cand.sort_values([COL_SID, "pred_prob"], ascending=[True, False])
        .groupby(COL_SID)
        .head(1)[[COL_SID, "課程ID", "pred_prob", "推薦原因"]]
        .rename(columns={"課程ID": "推薦課程ID", "pred_prob": "預測報名機率"})
    )
    top3 = (
        cand.sort_values([COL_SID, "pred_prob"], ascending=[True, False])
        .groupby(COL_SID)
        .head(3)[[COL_SID, "課程ID", "pred_prob", "推薦原因"]]
        .rename(columns={"課程ID": "推薦課程ID", "pred_prob": "預測報名機率"})
    )

    top1.to_csv(os.path.join(OUTPUT_DIR, "topic1_recommendation_top1.csv"), index=False, encoding="utf-8-sig")
    top3.to_csv(os.path.join(OUTPUT_DIR, "topic1_recommendation_top3.csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "topic1_recommendations.xlsx"), engine="openpyxl") as writer:
        top1.to_excel(writer, sheet_name="top1", index=False)
        top3.to_excel(writer, sheet_name="top3", index=False)

    return top1, top3


def run_topic2(seg_df: pd.DataFrame):
    df = seg_df.copy()
    if df[COL_SID].nunique() != len(df):
        agg = {c: "max" for c in df.columns if c != COL_SID}
        df = df.groupby(COL_SID, as_index=False).agg(agg)

    feat_cols = [c for c in df.columns if c != COL_SID and pd.api.types.is_numeric_dtype(df[c])]
    x = df[feat_cols].fillna(0)
    xs = StandardScaler().fit_transform(x)

    ks = list(range(2, 7))
    inertias = []
    sils = []
    labels = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        lb = km.fit_predict(xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(xs, lb))
        labels[k] = lb

    best_k = ks[int(np.argmax(sils))]
    df["cluster"] = labels[best_k]

    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("KMeans Elbow Plot")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow_plot.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ks, sils, marker="o")
    plt.title("KMeans Silhouette Plot")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_silhouette_plot.png"), dpi=200)
    plt.close()

    title_cols = [c for c in df.columns if str(c).startswith(COL_TITLE_PREFIX)]
    interest_cols = [c for c in df.columns if str(c).startswith(COL_INTEREST_PREFIX)]

    if title_cols:
        df["職稱類別"] = df[title_cols].idxmax(axis=1).str.replace(COL_TITLE_PREFIX, "", regex=False)
        df.loc[df[title_cols].sum(axis=1) == 0, "職稱類別"] = "未標記"
    else:
        df["職稱類別"] = "未標記"

    if interest_cols:
        df["主要興趣"] = df[interest_cols].idxmax(axis=1).str.replace(COL_INTEREST_PREFIX, "", regex=False)
        df.loc[df[interest_cols].sum(axis=1) == 0, "主要興趣"] = "未標記"
    else:
        df["主要興趣"] = "未標記"

    profile_rows = []
    for c, g in df.groupby("cluster"):
        top_interest_keys = g[interest_cols].mean().sort_values(ascending=False).head(3).index.tolist() if interest_cols else []
        if any("專業證照" in k for k in top_interest_keys):
            seg_name = "證照導向型"
        elif any("線上課程" in k for k in top_interest_keys):
            seg_name = "線上學習型"
        elif any("國貿實務" in k for k in top_interest_keys):
            seg_name = "國貿實務型"
        else:
            seg_name = f"多元探索型_{c}"

        profile_rows.append(
            {
                "cluster": c,
                "segment_name": seg_name,
                "群體人數": len(g),
                "群體比例": len(g) / len(df),
                "性別編平均(1比例)": g[COL_GENDER].mean() if COL_GENDER in g.columns else np.nan,
                "主要職稱": g["職稱類別"].value_counts().idxmax(),
                "主要興趣": g["主要興趣"].value_counts().idxmax(),
            }
        )

    profile_df = pd.DataFrame(profile_rows).sort_values("cluster")
    profile_df.to_csv(os.path.join(OUTPUT_DIR, "topic2_cluster_profile.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.groupby("cluster")[feat_cols].mean(), cmap="YlGnBu")
    plt.title("Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_profile_heatmap.png"), dpi=200)
    plt.close()

    # 卡方檢定
    chi_rows = []
    test_vars = []
    if COL_GENDER in df.columns:
        test_vars.append(COL_GENDER)
    test_vars += ["職稱類別", "主要興趣"]
    for k in ["興趣_專業證照", "興趣_國貿實務"]:
        if k in df.columns:
            test_vars.append(k)

    for v in test_vars:
        ct = pd.crosstab(df["cluster"], df[v].astype(str))
        chi2, p, dof, _ = chi2_contingency(ct)
        chi_rows.append(
            {
                "variable": v,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "significant(p<0.05)": int(p < 0.05),
                "crosstab": ct.to_string(),
            }
        )

    chi_df = pd.DataFrame(chi_rows)
    chi_df.to_csv(os.path.join(OUTPUT_DIR, "topic2_chi_square_results.csv"), index=False, encoding="utf-8-sig")

    return best_k, profile_df, pd.DataFrame({"K": ks, "Inertia": inertias, "Silhouette": sils})


def save_topic1_reports(
    m1_attr: ChoiceModelResult,
    m1_course: ChoiceModelResult,
    xgb_attr: Dict[str, object],
    xgb_course: Dict[str, object],
):
    coef_rows = []
    for tag, res in [("Model1A_課程屬性模型", m1_attr), ("Model1B_課程Dummy模型", m1_course)]:
        for f in res.coef.index:
            coef_rows.append(
                {
                    "model": tag,
                    "model_type": res.model_type,
                    "feature": f,
                    "coef": float(res.coef[f]),
                    "p_value": float(res.pvalue[f]) if f in res.pvalue.index else np.nan,
                    "odds_ratio": float(res.odds_ratio[f]),
                }
            )
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(os.path.join(OUTPUT_DIR, "topic1_logistic_or_mnl_coefficients.csv"), index=False, encoding="utf-8-sig")

    fi_df = pd.concat(
        [
            xgb_attr["importance"].assign(model="XGBoost_1A_屬性模型"),
            xgb_course["importance"].assign(model="XGBoost_1B_課程Dummy模型"),
        ],
        ignore_index=True,
    )
    fi_df.to_csv(os.path.join(OUTPUT_DIR, "topic1_xgboost_feature_importance.csv"), index=False, encoding="utf-8-sig")

    shap_df = pd.concat(
        [
            xgb_attr["shap_importance"].assign(model="XGBoost_1A_屬性模型"),
            xgb_course["shap_importance"].assign(model="XGBoost_1B_課程Dummy模型"),
        ],
        ignore_index=True,
    )
    shap_df.to_csv(os.path.join(OUTPUT_DIR, "topic1_shap_importance.csv"), index=False, encoding="utf-8-sig")

    perf_lines = []
    perf_lines.append("=== 主題一模型表現報告 ===")
    for name, obj in [("XGBoost_1A_屬性模型", xgb_attr), ("XGBoost_1B_課程Dummy模型", xgb_course)]:
        perf_lines.append(f"[{name}]")
        for k, v in obj["metrics"].items():
            perf_lines.append(f"{k}: {v:.4f}")
        perf_lines.append("Confusion Matrix:")
        perf_lines.append(np.array2string(obj["cm"]))
        perf_lines.append("")

    perf_lines.append("[MNL/Logit 模型註記]")
    perf_lines.append(f"Model1A: {m1_attr.model_type}")
    perf_lines.append(f"Model1B: {m1_course.model_type}")
    if "ConditionalLogit" not in (m1_attr.model_type + m1_course.model_type):
        perf_lines.append("本研究因工具/收斂限制，以二元報名模型近似選擇模型。")
    if "regularized" in (m1_attr.model_type + m1_course.model_type):
        perf_lines.append("部份模型使用 regularized logit 解決奇異矩陣，p-value 可能不可得。")

    with open(os.path.join(OUTPUT_DIR, "topic1_model_performance_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(perf_lines))


def save_final_reports(
    data_check: Dict[str, object],
    m1_attr: ChoiceModelResult,
    m1_course: ChoiceModelResult,
    xgb_attr: Dict[str, object],
    xgb_course: Dict[str, object],
    top1: pd.DataFrame,
    best_k: int,
    profile_df: pd.DataFrame,
):
    top_pos_attr = (
        pd.DataFrame({"feature": m1_attr.coef.index, "coef": m1_attr.coef.values})
        .query("feature != 'const'")
        .sort_values("coef", ascending=False)
        .head(5)
    )
    top_neg_attr = (
        pd.DataFrame({"feature": m1_attr.coef.index, "coef": m1_attr.coef.values})
        .query("feature != 'const'")
        .sort_values("coef", ascending=True)
        .head(5)
    )
    top_pos_course = (
        pd.DataFrame({"feature": m1_course.coef.index, "coef": m1_course.coef.values})
        .query("feature != 'const'")
        .sort_values("coef", ascending=False)
        .head(5)
    )

    md = []
    md.append("# 課程報名行為分析與市場區隔分析報告")
    md.append("## 一、研究目的\n本研究目的為找出影響學員報名課程的關鍵因素，建立可解釋的預測模型，並進一步進行市場區隔，支持精準行銷與推薦。")
    md.append("## 二、資料來源與資料結構\n資料來自 Excel 檔案，主題一為 long format（學員×課程），主題二為學員層級資料。")
    md.append("## 三、資料前處理與變數編碼說明\n已進行缺失值、欄位型態、dummy 合法性、target 分布檢查。")
    md.append("## 四、老師建議後的修正：課程ID改為課程dummy\n以課程1為參照組，模型使用課程_2~課程_8，避免把課程ID當連續數值。")
    md.append("## 五、主題一分析：影響學員選課的決定性屬性\n分為課程屬性模型（Model 1A）與課程 dummy 模型（Model 1B）。")
    md.append(
        "## 六、MNL / 選擇模型結果\n"
        f"Model 1A：{m1_attr.model_type}\n\n"
        f"Model 1B：{m1_course.model_type}\n\n"
        "若因收斂或工具限制未使用 ConditionalLogit，已採二元模型近似並於限制中說明。"
    )
    md.append("Model 1A 正向係數前5：\n" + top_pos_attr.to_string(index=False))
    md.append("Model 1A 負向係數前5：\n" + top_neg_attr.to_string(index=False))
    md.append("Model 1B 正向係數前5：\n" + top_pos_course.to_string(index=False))
    md.append(
        "## 七、XGBoost 報名機率預測結果\n"
        f"屬性模型：{xgb_attr['metrics']}\n\n"
        f"課程dummy模型：{xgb_course['metrics']}\n\n"
        "本研究重視 Y=1 的 Recall，用於找出高潛力名單。"
    )
    md.append("## 八、SHAP 個人偏好與重要因素解釋\n已輸出 summary 與 bar 圖，並提供前20重要變數表。")
    md.append("## 九、課程推薦系統結果\n以 predict_proba 對未報名課程排序，輸出每位學員 Top1 / Top3 與推薦原因。")
    md.append("Top1 推薦範例前5：\n" + top1.head(5).to_string(index=False))
    md.append("## 十、主題二分析：市場區隔與輪廓描述\n以興趣、職稱、性別進行 K-means 分群。")
    md.append(f"## 十一、K-means 分群結果\n測試 K=2~6，最佳 K={best_k}（以 silhouette score 最大為準）。")
    md.append("群輪廓摘要：\n" + profile_df.to_string(index=False))
    md.append("## 十二、卡方檢定結果\n已輸出交叉表、chi-square、p-value 與顯著性判定。")
    md.append("## 十三、行銷策略建議\n可先用高召回模型圈選名單，再搭配人工作業或規則做第二層篩選，提升轉換效率。")
    md.append(
        "## 十四、研究限制\n1) 類別不平衡明顯。\n2) 若使用 Logit 近似 MNL，代表效用結構簡化。\n3) 部分係數在 regularized 情境下無法提供傳統 p-value。"
    )
    md.append("## 十五、結論\n課程屬性、課程別、個人興趣與職稱確實影響報名，模型可落地於推薦與分群行銷。")

    with open(os.path.join(OUTPUT_DIR, "final_analysis_report.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))

    # 額外：詳細流程與成果檔
    flow = []
    flow.append("# 詳細分析流程與成果")
    flow.append("## A. 分析流程")
    flow.append("1. 讀取 Excel 全部工作表並檢查欄位、型態、缺失值。")
    flow.append("2. 驗證主題一 long format 規則（每位學員 8 筆、且僅 1 筆 Y=1）。")
    flow.append("3. 建立 Model 1A/1B（優先 ConditionalLogit，否則 Logit 近似）。")
    flow.append("4. 建立 XGBoost（屬性版、課程dummy版），輸出分類指標。")
    flow.append("5. 進行 SHAP 解釋（summary + bar + 重要度表）。")
    flow.append("6. 依 predict_proba 產出 Top1 / Top3 課程推薦與推薦原因。")
    flow.append("7. 主題二進行 KMeans K=2~6、選最佳 K、輸出群輪廓。")
    flow.append("8. 執行 cluster 與類別變數卡方檢定。")
    flow.append("## B. 主要成果")
    flow.append(f"- 主題一 target 分布：Y=0 {data_check['y0']}、Y=1 {data_check['y1']}（不平衡比 {data_check['imbalance']:.2f}）")
    flow.append(f"- XGBoost 屬性模型：{xgb_attr['metrics']}")
    flow.append(f"- XGBoost 課程dummy模型：{xgb_course['metrics']}")
    flow.append(f"- 主題二最佳 K：{best_k}")
    flow.append("## C. 檔案清單")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        flow.append(f"- {fn}")

    with open(os.path.join(OUTPUT_DIR, "detailed_process_and_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(flow))


def main():
    print("=== 啟動分析 ===")
    print(f"輸出資料夾：{OUTPUT_DIR}")

    xlsx = find_excel_file()
    print(f"讀取檔案：{xlsx}")
    sheets = pd.read_excel(xlsx, sheet_name=None)
    sheet_names = list(sheets.keys())
    print("Sheet：", sheet_names)

    # 依你檔案固定順序抓取
    course_df = sheets[sheet_names[1]].copy()   # 主題一_課程Dummy模型
    attr_df = sheets[sheet_names[2]].copy()     # 主題一_屬性模型
    seg_df = sheets[sheet_names[3]].copy()      # 主題二_市場區隔資料

    data_check = run_data_check(sheets, course_df, attr_df)
    print("資料檢查完成。")

    print("建立主題一選擇模型...")
    m1_attr = fit_choice_model(attr_df, select_numeric_features(attr_df))
    m1_course = fit_choice_model(course_df, select_numeric_features(course_df, [COL_COURSE_RAW, COL_COURSE_DUMMY_CHECK]))

    print("建立 XGBoost 與 SHAP（主題一）...")
    xgb_attr = eval_xgboost(attr_df, "attribute")
    xgb_course = eval_xgboost(course_df, "course_dummy")

    print("輸出主題一報表...")
    save_topic1_reports(m1_attr, m1_course, xgb_attr, xgb_course)

    print("建立推薦名單...")
    top1, _ = build_recommendation(course_df, xgb_course)

    print("執行主題二分群與卡方...")
    best_k, profile_df, _ = run_topic2(seg_df)

    print("輸出最終報告...")
    save_final_reports(data_check, m1_attr, m1_course, xgb_attr, xgb_course, top1, best_k, profile_df)

    # 終端摘要（依你要求）
    print("\n=== 1) 資料檢查摘要 ===")
    print(f"Y=0: {data_check['y0']}, Y=1: {data_check['y1']}, 不平衡比: {data_check['imbalance']:.2f}")
    print(f"發現問題數: {len(data_check['problems'])}, 修正/確認數: {len(data_check['fixes'])}")

    print("\n=== 2) 主題一模型表現摘要 ===")
    print("XGBoost 1A（屬性模型）：", xgb_attr["metrics"])
    print("XGBoost 1B（課程Dummy模型）：", xgb_course["metrics"])

    print("\n=== 3) Top 10 重要變數 ===")
    print(xgb_course["importance"].head(10).to_string(index=False))

    print("\n=== 4) Top 5 推薦結果範例 ===")
    print(top1.head(5).to_string(index=False))

    print("\n=== 5) 主題二分群摘要 ===")
    print(f"最佳 K: {best_k}")
    print(profile_df.to_string(index=False))

    print("\n=== 6) 最終報告與輸出檔案位置 ===")
    print(OUTPUT_DIR)
    print("新增：detailed_process_and_results.md（詳細流程與成果）")
    print("完成。")


if __name__ == "__main__":
    main()

