# 課程精準行銷與會員推薦儀表板

## 1. 專案簡介
本專案使用 Streamlit 建立互動式 dashboard，整合：
- 報名因素分析（MNL/Logit、XGBoost、SHAP）
- 課程推薦名單（Top1 / Top3）
- 市場區隔分析（K-means、卡方檢定）
- 行銷策略產生器（規則式文案）
- 模型表現與研究限制說明

儀表板名稱：
**「課程精準行銷與會員推薦儀表板」**

---

## 2. 需要的檔案
請確認專案目錄下存在 `outputs/`，並包含以下分析結果（可缺檔，程式會警示但不崩潰）：

### 表格 / 文字
- `topic1_model_performance_report.txt`
- `topic1_logistic_coefficients.csv` 或 `topic1_logistic_or_mnl_coefficients.csv`
- `topic1_xgboost_feature_importance.csv`
- `topic1_shap_importance.csv`
- `topic1_recommendation_top1.csv`
- `topic1_recommendation_top3.csv`
- `topic2_cluster_profile.csv`
- `topic2_chi_square_results.csv`
- `final_analysis_report.md`

### 圖片
- `xgboost_feature_importance_attribute_model.png`
- `xgboost_feature_importance_course_dummy_model.png`
- `shap_summary_attribute_model.png`
- `shap_bar_attribute_model.png`
- `shap_summary_course_dummy_model.png`
- `shap_bar_course_dummy_model.png`
- `kmeans_elbow_plot.png`
- `kmeans_silhouette_plot.png`
- `cluster_profile_heatmap.png`

---

## 3. 如何安裝套件
```bash
pip install -r requirements.txt
```

---

## 4. 如何啟動 Streamlit
```bash
streamlit run app.py
```

---

## 5. Dashboard 頁面說明
左側 Sidebar 共 6 頁：

1. **總覽**
- KPI 卡片（學員數、課程數、推薦名單數、ROC-AUC、Top1/Top3）
- 研究發現摘要

2. **報名因素分析**
- SHAP / Feature Importance 圖片
- SHAP Top20 表格
- 行銷解釋表

3. **課程推薦名單**
- Top1 / Top3 切換
- 機率門檻、課程ID、優先級篩選
- 高中低優先級統計
- 推薦次數圖與高潛力名單
- 篩選結果下載 CSV

4. **市場區隔分析**
- 分群輪廓表、群人數長條圖、比例圓餅圖
- Elbow / Silhouette / Heatmap 圖
- 卡方檢定結果與白話結論

5. **行銷策略產生器**
- 依課程、客群、賣點、平台、語氣產生文案
- 產出廣告標題、社群文案、LINE 推播、EDM 標題、短影音腳本、AI Prompt

6. **模型表現與研究限制**
- Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC
- Confusion Matrix
- 研究限制清單

---

## 6. 注意事項
- 程式使用 `outputs_path = Path("outputs")` 讀取資料。
- 已實作：
  - `safe_read_csv()`
  - `safe_read_txt()`
  - `safe_load_image()`
- 若檔案不存在，會在畫面顯示 warning，不會中斷程式。
- CSV 編碼若出現亂碼，建議優先使用 `utf-8-sig` 或直接改讀 xlsx 版本。

