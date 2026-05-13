# 詳細分析流程與成果
## A. 分析流程
1. 讀取 Excel 全部工作表並檢查欄位、型態、缺失值。
2. 驗證主題一 long format 規則（每位學員 8 筆、且僅 1 筆 Y=1）。
3. 建立 Model 1A/1B（優先 ConditionalLogit，否則 Logit 近似）。
4. 建立 XGBoost（屬性版、課程dummy版），輸出分類指標。
5. 進行 SHAP 解釋（summary + bar + 重要度表）。
6. 依 predict_proba 產出 Top1 / Top3 課程推薦與推薦原因。
7. 主題二進行 KMeans K=2~6、選最佳 K、輸出群輪廓。
8. 執行 cluster 與類別變數卡方檢定。
## B. 主要成果
- 主題一 target 分布：Y=0 665、Y=1 95（不平衡比 7.00）
- XGBoost 屬性模型：{'Accuracy': 0.7960526315789473, 'Precision': 0.2857142857142857, 'Recall': 0.42105263157894735, 'F1': 0.3404255319148936, 'ROC_AUC': 0.8672338741590819, 'PR_AUC': 0.33927086141225615}
- XGBoost 課程dummy模型：{'Accuracy': 0.8026315789473685, 'Precision': 0.3103448275862069, 'Recall': 0.47368421052631576, 'F1': 0.375, 'ROC_AUC': 0.8664424218440838, 'PR_AUC': 0.3370027680158901}
- 主題二最佳 K：2
## C. 檔案清單
- cluster_profile_heatmap.png
- data_check_report.txt
- final_analysis_report.md
- kmeans_elbow_plot.png
- kmeans_silhouette_plot.png
- shap_bar_attribute_model.png
- shap_bar_course_dummy_model.png
- shap_summary_attribute_model.png
- shap_summary_course_dummy_model.png
- topic1_logistic_or_mnl_coefficients.csv
- topic1_model_performance_report.txt
- topic1_recommendation_top1.csv
- topic1_recommendation_top3.csv
- topic1_recommendations.xlsx
- topic1_shap_importance.csv
- topic1_xgboost_feature_importance.csv
- topic2_chi_square_results.csv
- topic2_cluster_profile.csv
- xgboost_feature_importance_attribute_model.png
- xgboost_feature_importance_course_dummy_model.png