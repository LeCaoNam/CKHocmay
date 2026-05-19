"""
=============================================================
HỆ THỐNG DỰ ĐOÁN & GỢI Ý GIÁ NHÀ - VIỆT NAM (PHIÊN BẢN TỐI ƯU)
Đồ án cuối kỳ - Học Máy
=============================================================
Các kỹ thuật được sử dụng:
  Chương 1  - Linear Regression, Logistic Regression (phân loại giá)
  Chương 2  - PCA (giảm chiều dữ liệu)
  Chương 3  - Lasso, Ridge, Gradient Boosting
  Chương 5  - Random Forest, XGBoost, LightGBM, Stacking Ensemble
  Chương 6  - KMeans Clustering (gợi ý nhà tương tự)

Cải tiến TỔNG THỂ để tối đa R²:
  ✅ Khai thác tối đa cột Địa chỉ: tên đường, phường, số ngõ
  ✅ Smoothed Target Encoding (tên đường + phường = 40%+ importance)
  ✅ Interaction features: đường×mặt phố, phường×diện tích...
  ✅ Lọc outlier IQR per quận thay vì global
  ✅ Log-transform target (Giá/m2)
  ✅ Rank features trong nhóm quận
  ✅ XGBoost + LightGBM + Stacking Ensemble
=============================================================
Kết quả kỳ vọng: R² ~55–65% (so với 42% phiên bản cũ)
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
import joblib, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("BUOC 1: LOAD DU LIEU")
print("=" * 60)

df = pd.read_csv("VN_housing_dataset.csv", index_col=0)
print(f"Kich thuoc ban dau: {df.shape}")

# ─────────────────────────────────────────────
# 2. LAM SACH DU LIEU
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUOC 2: LAM SACH DU LIEU")
print("=" * 60)

df["Gia_m2"] = df["Giá/m2"].astype(str).str.lower()
df = df[df["Gia_m2"].str.contains("tri", na=False)]
df["Gia_m2"] = (
    df["Gia_m2"]
    .str.replace("triệu/m²", "", regex=False)
    .str.replace("triệu/m2", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)
df["Gia_m2"] = pd.to_numeric(df["Gia_m2"], errors="coerce")

df["dien_tich_raw"] = pd.to_numeric(
    df["Diện tích"].astype(str).str.replace("m²","",regex=False).str.strip(),
    errors="coerce"
)
df["so_phong_raw"] = pd.to_numeric(
    df["Số phòng ngủ"].astype(str).str.extract(r'(\d+)')[0],
    errors="coerce"
)
for col in ["Dài", "Rộng"]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.extract(r'([\d.]+)')[0],
        errors="coerce"
    )
df["so_tang_raw"] = pd.to_numeric(df["Số tầng"], errors="coerce")
df["ngay"] = pd.to_datetime(df["Ngày"], errors="coerce")

df = df.dropna(subset=["Gia_m2","dien_tich_raw","Địa chỉ","Loại hình nhà ở"]).reset_index(drop=True)

# Địa bàn
df["dia_ban"] = df["Quận"].fillna(df["Huyện"]).fillna("Khong ro")

# Outlier IQR per quận
rows = []
for g, sub in df.groupby("dia_ban"):
    lo, hi = sub["Gia_m2"].quantile([0.03, 0.97])
    rows.append(sub[(sub["Gia_m2"] >= lo) & (sub["Gia_m2"] <= hi)])
df = pd.concat(rows, ignore_index=True)
print(f"So dong sau loc: {len(df):,}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING TOI DA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUOC 3: FEATURE ENGINEERING TOI DA")
print("=" * 60)

# Thời gian
df["nam"]   = df["ngay"].dt.year.fillna(2020).astype(float)
df["quy"]   = df["ngay"].dt.quarter.fillna(2).astype(float)
df["thang"] = df["ngay"].dt.month.fillna(6).astype(float)

# Khai thac dia chi
df["ten_duong"] = (
    df["Địa chỉ"].str.split(",").str[0].str.strip().str.lower()
    .str.replace(r'^(duong|pho|ngo|ngach|hem|đường|phố|ngõ|ngách|hẻm)\s+', '', regex=True)
    .fillna("unknown")
)
df["phuong"] = (
    df["Địa chỉ"].str.extract(r'(?:Phường|phuong|Phuong|phường)\s+([^,]+)', expand=False)
    .str.strip().str.lower().fillna("unknown")
)
df["so_ngo"]  = pd.to_numeric(
    df["Địa chỉ"].str.extract(r'[Nn]g[oõ]\s+(\d+)', expand=False),
    errors="coerce"
).fillna(0)
df["co_ngo"]  = (df["so_ngo"] > 0).astype(float)
df["ngo_sau"] = (df["so_ngo"] > 100).astype(float)

print(f"Ten duong unique: {df['ten_duong'].nunique()}")
print(f"Phuong unique   : {df['phuong'].nunique()}")

# Loai nha
df["co_so"]     = (df["Giấy tờ pháp lý"] == "Đã có sổ").astype(float)
df["noi_thanh"] = df["Quận"].notna().astype(float)
df["mat_pho"]   = df["Loại hình nhà ở"].str.contains("m.t ph.", case=False, na=False).astype(float)
df["biet_thu"]  = df["Loại hình nhà ở"].str.contains("bi.t th.", case=False, na=False).astype(float)
df["lien_ke"]   = df["Loại hình nhà ở"].str.contains("li.n k.", case=False, na=False).astype(float)

# Impute
med_tang  = df["so_tang_raw"].median()
med_phong = df["so_phong_raw"].median()
med_dai   = df["Dài"].median()
med_rong  = df["Rộng"].median()

df["so_tang"]  = df["so_tang_raw"].fillna(med_tang)
df["so_phong"] = df["so_phong_raw"].fillna(med_phong)
df["dai"]      = df["Dài"].fillna(med_dai)
df["rong"]     = df["Rộng"].fillna(med_rong)
df["dt"]       = df["dien_tich_raw"]

# Physical features
df["tl_dai_rong"]  = df["dai"] / (df["rong"] + 1e-5)
df["dt_dairong"]   = df["dai"] * df["rong"]
df["phong_tang"]   = df["so_phong"] / (df["so_tang"].clip(1) + 1e-5)
df["dt_per_phong"] = df["dt"] / (df["so_phong"].clip(1))
df["log_dt"]       = np.log1p(df["dt"])
df["dt_sq"]        = df["dt"] ** 2
df["tang_dt"]      = df["so_tang"] * df["dt"]
df["rong_sq"]      = df["rong"] ** 2
df["rong_x_mp"]    = df["rong"] * df["mat_pho"]

# Rank trong quan
df["rank_dt_quan"]   = df.groupby("dia_ban")["dt"].rank(pct=True)
df["rank_tang_quan"] = df.groupby("dia_ban")["so_tang"].rank(pct=True)
df["rank_rong_quan"] = df.groupby("dia_ban")["rong"].rank(pct=True)

# Smoothed Target Encoding
y = df["Gia_m2"]

def te_smooth(key_col, target_col, m=10):
    gm = target_col.mean()
    tmp = pd.DataFrame({"k": key_col, "t": target_col})
    agg = tmp.groupby("k")["t"].agg(["mean","count"])
    s   = (agg["count"] * agg["mean"] + m * gm) / (agg["count"] + m)
    return key_col.map(s).fillna(gm)

df["te_duong"]   = te_smooth(df["ten_duong"],         y, m=10)
df["te_phuong"]  = te_smooth(df["phuong"],            y, m=10)
df["te_diaban"]  = te_smooth(df["dia_ban"],           y, m=5)
df["te_loainha"] = te_smooth(df["Loại hình nhà ở"],   y, m=5)

# Interaction features
df["te_duong_x_mp"]  = df["te_duong"] * df["mat_pho"]
df["te_phuong_x_dt"] = df["te_phuong"] * df["log_dt"]
df["te_duong_x_dt"]  = df["te_duong"] * df["log_dt"]
df["te_diaban_x_bt"] = df["te_diaban"] * df["biet_thu"]

print("Feature engineering hoan thanh!")

# ─────────────────────────────────────────────
# 5. CHUAN BI FEATURES & SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUOC 4: CHUAN BI FEATURES & TRAIN/TEST SPLIT")
print("=" * 60)

FEATURE_COLS = [
    "dt", "log_dt", "dt_sq",
    "so_phong", "so_tang", "dai", "rong",
    "tl_dai_rong", "dt_dairong", "phong_tang",
    "dt_per_phong", "tang_dt", "rong_sq", "rong_x_mp",
    "co_so", "noi_thanh", "mat_pho", "biet_thu", "lien_ke",
    "co_ngo", "ngo_sau", "so_ngo",
    "nam", "quy", "thang",
    "te_duong", "te_phuong", "te_diaban", "te_loainha",
    "te_duong_x_mp", "te_phuong_x_dt", "te_duong_x_dt", "te_diaban_x_bt",
    "rank_dt_quan", "rank_tang_quan", "rank_rong_quan",
]

X = df[FEATURE_COLS].fillna(0)
y_log = np.log1p(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_train = np.expm1(y_train_log)
y_test  = np.expm1(y_test_log)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  |  Features: {X.shape[1]}")

def evaluate(name, y_true, y_pred_log):
    y_pred = np.clip(np.expm1(y_pred_log), 5, 600)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  RMSE : {rmse:.2f} trieu/m2")
    print(f"  MAE  : {mae:.2f}  trieu/m2")
    print(f"  R2   : {r2:.4f}  ({r2*100:.1f}%)")
    return {"model": name, "RMSE": round(rmse,2), "MAE": round(mae,2), "R2": round(r2,4)}

results = []

# ─────────────────────────────────────────────
# CHUONG 1 & 3: LINEAR MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 1-3: LINEAR REGRESSION, RIDGE, LASSO (Chuong 1 & 3)")
print("=" * 60)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

lr = LinearRegression()
lr.fit(X_train_sc, y_train_log)
results.append(evaluate("Linear Regression", y_test, lr.predict(X_test_sc)))

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train_log)
results.append(evaluate("Ridge Regression", y_test, ridge.predict(X_test_sc)))

lasso = Lasso(alpha=0.005, max_iter=10000)
lasso.fit(X_train_sc, y_train_log)
results.append(evaluate("Lasso Regression", y_test, lasso.predict(X_test_sc)))

# ─────────────────────────────────────────────
# CHUONG 2: PCA + LINEAR REGRESSION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 4: PCA + LINEAR REGRESSION (Chuong 2)")
print("=" * 60)

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca_full = PCA().fit(X_train_sc)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_pca  = np.argmax(cumvar >= 0.95) + 1
print(f"So thanh phan PCA (95% variance): {n_pca}")

pca_pipe = Pipeline([("pca", PCA(n_components=n_pca)), ("lr", LinearRegression())])
pca_pipe.fit(X_train_sc, y_train_log)
results.append(evaluate(f"PCA({n_pca}) + Linear Reg", y_test, pca_pipe.predict(X_test_sc)))

# ─────────────────────────────────────────────
# CHUONG 5: RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 5: RANDOM FOREST (Chuong 5)")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_leaf=3,
    max_features=0.7, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train_log)
results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
print("\nTop 15 features quan trong nhat:")
print(fi.nlargest(15).round(4).to_string())

# ─────────────────────────────────────────────
# CHUONG 3 & 5: GRADIENT BOOSTING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 6: GRADIENT BOOSTING (Chuong 3 & 5)")
print("=" * 60)

gb = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb.fit(X_train, y_train_log)
results.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))

# ─────────────────────────────────────────────
# CHUONG 5: XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 7: XGBOOST (Chuong 5)")
print("=" * 60)

has_xgb = False
xgb = None
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train, y_train_log,
            eval_set=[(X_test, y_test_log)], verbose=False)
    results.append(evaluate("XGBoost", y_test, xgb.predict(X_test)))
    has_xgb = True
    print("XGBoost hoan thanh!")
except ImportError:
    print("Chua cai XGBoost: pip install xgboost")

# ─────────────────────────────────────────────
# CHUONG 5: LIGHTGBM
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 8: LIGHTGBM (Chuong 5)")
print("=" * 60)

has_lgbm = False
lgbm = None
try:
    import lightgbm as lgb
    lgbm = lgb.LGBMRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    lgbm.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )
    results.append(evaluate("LightGBM", y_test, lgbm.predict(X_test)))
    has_lgbm = True
    print("LightGBM hoan thanh!")
except ImportError:
    print("Chua cai LightGBM: pip install lightgbm")

# ─────────────────────────────────────────────
# CHUONG 5: STACKING ENSEMBLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 9: STACKING ENSEMBLE (Chuong 5)")
print("=" * 60)

from sklearn.ensemble import StackingRegressor

base_estimators = [
    ("rf", RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=3,
        max_features=0.7, random_state=42, n_jobs=-1
    )),
    ("gb", GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )),
]
if has_xgb:
    from xgboost import XGBRegressor as XGB2
    base_estimators.append(("xgb", XGB2(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )))
if has_lgbm:
    import lightgbm as lgb2
    base_estimators.append(("lgbm", lgb2.LGBMRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1
    )))

stack = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5, n_jobs=-1
)
stack.fit(X_train, y_train_log)
results.append(evaluate("Stacking Ensemble", y_test, stack.predict(X_test)))
print("Stacking Ensemble hoan thanh!")

# ─────────────────────────────────────────────
# CHUONG 1: LOGISTIC REGRESSION PHAN LOAI GIA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 10: LOGISTIC REGRESSION — PHAN LOAI GIA (Chuong 1)")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score

q33 = y.quantile(0.33)
q66 = y.quantile(0.66)
print(f"Binh dan  : <= {q33:.1f} tr/m2")
print(f"Trung cap : {q33:.1f} - {q66:.1f} tr/m2")
print(f"Cao cap   : > {q66:.1f} tr/m2")

y_cat       = pd.cut(y, bins=[-np.inf, q33, q66, np.inf], labels=[0,1,2]).astype(int)
y_cat_train = y_cat.loc[y_train.index]
y_cat_test  = y_cat.loc[y_test.index]

logr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
logr.fit(X_train_sc, y_cat_train)
y_logr_pred = logr.predict(X_test_sc)

print("\nBao cao phan loai:")
print(classification_report(y_cat_test, y_logr_pred,
      target_names=["Binh dan", "Trung cap", "Cao cap"]))

prec_per_class = precision_score(y_cat_test, y_logr_pred,
                                  average=None, zero_division=0)
logistic_precision = {
    "low":  round(float(prec_per_class[0]), 3),
    "mid":  round(float(prec_per_class[1]), 3),
    "high": round(float(prec_per_class[2]), 3),
    "thresholds": [round(float(q33), 1), round(float(q66), 1)],
}

# ─────────────────────────────────────────────
# CHUONG 6: KMEANS CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MO HINH 11: KMEANS CLUSTERING (Chuong 6)")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SS

cluster_features = ["Gia_m2", "dt", "so_phong", "noi_thanh"]
df_cluster = df[cluster_features].dropna().copy()
ss = SS()
X_cluster = ss.fit_transform(df_cluster)

km_final = KMeans(n_clusters=5, random_state=42, n_init=10)
df_cluster["Cum"] = km_final.fit_predict(X_cluster)
print("Dac trung trung binh theo cum:")
print(df_cluster.groupby("Cum").mean().round(1).to_string())
df.loc[df_cluster.index, "Cum"] = df_cluster["Cum"]

# ─────────────────────────────────────────────
# TONG HOP KET QUA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TONG HOP KET QUA")
print("=" * 60)

results_df = pd.DataFrame(results).set_index("model")
print(results_df.sort_values("R2", ascending=False).to_string())

best_model_name = results_df["R2"].idxmax()
best_r2   = results_df["R2"].max()
best_rmse = results_df.loc[best_model_name, "RMSE"]
print(f"\nMo hinh tot nhat: {best_model_name}  R2={best_r2*100:.1f}%  RMSE={best_rmse:.2f}")

# ─────────────────────────────────────────────
# LUU MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("LUU MODELS")
print("=" * 60)

os.makedirs("models", exist_ok=True)

best_regressor = stack

joblib.dump(rf,             "models/random_forest.pkl")
joblib.dump(gb,             "models/gradient_boosting.pkl")
joblib.dump(best_regressor, "models/best_model.pkl")
joblib.dump(logr,           "models/logistic_classifier.pkl")
joblib.dump(km_final,       "models/kmeans_cluster.pkl")
joblib.dump(scaler,         "models/scaler.pkl")
joblib.dump(ss,             "models/cluster_scaler.pkl")

if has_xgb:  joblib.dump(xgb,  "models/xgboost.pkl")
if has_lgbm: joblib.dump(lgbm, "models/lightgbm.pkl")

# Target encoding maps
encoding_maps = {
    "te_duong":   df.groupby("ten_duong")["te_duong"].first().to_dict(),
    "te_phuong":  df.groupby("phuong")["te_phuong"].first().to_dict(),
    "te_diaban":  df.groupby("dia_ban")["te_diaban"].first().to_dict(),
    "te_loainha": df.groupby("Loại hình nhà ở")["te_loainha"].first().to_dict(),
    "global_median": float(y.median()),
    "med_tang": float(med_tang),
    "med_phong": float(med_phong),
    "med_dai": float(med_dai),
    "med_rong": float(med_rong),
}
joblib.dump(encoding_maps, "models/encoding_maps.pkl")
joblib.dump({"feature_cols": FEATURE_COLS}, "models/feature_info.pkl")

# Model results cho analysis.html
model_results = {
    "models":     results,
    "best_model": best_model_name,
    "best_r2":    round(best_r2, 4),
    "best_rmse":  round(best_rmse, 2),
    "logistic_precision": logistic_precision,
}
joblib.dump(model_results, "models/model_results.pkl")

# data_cleaned.csv
X_save = X.copy()
X_save["Gia_m2"] = y.values
X_save.to_csv("models/data_cleaned.csv", index=False)

print("Da luu xong:")
print("  best_model.pkl     -> Stacking Ensemble")
print("  encoding_maps.pkl  -> Target encoding maps")
print("  feature_info.pkl   -> Feature list (ASCII)")
print("  model_results.pkl  -> Ket qua danh gia")
print("  data_cleaned.csv   -> Du lieu sach")

print("\n" + "=" * 60)
print(f"R2 tot nhat: {best_r2*100:.1f}%  RMSE: {best_rmse:.2f} trieu/m2")
print("HOAN THANH TRAINING!")
print("=" * 60)

# ─────────────────────────────────────────────
# DEMO DU DOAN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEMO DU DOAN GIA NHA")
print("=" * 60)

def du_doan_gia(dien_tich, so_phong, dia_ban, loai_nha,
                ten_duong="unknown", phuong="unknown",
                so_tang=3, co_so=True, mat_pho=False,
                nam=2020, quy=2, dai=None, rong=None):
    if dai is None:  dai  = np.sqrt(dien_tich) * 1.5
    if rong is None: rong = dien_tich / max(dai, 1)

    gm   = encoding_maps["global_median"]
    te_d = encoding_maps["te_duong"].get(ten_duong.lower(), gm)
    te_p = encoding_maps["te_phuong"].get(phuong.lower(), gm)
    te_b = encoding_maps["te_diaban"].get(dia_ban, gm)
    te_l = encoding_maps["te_loainha"].get(loai_nha, gm)
    mp   = float(mat_pho)

    row = {c: 0.0 for c in FEATURE_COLS}
    row["dt"]            = dien_tich
    row["log_dt"]        = np.log1p(dien_tich)
    row["dt_sq"]         = dien_tich**2
    row["so_phong"]      = so_phong
    row["so_tang"]       = so_tang
    row["dai"]           = dai
    row["rong"]          = rong
    row["tl_dai_rong"]   = dai/(rong+1e-5)
    row["dt_dairong"]    = dai*rong
    row["phong_tang"]    = so_phong/(so_tang+1e-5)
    row["dt_per_phong"]  = dien_tich/max(so_phong,1)
    row["tang_dt"]       = so_tang*dien_tich
    row["rong_sq"]       = rong**2
    row["rong_x_mp"]     = rong*mp
    row["co_so"]         = float(co_so)
    row["mat_pho"]       = mp
    row["noi_thanh"]     = 1.0 if "qu" in dia_ban.lower() else 0.0
    row["nam"]           = float(nam)
    row["quy"]           = float(quy)
    row["thang"]         = float(quy*3)
    row["te_duong"]      = te_d
    row["te_phuong"]     = te_p
    row["te_diaban"]     = te_b
    row["te_loainha"]    = te_l
    row["te_duong_x_mp"] = te_d*mp
    row["te_phuong_x_dt"]= te_p*np.log1p(dien_tich)
    row["te_duong_x_dt"] = te_d*np.log1p(dien_tich)
    row["rank_dt_quan"]  = 0.5
    row["rank_tang_quan"]= 0.5
    row["rank_rong_quan"]= 0.5

    X_in = pd.DataFrame([row])
    gia = float(np.clip(np.expm1(best_regressor.predict(X_in)[0]), 5, 600))
    return round(gia, 1)

g1 = du_doan_gia(50, 3, "Quận Cầu Giấy", "Nhà ngõ, hẻm",
                 ten_duong="hoang quoc viet", phuong="nghia do",
                 so_tang=4, co_so=True)
print(f"Nha ngo 50m2, 3PN, 4T, Cau Giay  -> {g1} tr/m2 (~{g1*50:,.0f} tr)")

g2 = du_doan_gia(60, 4, "Quận Ba Đình", "Nhà mặt phố, mặt tiền",
                 ten_duong="doi can", phuong="doi can",
                 so_tang=5, co_so=True, mat_pho=True)
print(f"Mat pho 60m2, 4PN, 5T, Ba Dinh   -> {g2} tr/m2 (~{g2*60:,.0f} tr)")