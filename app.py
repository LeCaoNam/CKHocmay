"""
app.py — Flask backend cho NhàGiá.AI
Chạy: python app.py
Truy cập: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ══════════════════════════════════════════════
# LOAD MODELS (chạy 1 lần khi khởi động server)
# ══════════════════════════════════════════════
MODELS_DIR = "models"

def load_models():
    """Load tất cả models đã train. Trả về dict hoặc None nếu chưa train."""
    required = [
        "random_forest.pkl",
        "gradient_boosting.pkl",
        "logistic_classifier.pkl",
        "kmeans_cluster.pkl",
        "scaler.pkl",
        "cluster_scaler.pkl",
        "data_cleaned.csv",
        "encoding_maps.pkl",   # ← BẮT BUỘC: target encoding maps
        "feature_info.pkl",    # ← BẮT BUỘC: danh sách feature names
        "model_results.pkl",   # ← BẮT BUỘC: kết quả đánh giá model
    ]
    for f in required:
        path = os.path.join(MODELS_DIR, f)
        if not os.path.exists(path):
            print(f"⚠️  Chưa tìm thấy: {path}")
            print("   → Chạy: python train_model.py trước!")
            return None

    models = {
        "rf":         joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
        "gb":         joblib.load(os.path.join(MODELS_DIR, "gradient_boosting.pkl")),
        "logr":       joblib.load(os.path.join(MODELS_DIR, "logistic_classifier.pkl")),
        "km":         joblib.load(os.path.join(MODELS_DIR, "kmeans_cluster.pkl")),
        "scaler":     joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "km_scaler":  joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.pkl")),
        "enc_maps":   joblib.load(os.path.join(MODELS_DIR, "encoding_maps.pkl")),
        "model_results": joblib.load(os.path.join(MODELS_DIR, "model_results.pkl")),
        "data":       pd.read_csv(os.path.join(MODELS_DIR, "data_cleaned.csv")),
    }

    # FIX 1: Load feature_cols từ feature_info.pkl (đúng tên train_model dùng)
    feature_info = joblib.load(os.path.join(MODELS_DIR, "feature_info.pkl"))
    models["feature_cols"] = feature_info["feature_cols"]

    print("✅ Đã load xong tất cả models!")
    print(f"   Features: {len(models['feature_cols'])} | Data rows: {len(models['data'])}")
    return models

MODELS = load_models()


# ══════════════════════════════════════════════
# HELPER — xây dựng input vector từ form data
# ══════════════════════════════════════════════
def build_input_vector(data: dict, feature_cols: list, enc_maps: dict) -> pd.DataFrame:
    """
    Chuyển dict từ form → DataFrame 1 dòng đúng format cho model.
    FIX 2: Dùng đúng tên feature của train_model.py (dt, so_phong, te_duong, ...)
    và tính target encoding từ encoding_maps.pkl
    """
    dt      = float(data.get("dientich", 50))
    sophong = float(data.get("sophong", 3))
    sotang  = float(data.get("sotang", 4))
    co_so   = float(data.get("phaplý", 1))
    quan    = data.get("quan", "")
    loainha = data.get("loainha", "")

    # Dài / Rộng — tự tính nếu để trống
    dai  = float(data.get("dai") or 0) or np.sqrt(dt) * 1.5
    rong = float(data.get("rong") or 0) or (dt / max(dai, 1))

    # Tên đường / Phường từ form (predict.html gửi ten_duong, phuong)
    ten_duong = str(data.get("ten_duong", "unknown")).strip().lower() or "unknown"
    phuong    = str(data.get("phuong", "unknown")).strip().lower() or "unknown"

    # Loại nhà flags
    mp        = 1.0 if "mặt phố" in loainha.lower() or "mat pho" in loainha.lower() else 0.0
    biet_thu  = 1.0 if "biệt thự" in loainha.lower() or "biet thu" in loainha.lower() else 0.0
    lien_ke   = 1.0 if "liền kề" in loainha.lower() or "lien ke" in loainha.lower() else 0.0
    noi_thanh = 1.0 if "quận" in quan.lower() else 0.0

    # Target encoding — tra bảng map, fallback về median
    gm    = enc_maps.get("global_median", 90.0)
    te_d  = enc_maps["te_duong"].get(ten_duong, gm)
    te_p  = enc_maps["te_phuong"].get(phuong, gm)
    te_b  = enc_maps["te_diaban"].get(quan, gm)
    te_l  = enc_maps["te_loainha"].get(loainha, gm)

    # Khởi tạo dict với tất cả features = 0
    row = {c: 0.0 for c in feature_cols}

    # Điền đúng tên feature theo train_model.py
    row["dt"]             = dt
    row["log_dt"]         = np.log1p(dt)
    row["dt_sq"]          = dt ** 2
    row["so_phong"]       = sophong
    row["so_tang"]        = sotang
    row["dai"]            = dai
    row["rong"]           = rong
    row["tl_dai_rong"]    = dai / (rong + 1e-5)
    row["dt_dairong"]     = dai * rong
    row["phong_tang"]     = sophong / (sotang + 1e-5)
    row["dt_per_phong"]   = dt / max(sophong, 1)
    row["tang_dt"]        = sotang * dt
    row["rong_sq"]        = rong ** 2
    row["rong_x_mp"]      = rong * mp
    row["co_so"]          = co_so
    row["noi_thanh"]      = noi_thanh
    row["mat_pho"]        = mp
    row["biet_thu"]       = biet_thu
    row["lien_ke"]        = lien_ke
    row["co_ngo"]         = 0.0
    row["ngo_sau"]        = 0.0
    row["so_ngo"]         = 0.0
    row["nam"]            = 2020.0
    row["quy"]            = 2.0
    row["thang"]          = 6.0
    row["te_duong"]       = te_d
    row["te_phuong"]      = te_p
    row["te_diaban"]      = te_b
    row["te_loainha"]     = te_l
    row["te_duong_x_mp"]  = te_d * mp
    row["te_phuong_x_dt"] = te_p * np.log1p(dt)
    row["te_duong_x_dt"]  = te_d * np.log1p(dt)
    row["te_diaban_x_bt"] = te_b * biet_thu
    row["rank_dt_quan"]   = 0.5   # mid-rank khi không biết
    row["rank_tang_quan"] = 0.5
    row["rank_rong_quan"] = 0.5

    return pd.DataFrame([row])


def classify_segment(price: float) -> dict:
    """Phân loại phân khúc giá."""
    if price <= 78.8:
        return {"label": "Bình dân",  "class": "low",  "range": "≤ 79 tr/m²"}
    elif price <= 100:
        return {"label": "Trung cấp", "class": "mid",  "range": "79–100 tr/m²"}
    else:
        return {"label": "Cao cấp",   "class": "high", "range": "> 100 tr/m²"}


# ══════════════════════════════════════════════
# ROUTES — CÁC TRANG HTML
# ══════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/suggest")
def suggest_page():
    return render_template("suggest.html")

@app.route("/compare")
def compare_page():
    return render_template("compare.html")

@app.route("/calculator")
def calculator_page():
    return render_template("calculator.html")

@app.route("/heatmap")
def heatmap_page():
    return render_template("heatmap.html")


# ══════════════════════════════════════════════
# API — ĐỊNH GIÁ NHÀ
# POST /api/predict
# ══════════════════════════════════════════════
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODELS is None:
        return jsonify({"error": "Models chưa được train. Chạy train_model.py trước!"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Không có dữ liệu đầu vào"}), 400

    try:
        dt = float(data.get("dientich", 50))

        # FIX 2: build_input_vector dùng đúng feature names + encoding maps
        X_input = build_input_vector(data, MODELS["feature_cols"], MODELS["enc_maps"])

        # Dự đoán bằng Random Forest (log-space, cần expm1)
        price_rf = float(np.expm1(MODELS["rf"].predict(X_input)[0]))

        # Dự đoán bằng Gradient Boosting (log-space, cần expm1)
        price_gb = float(np.expm1(MODELS["gb"].predict(X_input)[0]))

        # Trung bình 2 model
        price = round((price_rf * 0.5 + price_gb * 0.5), 1)
        price = max(10, min(price, 500))

        segment = classify_segment(price)

        # Phân loại bằng Logistic Regression (cần scaled input)
        X_scaled = MODELS["scaler"].transform(X_input)
        label_logr = int(MODELS["logr"].predict(X_scaled)[0])
        label_map  = {0: "Bình dân", 1: "Trung cấp", 2: "Cao cấp"}

        # Lấy RMSE từ model_results
        mr   = MODELS["model_results"]
        rmse = mr.get("best_rmse", 39.2)
        r2   = mr.get("best_r2", 0.42)

        return jsonify({
            "price":      price,
            "price_rf":   round(price_rf, 1),
            "price_gb":   round(price_gb, 1),
            "total":      round(price * dt, 0),
            "segment":    segment,
            "logr_label": label_map.get(label_logr, "—"),
            "model_used": "Random Forest + Gradient Boosting",
            "rmse":       rmse,
            "r2":         r2,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════
# API — GỢI Ý NHÀ TƯƠNG TỰ
# POST /api/suggest
# ══════════════════════════════════════════════
@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    if MODELS is None:
        return jsonify({"error": "Models chưa được train. Chạy train_model.py trước!"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Không có dữ liệu đầu vào"}), 400

    try:
        max_price    = float(data.get("maxPrice", 120))
        desired_area = float(data.get("desiredArea", 50))
        min_phong    = int(data.get("minPhong", 2))
        noi_thanh    = int(data.get("noiThanh", 1))

        # Xây dựng vector để predict cluster
        cluster_input = np.array([[
            max_price,
            desired_area,
            min_phong,
            noi_thanh if noi_thanh >= 0 else 1,
        ]])
        cluster_input_scaled = MODELS["km_scaler"].transform(cluster_input)
        cluster_id = int(MODELS["km"].predict(cluster_input_scaled)[0])

        cluster_names = ["Bình dân", "Trung cấp", "Phân khúc khá", "Cao cấp", "Sang trọng"]

        df = MODELS["data"].copy()

        # FIX 3: Dùng đúng tên cột của data_cleaned.csv (tên viết tắt từ train_model.py)
        # train_model lưu: Gia_m2, dt, so_phong, noi_thanh
        cluster_feature_cols = ["Gia_m2", "dt", "so_phong", "noi_thanh"]
        available = [c for c in cluster_feature_cols if c in df.columns]

        if len(available) == 4:
            X_all = MODELS["km_scaler"].transform(df[available].fillna(0))
            df["_cluster"] = MODELS["km"].predict(X_all)
        else:
            df["_cluster"] = cluster_id

        # Lọc theo cluster
        df_filtered = df[df["_cluster"] == cluster_id].copy()

        # FIX 3 (tiếp): Tên cột đúng trong data_cleaned.csv
        if "Gia_m2" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Gia_m2"] <= max_price]
        if "so_phong" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["so_phong"] >= min_phong]
        if noi_thanh >= 0 and "noi_thanh" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["noi_thanh"] == noi_thanh]

        # Mở rộng nếu quá ít
        if len(df_filtered) < 3:
            df_filtered = df[df["_cluster"] == cluster_id].head(10)

        # Lấy 9 kết quả gần ngân sách nhất
        if "Gia_m2" in df_filtered.columns:
            df_filtered["_price_diff"] = abs(df_filtered["Gia_m2"] - max_price * 0.8)
            df_result = df_filtered.nsmallest(9, "_price_diff")
        else:
            df_result = df_filtered.head(9)

        # Format kết quả
        houses = []
        for _, row in df_result.iterrows():
            # FIX 3 (tiếp): đọc đúng tên cột
            gia   = round(float(row.get("Gia_m2", 0)), 1)
            dt    = round(float(row.get("dt", 0)), 0)
            phong = int(row.get("so_phong", 0))
            tang  = int(row.get("so_tang", 0)) if "so_tang" in row else 0

            # Địa bàn — train_model dùng one-hot "Địa bàn_XXX" hoặc cột te_diaban
            addr = "Hà Nội"
            for col in df_result.columns:
                if col.startswith("Địa bàn_") and row.get(col, 0) == 1:
                    addr = col.replace("Địa bàn_", "")
                    break
            # Nếu không có cột one-hot, thử đọc giá trị te_diaban hoặc dia_ban
            if addr == "Hà Nội" and "dia_ban" in row:
                addr = str(row["dia_ban"])

            # Loại nhà
            loai = "Nhà ở"
            if row.get("mat_pho", 0) == 1:
                loai = "Nhà mặt phố"
            elif row.get("biet_thu", 0) == 1:
                loai = "Nhà biệt thự"
            elif row.get("lien_ke", 0) == 1:
                loai = "Nhà phố liền kề"
            else:
                loai = "Nhà ngõ, hẻm"

            houses.append({
                "gia":   gia,
                "total": round(gia * dt, 0),
                "dt":    int(dt),
                "phong": phong,
                "tang":  tang,
                "addr":  addr,
                "loai":  loai,
            })

        return jsonify({
            "cluster_id":   cluster_id,
            "cluster_name": cluster_names[cluster_id] if cluster_id < len(cluster_names) else "Khác",
            "total_found":  len(houses),
            "houses":       houses,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════
# API — THỐNG KÊ MODEL (FIX 1: Route mới)
# GET /api/model-stats
# Dùng bởi: analysis.html, index.html
# ══════════════════════════════════════════════
@app.route("/api/model-stats")
def api_model_stats():
    if MODELS is None:
        return jsonify({"error": "Models chưa được train. Chạy train_model.py trước!"}), 503

    try:
        mr = MODELS["model_results"]
        df = MODELS["data"]

        # Tính giá trung bình từ data thực
        avg_price = float(df["Gia_m2"].mean()) if "Gia_m2" in df.columns else 93.0

        # Kiểm tra xem có Stacking không
        models_list = mr.get("models", [])
        using_stacking = any(
            "Stacking" in (m.get("model", "") or m.get("name", ""))
            for m in models_list
        )

        return jsonify({
            "models":            models_list,
            "best_model":        mr.get("best_model", "—"),
            "best_r2":           mr.get("best_r2", 0.42),
            "best_rmse":         mr.get("best_rmse", 39.2),
            "logistic_precision": mr.get("logistic_precision", {}),
            "avg_price":         round(avg_price, 1),
            "total_rows":        len(df),
            "feature_cols":      MODELS["feature_cols"],
            "using_stacking":    using_stacking,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════
# API — HEALTH CHECK
# GET /api/health
# ══════════════════════════════════════════════
@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok" if MODELS else "models_not_loaded",
        "models_ready": MODELS is not None,
        "endpoints":    ["/api/predict", "/api/suggest", "/api/model-stats"],
    })


# ══════════════════════════════════════════════
# CHẠY SERVER
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  NhàGiá.AI — Flask Server")
    print("="*50)
    print("  URL: http://127.0.0.1:5000")
    print("  API: http://127.0.0.1:5000/api/health")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)