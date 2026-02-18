"""
Kampüs Enerji Optimizasyon Sistemi — Makine Öğrenmesi Modülü
==============================================================
Bina bazlı enerji tüketim tahmini için RandomForestRegressor modeli.

Özellikler (Features):
  • Zamansal: hour, day_of_week, month, is_weekend
  • Çevresel: temperature, humidity, solar_radiation, temperature_sq
  • Lag: prev_1h_kwh, prev_2h_kwh, prev_24h_kwh
  • Rolling: rolling_3h_mean, rolling_24h_mean
  • Etkileşim: occupancy_temp_interaction (occupancy_rate × temperature)
  • Bina: occupancy_rate, building_type (one-hot)

Model çıktısı: total_kwh (sonraki saat tahmini)

Kullanım:
    from modules.ml_model import train_model, predict_next_24h, get_model_accuracy

    metrics = train_model(building_id=1)
    predictions = predict_next_24h(building_id=1)
    accuracy = get_model_accuracy(building_id=1)
"""

import json
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from modules.database import Building, EnergyReading, Prediction, WeatherData, db

# ---------------------------------------------------------------------------
# SABİTLER
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

BUILDING_TYPES = ["classroom", "laboratory", "office", "social"]
TRAIN_DAYS = 30          # Son kaç günlük veri kullanılacak
TRAIN_RATIO = 0.80       # %80 eğitim, %20 test

MODEL_VERSION = "rf_v2"  # Model sürümü etiketi

# Sabit random seed
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------------------


def _ensure_models_dir():
    """models/ klasörünün var olduğundan emin ol."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def _model_path(building_id: int) -> str:
    """Bina bazlı model dosya yolu."""
    return os.path.join(MODELS_DIR, f"rf_model_{building_id}.pkl")


def _load_metrics() -> dict:
    """Kaydedilmiş metrikleri yükle."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metrics(all_metrics: dict):
    """Metrikleri diske kaydet."""
    _ensure_models_dir()
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)


def _build_dataframe(building_id: int, days: int = TRAIN_DAYS) -> pd.DataFrame:
    """Veritabanından enerji + hava durumu verilerini birleştirip
    feature-engineered DataFrame döner.

    Lag ve rolling feature'lar hesaplanırken veri sızıntısı (leakage)
    önlenir: sadece geçmişteki değerler kullanılır (shift >= 1).

    Args:
        building_id: Hedef bina ID'si.
        days: Geriye dönük kaç günlük veri çekilecek.

    Returns:
        pd.DataFrame: Eğitim/tahmin için hazır DataFrame.
    """
    building = Building.query.get(building_id)
    if building is None:
        raise ValueError(f"Bina bulunamadı: id={building_id}")

    # Lag/rolling için 1 gün ekstra veri çek (24h lag dolsun diye)
    # Not: SQLite naive datetime saklar — tz-naive cutoff kullanıyoruz
    now_utc = datetime.utcnow()
    extended_cutoff = now_utc - timedelta(days=days + 1)
    actual_cutoff = now_utc - timedelta(days=days)

    # --- Enerji okumaları ---
    energy_rows = (
        EnergyReading.query
        .filter(
            EnergyReading.building_id == building_id,
            EnergyReading.timestamp >= extended_cutoff,
        )
        .order_by(EnergyReading.timestamp)
        .all()
    )

    if len(energy_rows) < 48:  # en az 2 günlük veri
        raise ValueError(
            f"Bina {building_id} için yeterli veri yok "
            f"(bulunan: {len(energy_rows)}, gereken: >=48)."
        )

    energy_df = pd.DataFrame([r.to_dict() for r in energy_rows])
    energy_df["timestamp"] = pd.to_datetime(energy_df["timestamp"])

    # --- Hava durumu ---
    weather_rows = (
        WeatherData.query
        .filter(WeatherData.timestamp >= extended_cutoff)
        .order_by(WeatherData.timestamp)
        .all()
    )
    weather_df = pd.DataFrame([w.to_dict() for w in weather_rows])
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    # --- Birleştir (en yakın saate göre) ---
    df = pd.merge_asof(
        energy_df.sort_values("timestamp"),
        weather_df[["timestamp", "temperature", "humidity", "solar_radiation"]].sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    # NaN hava durumu satırlarını doldur
    df["temperature"] = df["temperature"].fillna(df["temperature"].median())
    df["humidity"] = df["humidity"].fillna(df["humidity"].median())
    df["solar_radiation"] = df["solar_radiation"].fillna(0.0)

    # --- Temel zamansal feature'lar ---
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Lag features (veri sızıntısı yok — sadece geçmiş) ---
    df["prev_1h_kwh"] = df["total_kwh"].shift(1)
    df["prev_2h_kwh"] = df["total_kwh"].shift(2)
    df["prev_24h_kwh"] = df["total_kwh"].shift(24)

    # --- Rolling mean features (geçmiş pencere, mevcut saat dahil değil) ---
    df["rolling_3h_mean"] = (
        df["total_kwh"].shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["rolling_24h_mean"] = (
        df["total_kwh"].shift(1).rolling(window=24, min_periods=1).mean()
    )

    # --- Polinom / etkileşim feature'ları ---
    df["temperature_sq"] = df["temperature"] ** 2
    df["occupancy_temp_interaction"] = df["occupancy_rate"] * df["temperature"]

    # --- One-hot encode building type ---
    btype = building.type
    for t in BUILDING_TYPES:
        df[f"type_{t}"] = 1 if btype == t else 0

    # Ekstra çekilen 1 günlük veriyi kes + NaN satırları temizle
    df = df[df["timestamp"] >= actual_cutoff].copy()
    df = df.dropna(subset=["prev_1h_kwh", "prev_2h_kwh", "prev_24h_kwh"]).reset_index(drop=True)

    return df


def _get_feature_columns() -> list:
    """Model feature sütun adları (sıralı)."""
    return [
        # Zamansal
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        # Çevresel
        "temperature",
        "humidity",
        "solar_radiation",
        # Doluluk
        "occupancy_rate",
        # Lag
        "prev_1h_kwh",
        "prev_2h_kwh",
        "prev_24h_kwh",
        # Rolling
        "rolling_3h_mean",
        "rolling_24h_mean",
        # Polinom / etkileşim
        "temperature_sq",
        "occupancy_temp_interaction",
    ] + [f"type_{t}" for t in BUILDING_TYPES]


TARGET_COL = "total_kwh"


# ---------------------------------------------------------------------------
# ANA FONKSİYONLAR
# ---------------------------------------------------------------------------


def train_model(building_id: int, force_retrain: bool = False) -> dict:
    """Belirtilen bina için RandomForest modeli eğitir.

    Zaman serisi sırasını koruyarak son %80 eğitim, son %20 test olarak
    ayırır. Modeli diske kaydeder.

    Args:
        building_id: Hedef bina ID'si.
        force_retrain: True ise mevcut model varsa bile yeniden eğitir.

    Returns:
        dict: Eğitim metrikleri (MAE, RMSE, R²) ve ek bilgiler.
    """
    # Mevcut model varsa ve zorlama yoksa atla
    model_file = _model_path(building_id)
    if not force_retrain and os.path.exists(model_file):
        existing = _load_metrics()
        key = str(building_id)
        if key in existing:
            return existing[key]

    _ensure_models_dir()

    # Veriyi hazırla
    df = _build_dataframe(building_id, days=TRAIN_DAYS)
    features = _get_feature_columns()

    X = df[features].values
    y = df[TARGET_COL].values

    # Zaman serisi split — son %20 test
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Model oluştur ve eğit
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Tahmin ve metrikler
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "building_id": building_id,
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestRegressor",
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }

    # Modeli diske kaydet
    joblib.dump(model, model_file)

    # Metrikleri güncelle
    all_metrics = _load_metrics()
    all_metrics[str(building_id)] = metrics
    _save_metrics(all_metrics)

    return metrics


# ---------------------------------------------------------------------------
# TAHMİN YARDIMCISI
# ---------------------------------------------------------------------------


def _build_prediction_features(
    building: Building,
    hourly_stats: dict,
    kwh_buffer: list,
    target_ts: datetime,
) -> np.ndarray:
    """Tahmin için tek satırlık feature vektörü oluşturur.

    kwh_buffer: son bilinen kWh değerlerinin listesi (en yenisi sonda).
    Lag ve rolling hesaplamaları bu buffer üzerinden yapılır.
    """
    hour = target_ts.hour
    dow = target_ts.weekday()
    month = target_ts.month
    is_weekend = 1 if dow >= 5 else 0

    stats = hourly_stats.get(hour, {})
    occ = stats.get("occupancy_rate", 0.3)
    temp = stats.get("temperature", 15.0)
    hum = stats.get("humidity", 55.0)
    solar = stats.get("solar_radiation", 0.0)

    # Lag features
    prev_1h = kwh_buffer[-1] if len(kwh_buffer) >= 1 else 0.0
    prev_2h = kwh_buffer[-2] if len(kwh_buffer) >= 2 else prev_1h
    prev_24h = kwh_buffer[-24] if len(kwh_buffer) >= 24 else prev_1h

    # Rolling mean features
    last_3 = kwh_buffer[-3:] if len(kwh_buffer) >= 3 else kwh_buffer
    rolling_3h = float(np.mean(last_3)) if last_3 else 0.0
    last_24 = kwh_buffer[-24:] if len(kwh_buffer) >= 24 else kwh_buffer
    rolling_24h = float(np.mean(last_24)) if last_24 else 0.0

    # Polinom / etkileşim
    temp_sq = temp ** 2
    occ_temp = occ * temp

    # One-hot building type
    type_features = [1 if building.type == t else 0 for t in BUILDING_TYPES]

    row = [
        hour, dow, month, is_weekend,
        temp, hum, solar, occ,
        prev_1h, prev_2h, prev_24h,
        rolling_3h, rolling_24h,
        temp_sq, occ_temp,
    ] + type_features

    return np.array([row])


# ---------------------------------------------------------------------------
# TAHMİN FONKSİYONLARI
# ---------------------------------------------------------------------------


def predict_next_24h(building_id: int) -> list:
    """Sonraki 24 saat için enerji tüketimi tahmini üretir.

    Lag/rolling feature'lar için son 24 saatlik gerçek okumalar
    başlangıç buffer'ı olarak kullanılır; sonraki saatler kendi
    tahminleriyle buffer'ı günceller (auto-regressive).

    Args:
        building_id: Hedef bina ID'si.

    Returns:
        list[dict]: Her saat için {timestamp, predicted_kwh, confidence_lower,
                    confidence_upper} sözlüklerinden oluşan liste.
    """
    model_file = _model_path(building_id)

    # Model yoksa eğit
    if not os.path.exists(model_file):
        train_model(building_id)

    model: RandomForestRegressor = joblib.load(model_file)
    building = Building.query.get(building_id)

    if building is None:
        raise ValueError(f"Bina bulunamadı: id={building_id}")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Son 7 günün saatlik ortalamaları
    historical_df = _build_dataframe(building_id, days=7)
    hourly_stats = historical_df.groupby("hour").agg({
        "occupancy_rate": "mean",
        "temperature": "mean",
        "humidity": "mean",
        "solar_radiation": "mean",
    }).to_dict("index")

    # Lag/rolling buffer — son 24 saatlik gerçek okumalar
    kwh_buffer = historical_df["total_kwh"].tail(24).tolist()

    predictions = []

    for h in range(1, 25):
        target_ts = now + timedelta(hours=h)

        X_pred = _build_prediction_features(
            building, hourly_stats, kwh_buffer, target_ts
        )

        predicted_kwh = float(model.predict(X_pred)[0])

        # Güven aralığı — bireysel ağaç tahminlerinden
        tree_preds = np.array([
            tree.predict(X_pred)[0] for tree in model.estimators_
        ])
        std = float(tree_preds.std())
        confidence_lower = round(max(0.0, predicted_kwh - 1.96 * std), 2)
        confidence_upper = round(predicted_kwh + 1.96 * std, 2)

        prediction_record = {
            "timestamp": target_ts.isoformat(),
            "predicted_kwh": round(predicted_kwh, 2),
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
        }
        predictions.append(prediction_record)

        # Buffer'ı güncelle (auto-regressive)
        kwh_buffer.append(predicted_kwh)

        # Veritabanına da kaydet
        db_pred = Prediction(
            building_id=building_id,
            prediction_timestamp=now,
            target_timestamp=target_ts,
            predicted_kwh=round(predicted_kwh, 2),
            model_version=MODEL_VERSION,
            accuracy_score=None,
        )
        db.session.add(db_pred)

    db.session.commit()
    return predictions


def predict_next_7d(building_id: int) -> list:
    """Sonraki 7 gün için günlük toplam enerji tahmini.

    168 saat (7 x 24) auto-regressive tahmin yaparak günlük
    toplamlara dönüştürür.

    Args:
        building_id: Hedef bina ID'si.

    Returns:
    """
    # -----------------------------------------------------------------------
    # CACHING LOGIC (Simple In-Memory)
    # -----------------------------------------------------------------------
    global _FORECAST_CACHE
    if "_FORECAST_CACHE" not in globals():
        _FORECAST_CACHE = {}

    cache_entry = _FORECAST_CACHE.get(building_id)
    now_ts = datetime.now().timestamp()
    
    # 1 saat (3600 sn) cache süresi
    if cache_entry:
        timestamp, data = cache_entry
        if now_ts - timestamp < 3600:
            return data

    model_file = _model_path(building_id)
    if not os.path.exists(model_file):
        train_model(building_id)

    model: RandomForestRegressor = joblib.load(model_file)
    building = Building.query.get(building_id)

    if building is None:
        raise ValueError(f"Bina bulunamadı: id={building_id}")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    historical_df = _build_dataframe(building_id, days=7)
    hourly_stats = historical_df.groupby("hour").agg({
        "occupancy_rate": "mean",
        "temperature": "mean",
        "humidity": "mean",
        "solar_radiation": "mean",
    }).to_dict("index")

    # Lag/rolling buffer
    kwh_buffer = historical_df["total_kwh"].tail(24).tolist()

    daily_totals = {}

    for h in range(1, 169):  # 7 gün x 24 saat
        target_ts = now + timedelta(hours=h)

        X_pred = _build_prediction_features(
            building, hourly_stats, kwh_buffer, target_ts
        )

        predicted = float(model.predict(X_pred)[0])
        kwh_buffer.append(predicted)

        date_key = target_ts.strftime("%Y-%m-%d")
        daily_totals[date_key] = daily_totals.get(date_key, 0.0) + predicted

    result = [
        {"date": date, "predicted_kwh_total": round(total, 2)}
        for date, total in daily_totals.items()
    ]

    # Cache güncelle
    _FORECAST_CACHE[building_id] = (now_ts, result)
    return result


# ---------------------------------------------------------------------------
# DOĞRULUK & ANOMALİ
# ---------------------------------------------------------------------------


def get_model_accuracy(building_id: int) -> dict:
    """Eğitilmiş modelin performans metriklerini döner.

    Model eğitilmemişse önce eğitir.

    Args:
        building_id: Hedef bina ID'si.

    Returns:
        dict: MAE, RMSE, R2, eğitim zamanı ve model sürümü.
    """
    all_metrics = _load_metrics()
    key = str(building_id)

    if key in all_metrics:
        return all_metrics[key]

    # Model henüz eğitilmemiş — eğit ve döndür
    return train_model(building_id)


def get_feature_importance(building_id: int) -> list:
    """Eğitilmiş modelin feature importance değerlerini döner.

    Args:
        building_id: Hedef bina ID'si.

    Returns:
        list[dict]: Feature adı ve importance değeri, büyükten küçüğe sıralı.
    """
    model_path = _model_path(building_id)
    if not os.path.exists(model_path):
        train_model(building_id)

    model = joblib.load(model_path)
    features = _get_feature_columns()
    importances = model.feature_importances_

    result = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in zip(features, importances)
    ]
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result


def get_grouped_feature_importance(building_id: int) -> dict:
    """Feature importance değerlerini gruplara ayırarak yüzde katkı döner.

    Gruplar:
      - occupancy:   occupancy_rate, occupancy_temp_interaction
      - temperature: temperature, temperature_sq
      - lag:         prev_1h_kwh, prev_2h_kwh, prev_24h_kwh,
                     rolling_3h_mean, rolling_24h_mean
      - other:       humidity, solar_radiation + geri kalan her şey

    Args:
        building_id: Hedef bina ID'si.

    Returns:
        dict: {group_name: yüzde_katkı}
    """
    features = get_feature_importance(building_id)
    total = sum(f["importance"] for f in features) or 1.0

    GROUPS = {
        "occupancy":    {"occupancy_rate", "occupancy_temp_interaction"},
        "temperature":  {"temperature", "temperature_sq"},
        "lag":          {"prev_1h_kwh", "prev_2h_kwh", "prev_24h_kwh",
                         "rolling_3h_mean", "rolling_24h_mean"},
    }

    sums = {"occupancy": 0.0, "temperature": 0.0, "lag": 0.0, "other": 0.0}

    for f in features:
        placed = False
        for group, members in GROUPS.items():
            if f["feature"] in members:
                sums[group] += f["importance"]
                placed = True
                break
        if not placed:
            sums["other"] += f["importance"]

    return {k: round(v / total * 100, 1) for k, v in sums.items()}


def detect_anomalies(building_id: int, threshold: float = 2.5) -> list:
    """Z-score tabanlı anomali tespiti.

    Son 30 günlük veride ortalamadan threshold x standart sapma
    kadar sapan okumaları anomali olarak işaretler.

    Args:
        building_id: Hedef bina ID'si.
        threshold: Z-score eşiği (varsayılan 2.5).

    Returns:
        list[dict]: Anomali tespit edilen kayıtlar.
    """
    df = _build_dataframe(building_id, days=TRAIN_DAYS)
    return _detect_anomalies_in_df(df, threshold)


def _detect_anomalies_in_df(df: pd.DataFrame, threshold: float = 2.5) -> list:
    """Verilen DataFrame üzerinde Z-score tabanlı anomali tespiti."""
    mean = df[TARGET_COL].mean()
    std = df[TARGET_COL].std()

    if std == 0:
        return []

    # Orijinal DataFrame üzerinde işlem yapmamak için kopya alıyoruz
    # Ancak burada df zaten kopya gibi geliyor, yine de güvenli tarafta kalalım
    df = df.copy()

    df["z_score"] = (df[TARGET_COL] - mean) / std
    
    # Anomali indeksi ve detaylarını eşle
    anomalies = []
    
    # Sadece threshold'u aşanlar
    anomaly_rows = df[df["z_score"].abs() > threshold]

    for idx, row in anomaly_rows.iterrows():
        anomalies.append({
            "index": idx, # Orijinal dataframe indeksi (performans testi için önemli)
            "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            "total_kwh": round(row[TARGET_COL], 2),
            "expected_kwh": round(mean, 2),
            "z_score": round(row["z_score"], 2),
            "deviation_percent": round(abs(row[TARGET_COL] - mean) / mean * 100, 1),
        })

    return anomalies


def get_prediction_analysis(building_id: int, hours: int = 24) -> list:
    """Gerçek vs Tahmin (Hindcast) analizi yapar.
    
    Son 'hours' saatlik veriyi çeker, her saat için modelin o anki
    verilerle ne tahmin edeceğini hesaplar ve gerçek değerle kıyaslar.
    
    Args:
        building_id: Hedef bina ID.
        hours: Analiz edilecek saat sayısı.
        
    Returns:
        list[dict]: {timestamp, actual, predicted, deviation_pct}
    """
    model_file = _model_path(building_id)
    if not os.path.exists(model_file):
        train_model(building_id)

    model: RandomForestRegressor = joblib.load(model_file)
    
    # Veri çek (yeterince geçmişe git)
    df = _build_dataframe(building_id, days=3) 
    
    # Son 'hours' kadar veriyi al
    df_recent = df.tail(hours).copy()
    
    if df_recent.empty:
        return []

    features = _get_feature_columns()
    X = df_recent[features].values
    y_true = df_recent[TARGET_COL].values
    
    # Tahmin yap (hindcast)
    y_pred = model.predict(X)
    
    results = []
    for i in range(len(df_recent)):
        actual = float(y_true[i])
        predicted = float(y_pred[i])
        ts = df_recent.iloc[i]["timestamp"]
        
        # Sapma yüzdesi
        dev = 0.0
        if actual > 0:
            dev = abs(actual - predicted) / actual * 100
            
        results.append({
            "timestamp": ts.isoformat(),
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "deviation_pct": round(dev, 1)
        })
        
    return results


def train_all_models() -> dict:
    """Tüm binalar için modelleri eğitir.

    Returns:
        dict: {building_id: metrics} şeklinde sonuçlar.
    """
    buildings = Building.query.all()
    results = {}

    for b in buildings:
        try:
            metrics = train_model(b.id, force_retrain=True)
            results[b.id] = metrics
            print(f"   ✅ {b.name}: R2={metrics['r2']:.4f}, "
                  f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        except ValueError as e:
            results[b.id] = {"error": str(e)}
            print(f"   ⚠️  {b.name}: {e}")

    return results


def run_time_series_cv(building_id: int):
    """Bina için Time Series Cross Validation çalıştırır ve raporlar (5 Fold).

    Amaç: Modelin zamansal stabilitesini ve aşırı öğrenme (overfitting) riskini ölçmek.

    Args:
        building_id: Hedef bina ID.
    """
    print(f"\nTime Series CV Analysis for Building ID: {building_id}")
    print("=" * 60)

    # Veriyi hazırla
    df = _build_dataframe(building_id, days=TRAIN_DAYS)
    if len(df) < 100:
        print("Yetersiz veri (min 100 satır gerekli).")
        return

    features = _get_feature_columns()
    X = df[features].values
    y = df[TARGET_COL].values

    # TimeSeriesSplit (5 Fold)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Sonuçları sakla
    results = []

    print(f"{'Fold':<4} | {'Train R2':<8} | {'Test R2':<8} | {'MAE':<6} | {'RMSE':<6}")
    print("-" * 45)

    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Her split için model eğit
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Train Metrics
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)

        # Test Metrics
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        print(f"{fold:<4} | {train_r2:<8.4f} | {test_r2:<8.4f} | {mae:<6.2f} | {rmse:<6.2f}")

        results.append({
            "fold": fold,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mae": mae,
            "rmse": rmse
        })
        fold += 1

    print("-" * 45)
    
    # Ortalama Hesapla
    avg_test_r2 = np.mean([r['test_r2'] for r in results])
    avg_train_r2 = np.mean([r['train_r2'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])


def run_anomaly_performance_test(days: int = 30) -> list:
    """Anomali tespiti algoritması için performans testi (Sentetik Anomaliler).

    Son 30 günlük veriye rastgele 20 adet yapay anomali (normal + %30) ekler
    ve algoritmanın bunları ne kadar başarılı tespit ettiğini ölçer.

    Args:
        days: Test verisi süresi.

    Returns:
        list[dict]: Bina bazlı performans metrikleri (TP, FP, FN, Precision, Recall, F1).
    """
    buildings = Building.query.all()
    results = []

    for building in buildings:
        try:
            # Gerçek veriyi çek
            df = _build_dataframe(building.id, days=days)
            if len(df) < 50:
                print(f"Bina {building.name}: Yetersiz veri ({len(df)}), atlanıyor.")
                continue
            
            # --- Veri Manipülasyonu ---
            df_test = df.copy()
            
            # Rastgele 20 indeks seç (mevcut uzunluk içinde)
            import random
            available_indices = list(df_test.index)
            if len(available_indices) < 20:
                target_indices = available_indices
            else:
                target_indices = random.sample(available_indices, 20)
            
            injected_anomalies = set(target_indices)

            # Seçilen noktalara %30 anomali ekle
            for idx in target_indices:
                original_val = df_test.at[idx, TARGET_COL]
                # Pozitif yönlü sapma (+%30)
                # Eğer değer 0 veya çok küçükse, base bir değer ekle
                if original_val < 0.1:
                     df_test.at[idx, TARGET_COL] = 10.0 # yapay bir değer
                else:
                     df_test.at[idx, TARGET_COL] = original_val * 1.30

            # --- Tespit Algoritmasını Çalıştır ---
            # Threshold varsayılan 2.5
            detected_list = _detect_anomalies_in_df(df_test, threshold=2.5)
            detected_indices = {d["index"] for d in detected_list}

            # --- Orijinal (Doğal) Anomalileri Filtrele (Opsiyonel ama mantıklı) ---
            # Testin amacı SADECE eklenenleri bulmak mı, yoksa genel anomali yeteneği mi?
            # Görev: "True Positive", "False Positive". 
            # TP = Injected AND Detected
            # FP = NOT Injected BUT Detected (Bunlar doğal anomaliler olabilir)
            # FN = Injected BUT NOT Detected

            tp_count = len(injected_anomalies.intersection(detected_indices))
            fp_count = len(detected_indices - injected_anomalies)
            fn_count = len(injected_anomalies - detected_indices)

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "building_id": building.id,
                "building_name": building.name,
                "injected_count": len(injected_anomalies),
                "detected_total": len(detected_indices),
                "TP": tp_count,
                "FP": fp_count,
                "FN": fn_count,
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1_score": round(f1, 2)
            })

        except Exception as e:
            print(f"Hata (Bina {building.name}): {e}")

    return results


def run_anomaly_threshold_analysis(days: int = 30) -> list:
    """Anomali threshold hassasiyet analizi.

    Farklı Z-score eşik değerlerinin (2.5, 2.3, 2.0, 1.8) başarıya etkisini ölçer.
    Her threshold için 20 yapay anomali ekleyerek TP, FP, FN metriklerini çıkarır.

    Args:
        days: Test verisi süresi.

    Returns:
        list[dict]: Her threshold için özet performans (AVG Precision, Recall, F1).
    """
    thresholds = [2.5, 2.3, 2.0, 1.8]
    analysis_results = []
    
    buildings = Building.query.all()
    # Performans için veriyi bir kez çekip kopyalayabiliriz ama 
    # run_anomaly_performance_test fonksiyonunu doğrudan çağırmak daha temiz olurdu
    # fakat o fonksiyon threshold parametresi almıyor şu an.
    # O yüzden mantığı buraya taşıyacağız veya helper yapacağız.
    # En temizi: run_anomaly_performance_test'i güncelleyip threshold almasını sağlamak.
    # Ancak "Mevcut sistemi bozma" dendiği için, buraya copy-paste mantığıyla
    # ama threshold iterate eden bir yapı kuracağım.

    for th in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        building_count = 0

        for building in buildings:
            try:
                df = _build_dataframe(building.id, days=days)
                if len(df) < 50: continue

                df_test = df.copy()
                
                # Rastgele 20 anomali ekle (+%30)
                import random
                available_indices = list(df_test.index)
                if len(available_indices) < 20:
                    target_indices = available_indices
                else:
                    # Sabit seed kullan ki her threshold aynı noktaları test etsin (adil karşılaştırma)
                    # random.seed(42) # Loop içinde seed resetlemek yan etki yapabilir, dikkat.
                    # Basitlik için random kalsın veya indeksleri başta seçmek gerekirdi.
                    # Adil olması için her bina için seed'i bina ID ile sabitleyelim.
                    rng = random.Random(building.id + 42)
                    target_indices = rng.sample(available_indices, 20)
                
                injected_anomalies = set(target_indices)

                for idx in target_indices:
                    original_val = df_test.at[idx, TARGET_COL]
                    if original_val < 0.1:
                         df_test.at[idx, TARGET_COL] = 10.0
                    else:
                         df_test.at[idx, TARGET_COL] = original_val * 1.30

                # Tespit
                detected_list = _detect_anomalies_in_df(df_test, threshold=th)
                detected_indices = {d["index"] for d in detected_list}

                tp = len(injected_anomalies.intersection(detected_indices))
                fp = len(detected_indices - injected_anomalies)
                fn = len(injected_anomalies - detected_indices)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                building_count += 1
            
            except Exception:
                pass
        
        # Ortalama metrikler (Macro-average yerine Micro-average daha anlamlı olabilir toplam üzerinden)
        # Toplam TP/FP/FN üzerinden hesaplayalım
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        analysis_results.append({
            "threshold": th,
            "avg_TP": round(total_tp / max(1, building_count), 1),
            "avg_FP": round(total_fp / max(1, building_count), 1),
            "avg_FN": round(total_fn / max(1, building_count), 1),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2)
        })

    return analysis_results
    std_test_r2 = np.std([r['test_r2'] for r in results])

    print(f"{'Mean':<4} | {avg_train_r2:<8.4f} | {avg_test_r2:<8.4f} | {avg_mae:<6.2f} | {avg_rmse:<6.2f}")
    print(f"Std (Test R2): {std_test_r2:.4f}")
    
    # Overfitting Check
    diff = avg_train_r2 - avg_test_r2
    if diff > 0.07:
        print(f"\n⚠️  Possible overfitting detected! (Train-Test Diff: {diff:.4f})")
    elif avg_test_r2 < 0:
        print("\n⚠️  Model is performing poorly on unseen data (Negative R2).")
    else:
        print("\n✅ Model stability looks good.")
        
    print("=" * 60)


def compare_model_variants(building_id: int):
    """Farklı hiperparametre kombinasyonlarını karşılaştırır.
    
    Amaç: Overfitting'i azaltacak en iyi parametreleri bulmak.
    """
    print(f"\nModel Comparison for Building ID: {building_id}")
    print("=" * 100)
    
    # Veriyi hazırla
    df = _build_dataframe(building_id, days=TRAIN_DAYS)
    if len(df) < 100: return

    features = _get_feature_columns()
    X = df[features].values
    y = df[TARGET_COL].values
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    variants = [
        {"name": "Baseline (v2)", "params": {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2}},
        {"name": "Model A",       "params": {"n_estimators": 400, "max_depth": 20,   "min_samples_leaf": 5}},
        {"name": "Model B",       "params": {"n_estimators": 500, "max_depth": 15,   "min_samples_leaf": 5}},
        {"name": "Model C",       "params": {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 10}},
    ]
    
    print(f"{'Model':<15} | {'Train R2':<8} | {'Test R2':<8} | {'Diff':<6} | {'MAE':<6} | {'RMSE':<6}")
    print("-" * 75)
    
    best_model = None
    best_score = -999
    
    for v in variants:
        train_r2s, test_r2s, maes, rmses = [], [], [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = RandomForestRegressor(
                random_state=RANDOM_STATE, n_jobs=-1, **v["params"]
            )
            model.fit(X_train, y_train)
            
            train_r2s.append(r2_score(y_train, model.predict(X_train)))
            test_r2s.append(r2_score(y_test, model.predict(X_test)))
            maes.append(mean_absolute_error(y_test, model.predict(X_test)))
            rmses.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
            
        avg_train = np.mean(train_r2s)
        avg_test = np.mean(test_r2s)
        diff = avg_train - avg_test
        avg_mae = np.mean(maes)
        avg_rmse = np.mean(rmses)
        
        print(f"{v['name']:<15} | {avg_train:<8.4f} | {avg_test:<8.4f} | {diff:<6.4f} | {avg_mae:<6.2f} | {avg_rmse:<6.2f}")
        
        # Basit puanlama: Test R2 yüksek ve Diff düşük olmalı
        # Score = Test R2 - (Diff * 2) -> Diff cezası
        score = avg_test - (diff * 2)
        if score > best_score:
            best_score = score
            best_model = v["name"]
            
    print("-" * 75)
    print(f"🏆 Recommended Model: {best_model}")
    print("=" * 100)
