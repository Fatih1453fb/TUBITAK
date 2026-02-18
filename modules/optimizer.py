"""
Kampüs Enerji Optimizasyon Sistemi — Optimizasyon Motoru
==========================================================
ML tahminleri, enerji tarifesi, doluluk oranı ve hava durumu
verilerini kullanarak bina bazlı optimizasyon önerileri üretir.

Strateji katmanları:
  1. Tarife Optimizasyonu    — gece tarifesine yük kaydırma
  2. Doluluk Optimizasyonu   — boş binalarda HVAC azaltma
  3. Hava Durumu Adaptasyonu — pre-cooling / pre-heating
  4. Anomali Müdahalesi      — anormal tüketim uyarıları

Kullanım:
    from modules.optimizer import generate_recommendations, generate_all_recommendations

    result = generate_recommendations(building_id=1)
    all_results = generate_all_recommendations()
"""

from datetime import datetime, timedelta, timezone

from modules.database import (
    Building,
    EnergyReading,
    OptimizationRecommendation,
    WeatherData,
    db,
)
from modules.ml_model import detect_anomalies, predict_next_24h
from sqlalchemy import func

# ---------------------------------------------------------------------------
# SABİTLER
# ---------------------------------------------------------------------------

# Gece tarifesi saatleri (23:00 – 06:00) — %30 indirimli
NIGHT_TARIFF_START = 23
NIGHT_TARIFF_END = 6
NIGHT_DISCOUNT_RATIO = 0.30  # gece tarifesi indirimi

# Doluluk eşikleri
LOW_OCCUPANCY_THRESHOLD = 0.15   # bu altında bina "boş" sayılır
HVAC_REDUCTION_PERCENT = 30      # boş binada HVAC azaltma yüzdesi

# Sıcaklık eşikleri (°C)
HIGH_TEMP_THRESHOLD = 26  # pre-cooling
LOW_TEMP_THRESHOLD = 8    # pre-heating

# Enerji birim fiyatı (TL/kWh) — maliyet tahmini için
ENERGY_PRICE_TL = 3.50

# Karbon emisyon dönüşüm faktörü (kg CO₂ / kWh)
CO2_FACTOR = 0.43


# ---------------------------------------------------------------------------
# TAMİR STRATEJİLERİ
# ---------------------------------------------------------------------------


def _tariff_optimization(
    building: Building,
    predictions: list,
    recent_avg_kwh: float,
) -> list:
    """Gece/gündüz tarife farkından yararlanarak yük kaydırma önerileri.

    Gece saatlerinde (23:00-06:00) predicted_kwh düşükse ve gündüz
    saatlerinde yüksekse, esnek yükleri geceye kaydırmayı önerir.
    """
    recommendations = []

    daytime_preds = []
    nighttime_preds = []

    for p in predictions:
        ts = datetime.fromisoformat(p["timestamp"])
        hour = ts.hour
        kwh = p["predicted_kwh"]

        if NIGHT_TARIFF_END <= hour < NIGHT_TARIFF_START:
            daytime_preds.append(kwh)
        else:
            nighttime_preds.append(kwh)

    if not daytime_preds or not nighttime_preds:
        return recommendations

    avg_daytime = sum(daytime_preds) / len(daytime_preds)
    avg_nighttime = sum(nighttime_preds) / len(nighttime_preds)

    # Gündüz tüketimi yüksek ve geceye kaydırma potansiyeli varsa
    if avg_daytime > recent_avg_kwh * 0.8:
        shiftable_kwh = avg_daytime * 0.15 * len(daytime_preds)
        saving_kwh = shiftable_kwh * NIGHT_DISCOUNT_RATIO
        saving_percent = (saving_kwh / (sum(daytime_preds) + sum(nighttime_preds))) * 100

        recommendations.append({
            "action": "shift_load_to_night",
            "description": (
                f"{building.name}: Esnek yükleri gece tarifesi saatlerine "
                f"(23:00-06:00) kaydırın. Gece tarifesi %{int(NIGHT_DISCOUNT_RATIO*100)} "
                f"daha ucuz. Tahmini günlük kaydırılabilir yük: "
                f"{shiftable_kwh:.1f} kWh."
            ),
            "estimated_saving_kwh": round(saving_kwh, 2),
            "estimated_saving_percent": round(saving_percent, 1),
            "priority": "medium",
        })

    return recommendations


def _occupancy_optimization(
    building: Building,
    recent_readings: list,
    recent_avg_kwh: float,
) -> list:
    """Düşük doluluk oranlarında HVAC/aydınlatma azaltma önerileri."""
    recommendations = []

    low_occ_readings = [
        r for r in recent_readings
        if r.occupancy_rate is not None and r.occupancy_rate < LOW_OCCUPANCY_THRESHOLD
    ]

    if not low_occ_readings:
        return recommendations

    low_occ_ratio = len(low_occ_readings) / max(len(recent_readings), 1)

    if low_occ_ratio > 0.20:
        # Binanın önemli bir kısmı düşük dolulukta
        avg_kwh_low = sum(r.total_kwh for r in low_occ_readings) / len(low_occ_readings)
        saving_per_hour = avg_kwh_low * (HVAC_REDUCTION_PERCENT / 100)
        total_saving = saving_per_hour * len(low_occ_readings)
        saving_percent = (total_saving / max(sum(r.total_kwh for r in recent_readings), 1)) * 100

        recommendations.append({
            "action": "reduce_hvac",
            "description": (
                f"{building.name}: Son 24 saatte doluluk oranı "
                f"%{LOW_OCCUPANCY_THRESHOLD*100:.0f}'in altında kalan "
                f"{len(low_occ_readings)} saat tespit edildi. "
                f"Bu saatlerde HVAC kapasitesini %{HVAC_REDUCTION_PERCENT} azaltın."
            ),
            "estimated_saving_kwh": round(total_saving, 2),
            "estimated_saving_percent": round(saving_percent, 1),
            "priority": "high",
        })

    # Aydınlatma önerisi
    night_low_occ = [
        r for r in low_occ_readings
        if r.timestamp.hour >= 20 or r.timestamp.hour <= 6
    ]
    if len(night_low_occ) >= 4:
        lighting_saving = sum(r.total_kwh * 0.10 for r in night_low_occ)
        recommendations.append({
            "action": "reduce_lighting",
            "description": (
                f"{building.name}: Gece saatlerinde ({len(night_low_occ)} saat) "
                f"doluluk çok düşük. Aydınlatmayı minimum seviyeye indirin."
            ),
            "estimated_saving_kwh": round(lighting_saving, 2),
            "estimated_saving_percent": round(
                (lighting_saving / max(sum(r.total_kwh for r in recent_readings), 1)) * 100, 1
            ),
            "priority": "low",
        })

    return recommendations


def _weather_optimization(
    building: Building,
    predictions: list,
    current_weather: dict,
) -> list:
    """Hava durumu tahminine göre pre-cooling / pre-heating önerileri."""
    recommendations = []

    temp = current_weather.get("temperature")
    if temp is None:
        return recommendations

    # Toplam tahmin edilen tüketim
    total_predicted = sum(p["predicted_kwh"] for p in predictions) if predictions else 1.0

    if temp > HIGH_TEMP_THRESHOLD:
        # Pre-cooling: Sabah erken saatlerde binayı önceden soğut
        excess_deg = temp - HIGH_TEMP_THRESHOLD
        saving_kwh = min(excess_deg * 2.5, total_predicted * 0.10)

        recommendations.append({
            "action": "pre_cooling",
            "description": (
                f"{building.name}: Mevcut sıcaklık {temp:.1f}°C "
                f"(eşik: {HIGH_TEMP_THRESHOLD}°C). Sabah erken saatlerde "
                f"(05:00-07:00) binayı önceden soğutarak gün içi soğutma "
                f"yükünü azaltın. Gece tarifesinden de yararlanılır."
            ),
            "estimated_saving_kwh": round(saving_kwh, 2),
            "estimated_saving_percent": round((saving_kwh / total_predicted) * 100, 1),
            "priority": "medium",
        })

    elif temp < LOW_TEMP_THRESHOLD:
        # Pre-heating: Mesai öncesi binayı önceden ısıt
        deficit_deg = LOW_TEMP_THRESHOLD - temp
        saving_kwh = min(deficit_deg * 2.0, total_predicted * 0.08)

        recommendations.append({
            "action": "pre_heating",
            "description": (
                f"{building.name}: Mevcut sıcaklık {temp:.1f}°C "
                f"(eşik: {LOW_TEMP_THRESHOLD}°C). Mesai başlamadan önce "
                f"(06:00-08:00) binayı kademeli ısıtarak ani pik yükü "
                f"önleyin."
            ),
            "estimated_saving_kwh": round(saving_kwh, 2),
            "estimated_saving_percent": round((saving_kwh / total_predicted) * 100, 1),
            "priority": "medium",
        })

    return recommendations


def _anomaly_optimization(
    building: Building,
    anomalies: list,
    recent_avg_kwh: float,
) -> list:
    """Anomali tespit sonuçlarına göre acil müdahale önerileri."""
    recommendations = []

    if not anomalies:
        return recommendations

    for anomaly in anomalies[:3]:  # en fazla 3 anomali önerisi
        excess_kwh = anomaly["total_kwh"] - anomaly["expected_kwh"]
        saving_percent = anomaly.get("deviation_percent", 0.0)

        recommendations.append({
            "action": "anomaly_intervention",
            "description": (
                f"{building.name}: {anomaly['timestamp']} tarihinde "
                f"anormal tüketim tespit edildi. Beklenen: "
                f"{anomaly['expected_kwh']:.1f} kWh, Ölçülen: "
                f"{anomaly['total_kwh']:.1f} kWh "
                f"(Z-skoru: {anomaly['z_score']:.1f}). "
                f"HVAC, aydınlatma ve ekipman kontrolü yapın."
            ),
            "estimated_saving_kwh": round(max(0, excess_kwh), 2),
            "estimated_saving_percent": round(saving_percent, 1),
            "priority": "high",
        })

    return recommendations


# ---------------------------------------------------------------------------
# ANA FONKSİYONLAR
# ---------------------------------------------------------------------------


def generate_recommendations(building_id: int) -> dict:
    """Belirtilen bina için tüm optimizasyon stratejilerini çalıştırır.

    Args:
        building_id: Hedef bina ID'si.

    Returns:
        dict: Öneriler, tahmini tasarruflar ve öncelik bilgileri.
    """
    building = Building.query.get(building_id)
    if building is None:
        raise ValueError(f"Bina bulunamadı: id={building_id}")

    now = datetime.utcnow()

    # --- Son 24 saatlik enerji okumaları ---
    cutoff_24h = now - timedelta(hours=24)
    recent_readings = (
        EnergyReading.query
        .filter(
            EnergyReading.building_id == building_id,
            EnergyReading.timestamp >= cutoff_24h,
        )
        .order_by(EnergyReading.timestamp)
        .all()
    )

    recent_avg_kwh = (
        sum(r.total_kwh for r in recent_readings) / max(len(recent_readings), 1)
    )

    # --- 24 saatlik AI tahminleri ---
    try:
        predictions = predict_next_24h(building_id)
    except Exception:
        predictions = []

    # --- Güncel hava durumu ---
    latest_weather = (
        WeatherData.query
        .order_by(WeatherData.timestamp.desc())
        .first()
    )
    current_weather = latest_weather.to_dict() if latest_weather else {}

    # --- Anomali tespiti ---
    try:
        anomalies = detect_anomalies(building_id, threshold=2.5)
    except Exception:
        anomalies = []

    # --- Tüm stratejileri çalıştır ---
    all_recommendations = []

    all_recommendations.extend(
        _tariff_optimization(building, predictions, recent_avg_kwh)
    )
    all_recommendations.extend(
        _occupancy_optimization(building, recent_readings, recent_avg_kwh)
    )
    all_recommendations.extend(
        _weather_optimization(building, predictions, current_weather)
    )
    all_recommendations.extend(
        _anomaly_optimization(building, anomalies, recent_avg_kwh)
    )

    # --- Toplam tasarruf hesapla ---
    total_saving_kwh = sum(r["estimated_saving_kwh"] for r in all_recommendations)
    total_24h_kwh = sum(r.total_kwh for r in recent_readings) if recent_readings else 1.0
    total_saving_percent = (total_saving_kwh / max(total_24h_kwh, 1.0)) * 100

    # --- CO₂ tasarrufu hesapla ---
    for rec in all_recommendations:
        rec["estimated_co2_saving"] = round(rec["estimated_saving_kwh"] * CO2_FACTOR, 2)

    # --- Veritabanına kaydet ---
    for rec in all_recommendations:
        db_rec = OptimizationRecommendation(
            building_id=building_id,
            timestamp=now,
            recommendation_text=rec["description"],
            estimated_saving_kwh=rec["estimated_saving_kwh"],
            estimated_saving_percent=rec["estimated_saving_percent"],
            priority=rec["priority"],
            status="pending",
        )
        db.session.add(db_rec)

    db.session.commit()

    return {
        "building_id": building_id,
        "building_name": building.name,
        "recommendations": all_recommendations,
        "total_potential_saving_kwh": round(total_saving_kwh, 2),
        "total_potential_saving_percent": round(total_saving_percent, 1),
        "generated_at": now.isoformat(),
    }


def generate_all_recommendations() -> dict:
    """Tüm binalar için optimizasyon önerileri üretir.

    Returns:
        dict: {building_id: result} şeklinde tüm sonuçlar.
    """
    buildings = Building.query.all()
    results = {}

    for b in buildings:
        try:
            result = generate_recommendations(b.id)
            count = len(result["recommendations"])
            saving = result["total_potential_saving_kwh"]
            print(f"   ✅ {b.name}: {count} öneri, "
                  f"toplam tasarruf potansiyeli: {saving:.1f} kWh")
            results[b.id] = result
        except Exception as e:
            results[b.id] = {"error": str(e)}
            print(f"   ⚠️  {b.name}: {e}")

    return results


def calculate_campus_wide_savings() -> dict:
    """Kampüs genelinde toplam tasarruf potansiyelini hesaplar.

    Tüm aktif (pending) önerilerin tasarruf değerlerini toplar.

    Returns:
        dict: Kampüs geneli tasarruf özeti.
    """
    pending_recs = OptimizationRecommendation.query.filter_by(status="pending").all()

    total_saving_kwh = sum(r.estimated_saving_kwh or 0 for r in pending_recs)
    total_saving_tl = total_saving_kwh * ENERGY_PRICE_TL
    total_co2_saving_kg = round(total_saving_kwh * CO2_FACTOR, 2)

    # Öncelik dağılımı
    high = sum(1 for r in pending_recs if r.priority == "high")
    medium = sum(1 for r in pending_recs if r.priority == "medium")
    low = sum(1 for r in pending_recs if r.priority == "low")

    return {
        "total_pending_recommendations": len(pending_recs),
        "total_potential_saving_kwh": round(total_saving_kwh, 2),
        "total_potential_saving_tl": round(total_saving_tl, 2),
        "total_co2_saving_kg": total_co2_saving_kg,
        "priority_breakdown": {
            "high": high,
            "medium": medium,
            "low": low,
        },
    }


def simulate_optimization_scenario(params: dict) -> dict:
    """'Ne olurdu' senaryosu simüle eder.

    Args:
        params: Senaryo parametreleri — örnek:
            {
                "building_id": 1,
                "hvac_reduction_percent": 30,
                "lighting_reduction_percent": 20,
                "shift_load_percent": 15,
            }

    Returns:
        dict: Simülasyon sonuçları.
    """
    building_id = params.get("building_id")
    if building_id is None:
        raise ValueError("building_id gerekli.")

    building = Building.query.get(building_id)
    if building is None:
        raise ValueError(f"Bina bulunamadı: id={building_id}")

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=24)

    readings = (
        EnergyReading.query
        .filter(
            EnergyReading.building_id == building_id,
            EnergyReading.timestamp >= cutoff,
        )
        .all()
    )

    if not readings:
        raise ValueError(f"Bina {building_id} için son 24 saat verisi yok.")

    total_kwh = sum(r.total_kwh for r in readings)

    hvac_red = params.get("hvac_reduction_percent", 0) / 100
    light_red = params.get("lighting_reduction_percent", 0) / 100
    shift_pct = params.get("shift_load_percent", 0) / 100

    # Kabaca enerji dağılımı: HVAC %50, Aydınlatma %20, Diğer %30
    hvac_saving = total_kwh * 0.50 * hvac_red
    lighting_saving = total_kwh * 0.20 * light_red
    shift_saving = total_kwh * shift_pct * NIGHT_DISCOUNT_RATIO

    total_saving = hvac_saving + lighting_saving + shift_saving
    new_total = total_kwh - total_saving

    return {
        "building_id": building_id,
        "building_name": building.name,
        "scenario_params": params,
        "current_24h_kwh": round(total_kwh, 2),
        "projected_24h_kwh": round(max(0, new_total), 2),
        "savings": {
            "hvac_saving_kwh": round(hvac_saving, 2),
            "lighting_saving_kwh": round(lighting_saving, 2),
            "load_shift_saving_kwh": round(shift_saving, 2),
            "total_saving_kwh": round(total_saving, 2),
            "total_saving_percent": round((total_saving / max(total_kwh, 1)) * 100, 1),
            "estimated_cost_saving_tl": round(total_saving * ENERGY_PRICE_TL, 2),
        },
    }


def run_optimizer_backtest(days: int = 30, hvac_reduction_percent: float = None) -> dict:
    """Geçmiş veriler üzerinde optimizasyon simülasyonu (Backtesting).

    Son 'days' günlük gerçek veriyi alır, optimizer kurallarını (doluluk,
    tarife, hava durumu) uygulayarak "eğer optimizasyon yapılsaydı
    ne kadar tasarruf edilirdi" sorusunu cevaplar.

    Args:
        days: Geriye dönük kaç gün analiz edilecek.
        hvac_reduction_percent: (Opsiyonel) HVAC azaltma yüzdesi override.

    Returns:
        dict: Kampüs geneli ve bina bazlı sonuçlar.
    """
    buildings = Building.query.all()
    results = {
        "campus_summary": {},
        "building_details": [],
        "analysis_period_days": days,
    }

    total_actual_kwh = 0.0
    total_simulated_kwh = 0.0
    total_saved_kwh = 0.0

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Hava durumu verilerini önbelleğe al (performans için)
    weather_records = WeatherData.query.filter(WeatherData.timestamp >= cutoff).all()
    # Saat bazlı sözlük: "YYYY-MM-DD HH" -> {temp, ...}
    weather_map = {
        w.timestamp.strftime("%Y-%m-%d %H"): w.to_dict()
        for w in weather_records
    }

    for building in buildings:
        # Binanın enerji okumalarını çek
        readings = (
            EnergyReading.query
            .filter(
                EnergyReading.building_id == building.id,
                EnergyReading.timestamp >= cutoff
            )
            .order_by(EnergyReading.timestamp)
            .all()
        )

        if not readings:
            continue

        b_actual = 0.0
        b_simulated = 0.0
        b_hvac_save = 0.0
        b_shift_save = 0.0
        b_weather_save = 0.0

        # Her saatlik okuma için kuralları uygula
        for r in readings:
            actual = r.total_kwh
            timestamp = r.timestamp
            hour = timestamp.hour
            ts_key = timestamp.strftime("%Y-%m-%d %H")
            weather = weather_map.get(ts_key, {})
            temp = weather.get("temperature")

            # Mevcut durumu simülasyon başlangıcı yap
            simulated = actual

            # 1. Doluluk Optimizasyonu (HVAC/Aydınlatma Azaltma)
            # Eğer doluluk düşükse ve tüketim 0 değilse
            if r.occupancy_rate is not None and r.occupancy_rate < LOW_OCCUPANCY_THRESHOLD:
                # HVAC azaltma potansiyeli (basitçe toplamın %30'unun %30'u gibi varsayalım veya direkt %10)
                # Daha agresif: Tüm tüketimin %10'u tasarruf edilebilir (HVAC + Işık)
                use_percent = hvac_reduction_percent if hvac_reduction_percent is not None else HVAC_REDUCTION_PERCENT
                saving = actual * (use_percent / 100) * 0.5 # %30'un yarısı kadar net etki
                b_hvac_save += saving
                simulated -= saving

            # 2. Hava Durumu (Pre-cooling / Pre-heating)
            # Eğer hava çok sıcak/soğuk ise ve saat uygunsa
            if temp:
                if temp > HIGH_TEMP_THRESHOLD and 12 <= hour <= 18:
                     # Gündüz piki, eğer sabah pre-cooling yapılsaydı?
                     # Basitçe: %3 tasarruf varsayalım
                     saving = actual * 0.03
                     b_weather_save += saving
                     simulated -= saving
                elif temp < LOW_TEMP_THRESHOLD and 8 <= hour <= 10:
                     # Sabah ısınma piki, pre-heating ile engellenebilirdi
                     saving = actual * 0.03
                     b_weather_save += saving
                     simulated -= saving

            # 3. Tarife Kaydırma (Load Shifting)
            # Eğer gündüz saatiyse (pahalı) ve tüketim yüksekse, bir kısmı geceye kaymış olabilirdi.
            # Ancak backtest'te toplam kWh değişmez, sadece maliyet değişir.
            # "Tasarruf" olarak genelde maliyet değil kWh istendiği için
            # burada kWh düşüşü değil, "gece tarifesi etkisi"ni maliyetten kazanç olarak değil
            # verimlilik artışı olarak yansıtamayız. 
            # FAKAT kullanıcı isteği "Tasarruf kWh" diyor. Yük kaydırma kWh azaltmaz (hatta artırabilir).
            # O yüzden burayı sadece "gereksiz yükü kapatma" gibi düşünelim veya pas geçelim.
            # Talep: "Optimize edilmiş simülasyon tüketimi". 
            # Yük kaydırma sadece zamanı değiştirir. Biz burada NET kWh tasarrufuna odaklanalım.
            # Yine de gece tarifesi kuralı "shift_load_to_night" kWh tasarrufu da sağlayabilir (daha verimli çalışma).
            # Şimdilik Load Shifting'i kWh düşüşü olarak yansıtmayalım (0 değişim).
            
            b_actual += actual
            b_simulated += max(0, simulated)

        b_saved = b_actual - b_simulated
        b_saved_pct = (b_saved / b_actual * 100) if b_actual > 0 else 0

        detail = {
            "building_id": building.id,
            "building_name": building.name,
            "actual_kwh": round(b_actual, 2),
            "simulated_kwh": round(b_simulated, 2),
            "saved_kwh": round(b_saved, 2),
            "saved_percent": round(b_saved_pct, 1),
            "co2_saved_kg": round(b_saved * CO2_FACTOR, 2),
            "breakdown": {
                "hvac_occupancy_save": round(b_hvac_save, 2),
                "weather_save": round(b_weather_save, 2)
            }
        }
        results["building_details"].append(detail)

        total_actual_kwh += b_actual
        total_simulated_kwh += b_simulated
        total_saved_kwh += b_saved

    total_saved_pct = (total_saved_kwh / total_actual_kwh * 100) if total_actual_kwh > 0 else 0

    results["campus_summary"] = {
        "total_actual_kwh": round(total_actual_kwh, 2),
        "total_simulated_kwh": round(total_simulated_kwh, 2),
        "total_saved_kwh": round(total_saved_kwh, 2),
        "total_saved_percent": round(total_saved_pct, 1),
        "total_co2_saved_kg": round(total_saved_kwh * CO2_FACTOR, 2),
    }

    return results


def run_sensitivity_analysis(days: int = 30) -> list:
    """Optimizer hassasiyet analizi.

    Farklı HVAC azaltma oranlarının tasarruf üzerindeki etkisini test eder.
    Test edilen oranlar: %5, %10, %15, %20

    Args:
        days: Geriye dönük kaç gün analiz edilecek.

    Returns:
        list[dict]: Her oran için analiz sonuçları.
    """
    rates = [5, 10, 15, 20]
    analysis_results = []

    for rate in rates:
        # Backtest çalıştır
        backtest = run_optimizer_backtest(days=days, hvac_reduction_percent=rate)
        summary = backtest["campus_summary"]

        analysis_results.append({
            "hvac_reduction_percent": rate,
            "total_saved_kwh": summary["total_saved_kwh"],
            "saved_percent": summary["total_saved_percent"],
            "co2_saved_kg": summary["total_co2_saved_kg"]
        })

    return analysis_results
