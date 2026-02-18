"""
Kampüs Enerji Optimizasyon Sistemi — Zamanlayıcı Modülü
========================================================
APScheduler ile her 5 dakikada bir gerçek zamanlı veri
simülasyonu yapar:
  1. Her bina için yeni EnergyReading üretir
  2. Yeni WeatherData kaydı ekler
  3. Anomali tespiti çalıştırır
  4. Optimizasyon önerilerini yeniler
"""

import math
import random
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from modules.database import (
    Building,
    EnergyReading,
    WeatherData,
    db,
)

# ---------------------------------------------------------------------------
# VERİ ÜRETİM YARDIMCILARI  (seed_data.py ile tutarlı formüller)
# ---------------------------------------------------------------------------

BASE_LOAD_PER_M2 = {
    "laboratory": 0.045,
    "classroom":  0.030,
    "office":     0.025,
    "social":     0.020,
}


def _seasonal_temperature(dt: datetime) -> float:
    """İstanbul iklimi — mevsimsel + gün içi sinüs."""
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour
    yearly = math.sin(2 * math.pi * (day_of_year - 15) / 365)
    base = 17.0 + 11.0 * yearly
    daily = math.sin(2 * math.pi * (hour - 4) / 24)
    base += 4.0 * daily
    base += random.gauss(0, 1.2)
    return round(base, 1)


def _seasonal_humidity(temperature: float) -> float:
    base = 70.0 - 0.8 * temperature + random.gauss(0, 5)
    return round(max(20.0, min(98.0, base)), 1)


def _solar_radiation(dt: datetime) -> float:
    hour = dt.hour
    if hour < 6 or hour > 20:
        return 0.0
    day_progress = math.sin(math.pi * (hour - 6) / 14)
    seasonal = 0.7 + 0.3 * math.sin(
        2 * math.pi * (dt.timetuple().tm_yday - 80) / 365
    )
    return round(max(0.0, 900.0 * day_progress * seasonal + random.gauss(0, 30)), 1)


def _weather_condition(temp: float, hum: float, rad: float) -> str:
    if rad > 500:
        return "sunny"
    if hum > 80:
        return "rainy" if temp > 5 else "snowy"
    if rad > 200:
        return "partly_cloudy"
    if rad > 0:
        return "cloudy"
    return "clear_night"


def _occupancy_rate(dt: datetime, building_type: str) -> float:
    hour = dt.hour
    is_weekend = dt.weekday() >= 5

    if building_type == "laboratory":
        if is_weekend:
            rate = 0.05 + 0.10 * (1 if 10 <= hour <= 16 else 0)
        elif 8 <= hour <= 18:
            rate = random.uniform(0.60, 0.90)
        elif 18 < hour <= 22:
            rate = random.uniform(0.15, 0.35)
        else:
            rate = random.uniform(0.02, 0.08)
    elif building_type == "classroom":
        if is_weekend:
            rate = random.uniform(0.02, 0.08)
        elif 8 <= hour <= 17:
            rate = random.uniform(0.70, 0.95)
        elif 17 < hour <= 21:
            rate = random.uniform(0.10, 0.25)
        else:
            rate = random.uniform(0.01, 0.05)
    elif building_type == "office":
        if is_weekend:
            rate = random.uniform(0.02, 0.06)
        elif 8 <= hour <= 17:
            rate = random.uniform(0.65, 0.85)
        elif 17 < hour <= 19:
            rate = random.uniform(0.15, 0.30)
        else:
            rate = random.uniform(0.01, 0.05)
    else:  # social
        if is_weekend:
            rate = random.uniform(0.30, 0.60) if 10 <= hour <= 20 else random.uniform(0.02, 0.10)
        elif 11 <= hour <= 14:
            rate = random.uniform(0.60, 0.90)
        elif 17 <= hour <= 21:
            rate = random.uniform(0.50, 0.75)
        elif 8 <= hour <= 22:
            rate = random.uniform(0.20, 0.45)
        else:
            rate = random.uniform(0.02, 0.08)

    return round(rate, 3)


def _generate_energy(building_type: str, floor_area: float,
                     temperature: float, occupancy: float) -> dict:
    """Tek bir enerji okuması üret (seed_data formülü ile tutarlı)."""
    base = BASE_LOAD_PER_M2[building_type] * floor_area

    occupancy_effect = base * 0.5 * occupancy

    if temperature < 10:
        heating = base * 0.35 * (10 - temperature) / 15
        cooling = 0.0
    elif temperature > 25:
        heating = 0.0
        cooling = base * 0.40 * (temperature - 25) / 15
    else:
        heating = 0.0
        cooling = 0.0

    noise = random.gauss(0, base * 0.07)
    total = base + occupancy_effect + heating + cooling + noise
    total = max(total * 0.1, total)

    electricity = max(0.0, base + occupancy_effect + noise * 0.6)

    return {
        "electricity_kwh": round(electricity, 2),
        "heating_kwh":     round(max(0.0, heating + noise * 0.25), 2),
        "cooling_kwh":     round(max(0.0, cooling + noise * 0.15), 2),
        "total_kwh":       round(max(0.5, total), 2),
        "occupancy_rate":  occupancy,
    }


# ---------------------------------------------------------------------------
# SCHEDULER ÇEKİRDEĞİ
# ---------------------------------------------------------------------------


_scheduler = None  # type: BackgroundScheduler | None
_flask_app = None
_socketio = None


def _tick():
    """Her 5 dakikada bir çalışan ana görev."""
    global _flask_app, _socketio
    if _flask_app is None:
        return

    with _flask_app.app_context():
        try:
            buildings = Building.query.all()
            if not buildings:
                return

            now = datetime.utcnow().replace(second=0, microsecond=0)

            # ── Hava durumu: son kaydın timestamp'inden sonra mı? ─────
            last_weather = (
                WeatherData.query
                .order_by(WeatherData.timestamp.desc())
                .first()
            )
            weather_ts = now
            if last_weather and last_weather.timestamp >= now:
                # Duplicate önle — 5 dakika ekle
                weather_ts = last_weather.timestamp + timedelta(minutes=5)

            temp = _seasonal_temperature(weather_ts)
            hum = _seasonal_humidity(temp)
            rad = _solar_radiation(weather_ts)
            cond = _weather_condition(temp, hum, rad)

            weather = WeatherData(
                timestamp=weather_ts,
                temperature=temp,
                humidity=hum,
                wind_speed=round(random.uniform(0, 35), 1),
                solar_radiation=rad,
                weather_condition=cond,
            )
            db.session.add(weather)

            # ── Her bina için enerji ────────────────────────────────
            for b in buildings:
                last_reading = (
                    EnergyReading.query
                    .filter_by(building_id=b.id)
                    .order_by(EnergyReading.timestamp.desc())
                    .first()
                )

                if last_reading:
                    energy_ts = last_reading.timestamp + timedelta(minutes=5)
                    # Duplicate önle
                    if energy_ts <= last_reading.timestamp:
                        continue
                else:
                    energy_ts = now

                occ = _occupancy_rate(energy_ts, b.type)
                reading = _generate_energy(b.type, b.floor_area, temp, occ)

                er = EnergyReading(
                    building_id=b.id,
                    timestamp=energy_ts,
                    **reading,
                )
                db.session.add(er)

            db.session.commit()
            print(f"[Scheduler] ⚡ New energy data added  (ts={weather_ts.strftime('%H:%M')})")

            # ── WebSocket ile Bildir ────────────────────────────────
            if _socketio:
                _socketio.emit('new_data', {
                    'timestamp': weather_ts.isoformat(),
                    'message': 'New data available'
                })
                print("[Scheduler] 📡 Emitted 'new_data' via SocketIO")

            # ── Anomali tespiti ──────────────────────────────────────
            try:
                from modules.ml_model import detect_anomalies
                for b in buildings:
                    detect_anomalies(b.id)
            except Exception as e:
                print(f"[Scheduler] ⚠️  Anomaly detection skipped: {e}")

            # ── Optimizer önerileri ──────────────────────────────────
            try:
                from modules.optimizer import generate_all_recommendations
                generate_all_recommendations()
                print("[Scheduler] 🔄 Recommendations updated")
            except Exception as e:
                print(f"[Scheduler] ⚠️  Recommendations skipped: {e}")

        except Exception as e:
            db.session.rollback()
            print(f"[Scheduler] ❌ Error: {e}")


def init_scheduler(app, socketio=None):
    """Flask app başlatıldığında APScheduler'ı kur ve çalıştır.

    Args:
        app: Flask uygulama örneği.
        socketio: Flask-SocketIO örneği (opsiyonel).
    """
    global _scheduler, _flask_app, _socketio
    _flask_app = app
    _socketio = socketio

    # Çift başlatma koruması (reloader vs production)
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _tick,
        trigger="interval",
        minutes=5,
        id="campus_energy_tick",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    print("[Scheduler] ✅ Başlatıldı — her 5 dakikada bir veri üretilecek.")


def shutdown_scheduler():
    """Scheduler'ı durdur."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        print("[Scheduler] ⏹  Durduruldu.")
        _scheduler = None
