"""
Kampüs Enerji Optimizasyon Sistemi — Demo Veri Üreteci
=======================================================
Sistem ilk başlatıldığında boş veritabanına gerçekçi demo
veri yükler.

Üretilen veri:
  • 5 kampüs binası
  • 10.800 saatlik enerji okuma kaydı (5 bina × 90 gün × 24 saat)
  • 2.160 saatlik hava durumu kaydı (90 gün × 24 saat)
"""

import math
import random
from datetime import datetime, timedelta, timezone

from modules.database import (
    Alert,
    Building,
    EnergyReading,
    OptimizationRecommendation,
    WeatherData,
    db,
)

# ---------------------------------------------------------------------------
# SABİTLER
# ---------------------------------------------------------------------------

BUILDINGS = [
    {"name": "A Blok Laboratuvarları", "type": "laboratory",  "floor_area": 1200.0, "capacity": 150, "construction_year": 2015},
    {"name": "B Blok Derslikler",      "type": "classroom",   "floor_area": 2000.0, "capacity": 400, "construction_year": 2010},
    {"name": "C Blok Ofisler",         "type": "office",      "floor_area":  800.0, "capacity": 100, "construction_year": 2018},
    {"name": "Kütüphane",              "type": "social",      "floor_area":  600.0, "capacity": 200, "construction_year": 2012},
    {"name": "Spor Merkezi",           "type": "social",      "floor_area": 1000.0, "capacity": 300, "construction_year": 2020},
]

SEED_DAYS = 90  # Kaç günlük geçmiş veri üretilecek

# Bina tipine göre temel yük profilleri (kWh / saat / m²)
BASE_LOAD_PER_M2 = {
    "laboratory": 0.045,
    "classroom":  0.030,
    "office":     0.025,
    "social":     0.020,
}

# ---------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------------------


def _seasonal_temperature(dt: datetime) -> float:
    """Mevsimsel sinüs eğrisi ile sıcaklık üret (İstanbul iklimi).

    Yaz ortalaması ~28°C, kış ortalaması ~6°C.
    Gün içi varyasyon ±4°C (öğlen sıcak, gece soğuk).
    """
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour

    # Yıllık sinüs: Temmuz (yday≈200) pik, Ocak (yday≈15) dip
    yearly_cycle = math.sin(2 * math.pi * (day_of_year - 15) / 365)
    base_temp = 17.0 + 11.0 * yearly_cycle  # 6°C – 28°C arası

    # Gün içi sinüs: 14:00 pik, 04:00 dip
    daily_cycle = math.sin(2 * math.pi * (hour - 4) / 24)
    base_temp += 4.0 * daily_cycle

    # Rastgele gürültü
    base_temp += random.gauss(0, 1.2)
    return round(base_temp, 1)


def _seasonal_humidity(temperature: float) -> float:
    """Sıcaklık ile ters korelasyonlu nem üret."""
    base = 70.0 - 0.8 * temperature + random.gauss(0, 5)
    return round(max(20.0, min(98.0, base)), 1)


def _solar_radiation(dt: datetime) -> float:
    """Güneş radyasyonu (W/m²). Gece 0, öğlen pik."""
    hour = dt.hour
    if hour < 6 or hour > 20:
        return 0.0

    # Gündüz sinüs profili — öğlen 12:00-13:00 pik
    day_progress = math.sin(math.pi * (hour - 6) / 14)
    day_of_year = dt.timetuple().tm_yday
    seasonal_factor = 0.7 + 0.3 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

    radiation = 900.0 * day_progress * seasonal_factor + random.gauss(0, 30)
    return round(max(0.0, radiation), 1)


def _weather_condition(temperature: float, humidity: float, radiation: float) -> str:
    """Basit hava durumu etiketi."""
    if radiation > 500:
        return "sunny"
    if humidity > 80:
        return "rainy" if temperature > 5 else "snowy"
    if radiation > 200:
        return "partly_cloudy"
    if radiation > 0:
        return "cloudy"
    return "clear_night"


def _occupancy_rate(dt: datetime, building_type: str) -> float:
    """Bina tipi ve zamana göre doluluk oranı üret (0.0 – 1.0)."""
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
            if 10 <= hour <= 20:
                rate = random.uniform(0.30, 0.60)
            else:
                rate = random.uniform(0.02, 0.10)
        elif 11 <= hour <= 14:        # öğle pik
            rate = random.uniform(0.60, 0.90)
        elif 17 <= hour <= 21:        # akşam pik
            rate = random.uniform(0.50, 0.75)
        elif 8 <= hour <= 22:
            rate = random.uniform(0.20, 0.45)
        else:
            rate = random.uniform(0.02, 0.08)

    return round(rate, 3)


def _energy_reading(
    building: dict,
    dt: datetime,
    temperature: float,
    occupancy: float,
) -> dict:
    """Tek bir saatlik enerji okuması üret.

    Formül: base_load + occupancy_effect + temperature_effect + noise
    """
    area = building["floor_area"]
    btype = building["type"]
    base = BASE_LOAD_PER_M2[btype] * area  # kWh baz yük

    # --- Doluluk etkisi (doluluk arttıkça tüketim artar) ---
    occupancy_effect = base * 0.5 * occupancy

    # --- Sıcaklık etkisi ---
    # Soğukta ısıtma, sıcakta soğutma yükü
    if temperature < 10:
        heating = base * 0.35 * (10 - temperature) / 15
        cooling = 0.0
    elif temperature > 25:
        heating = 0.0
        cooling = base * 0.40 * (temperature - 25) / 15
    else:
        heating = 0.0
        cooling = 0.0

    temp_effect = heating + cooling

    # --- Rastgele gürültü (%5-10) ---
    noise = random.gauss(0, base * 0.07)

    total = base + occupancy_effect + temp_effect + noise
    total = max(total * 0.1, total)  # negatif olmasın

    # Elektrik / ısıtma / soğutma ayrımı
    electricity = base + occupancy_effect + noise * 0.6
    electricity = max(0.0, electricity)

    return {
        "timestamp": dt,
        "electricity_kwh": round(electricity, 2),
        "heating_kwh": round(max(0.0, heating + noise * 0.25), 2),
        "cooling_kwh": round(max(0.0, cooling + noise * 0.15), 2),
        "total_kwh": round(max(0.5, total), 2),
        "occupancy_rate": occupancy,
    }


# ---------------------------------------------------------------------------
# ANA FONKSİYON
# ---------------------------------------------------------------------------


def seed_database(app):
    """Veritabanına demo veri yükle.

    Building tablosu zaten doluysa hiçbir şey yapmaz.

    Args:
        app: Flask uygulama örneği (app context için).
    """
    with app.app_context():
        # Zaten veri varsa atla
        if Building.query.first() is not None:
            print("ℹ️  Veritabanında mevcut veri var — seed atlanıyor.")
            return

        print("🌱 Demo veriler yükleniyor...")

        # --- 1) Binaları oluştur ------------------------------------------
        building_models = []
        for b in BUILDINGS:
            model = Building(
                name=b["name"],
                type=b["type"],
                floor_area=b["floor_area"],
                capacity=b["capacity"],
                construction_year=b["construction_year"],
            )
            db.session.add(model)
            building_models.append((model, b))

        db.session.flush()  # id'lerin atanmasını sağla
        print(f"   ✅ {len(building_models)} bina oluşturuldu.")

        # --- 2) Zaman damgalarını hazırla ----------------------------------
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(days=SEED_DAYS)
        total_hours = SEED_DAYS * 24
        timestamps = [start + timedelta(hours=h) for h in range(total_hours)]

        # --- 3) Hava durumu verisi -----------------------------------------
        weather_records = []
        # Hava durumu değerlerini ön-hesapla (enerji hesabında da kullanılacak)
        temp_cache: dict[int, float] = {}

        for ts in timestamps:
            temp = _seasonal_temperature(ts)
            temp_cache[id(ts)] = temp  # aynı ts objesinin id'si ile eşle

            hum = _seasonal_humidity(temp)
            rad = _solar_radiation(ts)
            condition = _weather_condition(temp, hum, rad)

            weather_records.append(
                WeatherData(
                    timestamp=ts,
                    temperature=temp,
                    humidity=hum,
                    wind_speed=round(random.uniform(0, 35), 1),
                    solar_radiation=rad,
                    weather_condition=condition,
                )
            )

        db.session.bulk_save_objects(weather_records)
        print(f"   ✅ {len(weather_records)} hava durumu kaydı oluşturuldu.")

        # --- 4) Enerji okumaları -------------------------------------------
        energy_records = []

        for model, binfo in building_models:
            for idx, ts in enumerate(timestamps):
                temp = temp_cache[id(ts)]
                occ = _occupancy_rate(ts, binfo["type"])
                reading = _energy_reading(binfo, ts, temp, occ)

                energy_records.append(
                    EnergyReading(
                        building_id=model.id,
                        **reading,
                    )
                )

            # Her binadan sonra batch commit (bellek dostu)
            if len(energy_records) >= 5000:
                db.session.bulk_save_objects(energy_records)
                db.session.flush()
                energy_records = []

        # Kalan kayıtları yaz
        if energy_records:
            db.session.bulk_save_objects(energy_records)

        print(f"   ✅ {len(building_models) * total_hours} enerji kaydı oluşturuldu.")

        # --- 5) Örnek anomali alert'i --------------------------------------
        #   Spec demo senaryosu: B Blok'ta hafta sonu 03:00 anomalisi
        b_blok = building_models[1][0]
        anomaly_alert = Alert(
            building_id=b_blok.id,
            alert_type="anomaly",
            severity="critical",
            message=(
                "B Blok Derslikler'de hafta sonu sabah 03:00'te "
                "anormal enerji tüketimi tespit edildi. "
                "Beklenen: ~12 kWh, Ölçülen: ~58 kWh. "
                "Olası sebep: HVAC sistemi kapatılmamış."
            ),
            is_resolved=False,
        )
        db.session.add(anomaly_alert)

        # --- 6) Örnek optimizasyon önerisi ---------------------------------
        recommendation = OptimizationRecommendation(
            building_id=b_blok.id,
            timestamp=now,
            recommendation_text=(
                "B Blok Derslikler'de HVAC sistemini hafta sonu "
                "gece saatlerinde otomatik kapatın. "
                "Tahmini tasarruf: 45 kWh/hafta sonu."
            ),
            estimated_saving_kwh=45.0,
            estimated_saving_percent=18.5,
            priority="high",
            status="pending",
        )
        db.session.add(recommendation)

        # --- COMMIT --------------------------------------------------------
        db.session.commit()
        print("🎉 Tüm demo veriler başarıyla yüklendi!")
        print(f"   📊 Toplam: {len(building_models)} bina, "
              f"{len(building_models) * total_hours} enerji kaydı, "
              f"{len(weather_records)} hava durumu kaydı")
