"""
Kampüs Enerji Optimizasyon Sistemi — Veritabanı Modülleri
==========================================================
SQLAlchemy ORM modelleri ve veritabanı başlatma.
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

# ---------------------------------------------------------------------------
# SQLAlchemy örneği  — app.py içinde init_app() ile Flask'a bağlanır
# ---------------------------------------------------------------------------
db = SQLAlchemy()


# ========================== MODELLER =======================================

class Building(db.Model):
    """Kampüs binalarını temsil eden model.

    Bina tipleri: 'laboratory', 'classroom', 'office', 'social'
    """

    __tablename__ = "buildings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)          # laboratory | classroom | office | social
    floor_area = db.Column(db.Float, nullable=True)           # m²
    construction_year = db.Column(db.Integer, nullable=True)
    capacity = db.Column(db.Integer, nullable=True)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # İlişkiler
    energy_readings = db.relationship(
        "EnergyReading", backref="building", lazy="dynamic", cascade="all, delete-orphan"
    )
    predictions = db.relationship(
        "Prediction", backref="building", lazy="dynamic", cascade="all, delete-orphan"
    )
    recommendations = db.relationship(
        "OptimizationRecommendation", backref="building", lazy="dynamic", cascade="all, delete-orphan"
    )
    alerts = db.relationship(
        "Alert", backref="building", lazy="dynamic", cascade="all, delete-orphan"
    )

    def to_dict(self):
        """Model verisini JSON-uyumlu sözlüğe dönüştür."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "floor_area": self.floor_area,
            "construction_year": self.construction_year,
            "capacity": self.capacity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Building {self.id}: {self.name}>"


class EnergyReading(db.Model):
    """Bina bazlı enerji tüketim kayıtları."""

    __tablename__ = "energy_readings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    building_id = db.Column(
        db.Integer, db.ForeignKey("buildings.id"), nullable=False, index=True
    )
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    electricity_kwh = db.Column(db.Float, default=0.0)
    heating_kwh = db.Column(db.Float, default=0.0)
    cooling_kwh = db.Column(db.Float, default=0.0)
    total_kwh = db.Column(db.Float, nullable=False)
    occupancy_rate = db.Column(db.Float, default=0.0)         # 0.0 – 1.0
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "building_id": self.building_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "electricity_kwh": self.electricity_kwh,
            "heating_kwh": self.heating_kwh,
            "cooling_kwh": self.cooling_kwh,
            "total_kwh": self.total_kwh,
            "occupancy_rate": self.occupancy_rate,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<EnergyReading {self.id} — {self.building_id} @ {self.timestamp}>"


class WeatherData(db.Model):
    """Hava durumu verileri."""

    __tablename__ = "weather_data"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    temperature = db.Column(db.Float)            # °C
    humidity = db.Column(db.Float)                # %
    wind_speed = db.Column(db.Float)              # km/h
    solar_radiation = db.Column(db.Float)         # W/m²
    weather_condition = db.Column(db.String(50))
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "solar_radiation": self.solar_radiation,
            "weather_condition": self.weather_condition,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Prediction(db.Model):
    """AI model tahmin sonuçları."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    building_id = db.Column(
        db.Integer, db.ForeignKey("buildings.id"), nullable=False, index=True
    )
    prediction_timestamp = db.Column(db.DateTime, nullable=False)
    target_timestamp = db.Column(db.DateTime, nullable=False)
    predicted_kwh = db.Column(db.Float)
    actual_kwh = db.Column(db.Float)
    model_version = db.Column(db.String(20))
    accuracy_score = db.Column(db.Float)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "building_id": self.building_id,
            "prediction_timestamp": self.prediction_timestamp.isoformat()
            if self.prediction_timestamp
            else None,
            "target_timestamp": self.target_timestamp.isoformat()
            if self.target_timestamp
            else None,
            "predicted_kwh": self.predicted_kwh,
            "actual_kwh": self.actual_kwh,
            "model_version": self.model_version,
            "accuracy_score": self.accuracy_score,
        }


class OptimizationRecommendation(db.Model):
    """Enerji optimizasyon önerileri."""

    __tablename__ = "optimization_recommendations"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    building_id = db.Column(
        db.Integer, db.ForeignKey("buildings.id"), nullable=False, index=True
    )
    timestamp = db.Column(db.DateTime, nullable=False)
    recommendation_text = db.Column(db.Text)
    estimated_saving_kwh = db.Column(db.Float)
    estimated_saving_percent = db.Column(db.Float)
    priority = db.Column(db.String(20), default="medium")    # high | medium | low
    status = db.Column(db.String(20), default="pending")     # pending | applied | dismissed
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "building_id": self.building_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "recommendation_text": self.recommendation_text,
            "estimated_saving_kwh": self.estimated_saving_kwh,
            "estimated_saving_percent": self.estimated_saving_percent,
            "priority": self.priority,
            "status": self.status,
        }


class Alert(db.Model):
    """Sistem alertleri ve anomali bildirimleri."""

    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    building_id = db.Column(
        db.Integer, db.ForeignKey("buildings.id"), nullable=False, index=True
    )
    alert_type = db.Column(db.String(50))           # anomaly | threshold | prediction_drift
    severity = db.Column(db.String(20))              # critical | warning | info
    message = db.Column(db.Text)
    is_resolved = db.Column(db.Boolean, default=False)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "building_id": self.building_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "is_resolved": self.is_resolved,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
