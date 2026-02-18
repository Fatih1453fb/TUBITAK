"""
yapılandırmayı yükler ve veritabanını hazırlar.
"""

import eventlet
eventlet.monkey_patch()

import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from config import Config
from data.seed_data import seed_database
from modules.database import (
    Building,
    EnergyReading,
    OptimizationRecommendation,
    WeatherData,
    db,
)
from modules.ml_model import (
    detect_anomalies,
    get_feature_importance,
    get_grouped_feature_importance,
    get_model_accuracy,
    get_prediction_analysis,
    predict_next_24h,
    predict_next_7d,
    train_all_models,
    train_model,
)
from modules.optimizer import (
    calculate_campus_wide_savings,
    generate_all_recommendations,
    generate_recommendations,
    simulate_optimization_scenario,
)
from modules.scheduler import init_scheduler

# Global accessibility for extensions if needed
socketio = SocketIO()

def create_app(config_class=Config):
    """Flask uygulama fabrikası (Application Factory pattern)."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # ----- Eklentileri başlat -----
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # ----- Veritabanı tablolarını oluştur + demo veri yükle -----
    with app.app_context():
        db.create_all()
        seed_database(app)

    # ----- Route'ları kaydet -----
    _register_routes(app)

    # ----- Zamanlayıcıyı başlat -----
    init_scheduler(app, socketio)

    return app


# ===========================================================================
# ROUTE TANIMLARI
# ===========================================================================


def _register_routes(app: Flask):
    """Tüm API ve sayfa route'larını kaydet."""

    # -------------------------------------------------------------------
    # SAYFA ROUTE'LARI
    # -------------------------------------------------------------------

    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/predictions")
    def predictions():
        return render_template("predictions.html")

    @app.route("/analytics")
    def analytics():
        return render_template("analytics.html")

    @app.route("/system_overview")
    def system_overview():
        return render_template("system_overview.html")

    @app.route("/buildings")
    def buildings_page():
        return render_template("buildings.html")

    @app.route("/settings")
    def settings():
        return render_template("settings.html")

    # -------------------------------------------------------------------
    # DASHBOARD
    # -------------------------------------------------------------------

    @app.route("/api/dashboard/summary")
    def dashboard_summary():
        """Anlık KPI özeti: toplam tüketim, tasarruf, öneri sayısı, R²."""
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=1)

            # Son 1 saatlik toplam tüketim
            recent = (
                EnergyReading.query
                .filter(EnergyReading.timestamp >= cutoff)
                .all()
            )
            total_current_kwh = sum(r.total_kwh for r in recent)

            # Bu ayki toplam tüketim
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            monthly = (
                EnergyReading.query
                .filter(EnergyReading.timestamp >= month_start)
                .all()
            )
            monthly_kwh = sum(r.total_kwh for r in monthly)

            # Aktif öneri sayısı
            pending_count = OptimizationRecommendation.query.filter_by(
                status="pending"
            ).count()

            # Uygulanan önerilerden tasarruf
            applied = OptimizationRecommendation.query.filter_by(status="applied").all()
            total_saved_kwh = sum(r.estimated_saving_kwh or 0 for r in applied)

            # Kampüs geneli tasarruf potansiyeli
            campus_savings = calculate_campus_wide_savings()

            # Ortalama model R²
            buildings = Building.query.all()
            r2_scores = []
            for b in buildings:
                try:
                    acc = get_model_accuracy(b.id)
                    if "r2" in acc:
                        r2_scores.append(acc["r2"])
                except Exception:
                    pass
            avg_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else 0.0

            # Karbon emisyon hesabı (1 kWh = 0.43 kg CO₂)
            CO2_FACTOR = 0.43
            co2_saved_kg = total_saved_kwh * CO2_FACTOR
            monthly_co2_kg = monthly_kwh * CO2_FACTOR
            potential_co2_reduction_kg = campus_savings.get("total_co2_saving_kg", 0)

            return jsonify({
                "total_current_kwh": round(total_current_kwh, 2),
                "monthly_kwh": round(monthly_kwh, 2),
                "monthly_cost_tl": round(monthly_kwh * 3.50, 2),
                "monthly_co2_kg": round(monthly_co2_kg, 2),
                "potential_co2_reduction_kg": round(potential_co2_reduction_kg, 2),
                "pending_recommendations": pending_count,
                "total_saved_kwh": round(total_saved_kwh, 2),
                "co2_saved_kg": round(co2_saved_kg, 2),
                "campus_savings": campus_savings,
                "avg_model_r2": round(avg_r2, 4),
                "building_count": len(buildings),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # BİNALAR
    # -------------------------------------------------------------------

    @app.route("/api/buildings")
    def api_buildings():
        """Tüm binaları listele."""
        try:
            buildings = Building.query.all()
            return jsonify({
                "buildings": [b.to_dict() for b in buildings],
                "count": len(buildings),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/buildings", methods=["POST"])
    def api_add_building():
        """Yeni bina ekle."""
        try:
            data = request.get_json()
            if not data or "name" not in data or "type" not in data:
                return jsonify({"error": "name ve type alanları gerekli."}), 400

            building = Building(
                name=data["name"],
                type=data["type"],
                floor_area=data.get("floor_area"),
                construction_year=data.get("construction_year"),
                capacity=data.get("capacity"),
            )
            db.session.add(building)
            db.session.commit()

            return jsonify({"building": building.to_dict()}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # ENERJİ VERİLERİ
    # -------------------------------------------------------------------

    @app.route("/api/energy/realtime")
    def api_energy_realtime():
        """Son 1 saatlik tüketim (tüm binalar)."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=1)
            readings = (
                EnergyReading.query
                .filter(EnergyReading.timestamp >= cutoff)
                .order_by(EnergyReading.timestamp.desc())
                .all()
            )
            return jsonify({
                "readings": [r.to_dict() for r in readings],
                "count": len(readings),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/energy/history")
    def api_energy_history():
        """Enerji geçmişi. Query params: building_id (gerekli), days (varsayılan 7)."""
        try:
            building_id = request.args.get("building_id", type=int)
            days = request.args.get("days", 7, type=int)

            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            cutoff = datetime.utcnow() - timedelta(days=days)
            readings = (
                EnergyReading.query
                .filter(
                    EnergyReading.building_id == building_id,
                    EnergyReading.timestamp >= cutoff,
                )
                .order_by(EnergyReading.timestamp)
                .all()
            )

            # Hava verilerini timestamp'e göre eşle
            weather_records = (
                WeatherData.query
                .filter(WeatherData.timestamp >= cutoff)
                .all()
            )
            temp_map = {w.timestamp.strftime("%Y-%m-%d %H"): w.temperature
                        for w in weather_records}

            result = []
            for r in readings:
                rd = r.to_dict()
                ts_key = r.timestamp.strftime("%Y-%m-%d %H")
                rd["temperature"] = temp_map.get(ts_key)
                result.append(rd)

            return jsonify({
                "building_id": building_id,
                "days": days,
                "readings": result,
                "count": len(result),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # HAVA DURUMU
    # -------------------------------------------------------------------

    @app.route("/api/weather/current")
    def api_weather_current():
        """Güncel hava durumu (en son kayıt)."""
        try:
            latest = (
                WeatherData.query
                .order_by(WeatherData.timestamp.desc())
                .first()
            )
            if latest is None:
                return jsonify({"error": "Hava durumu verisi bulunamadı."}), 404

            return jsonify(latest.to_dict())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/weather/forecast")
    def api_weather_forecast():
        """Son 24 saatlik hava durumu verileri."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            data = (
                WeatherData.query
                .filter(WeatherData.timestamp >= cutoff)
                .order_by(WeatherData.timestamp)
                .all()
            )
            return jsonify({
                "forecast": [w.to_dict() for w in data],
                "count": len(data),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # TAHMİNLER (ML)
    # -------------------------------------------------------------------

    @app.route("/api/predictions/24h")
    def api_predictions_24h():
        """24 saatlik enerji tahmini. Query param: building_id (gerekli)."""
        try:
            building_id = request.args.get("building_id", type=int)
            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            building = Building.query.get(building_id)
            if building is None:
                return jsonify({"error": f"Bina bulunamadı: id={building_id}"}), 404

            predictions = predict_next_24h(building_id)
            return jsonify({
                "building_id": building_id,
                "building_name": building.name,
                "predictions": predictions,
                "count": len(predictions),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/predictions/7d")
    def api_predictions_7d():
        """7 günlük enerji tahmini. Query param: building_id (gerekli)."""
        try:
            building_id = request.args.get("building_id", type=int)
            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            building = Building.query.get(building_id)
            if building is None:
                return jsonify({"error": f"Bina bulunamadı: id={building_id}"}), 404

            predictions = predict_next_7d(building_id)
            return jsonify({
                "building_id": building_id,
                "building_name": building.name,
                "predictions": predictions,
                "count": len(predictions),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # MODEL DOĞRULUĞU
    # -------------------------------------------------------------------

    @app.route("/api/model/accuracy")
    def api_model_accuracy():
        """Model performans metrikleri. Query param: building_id (opsiyonel)."""
        try:
            building_id = request.args.get("building_id", type=int)

            if building_id is not None:
                building = Building.query.get(building_id)
                if building is None:
                    return jsonify({"error": f"Bina bulunamadı: id={building_id}"}), 404

                accuracy = get_model_accuracy(building_id)
                return jsonify({
                    "building_id": building_id,
                    "building_name": building.name,
                    "metrics": accuracy,
                })
            else:
                # Tüm binaların metrikleri
                buildings = Building.query.all()
                all_metrics = {}
                for b in buildings:
                    try:
                        all_metrics[b.id] = {
                            "name": b.name,
                            "metrics": get_model_accuracy(b.id),
                        }
                    except Exception:
                        all_metrics[b.id] = {"name": b.name, "error": "Model eğitilmemiş."}

                return jsonify({"buildings": all_metrics})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/model/retrain", methods=["POST"])
    def api_model_retrain():
        """Modelleri yeniden eğit. Body: {building_id} veya tümü."""
        try:
            data = request.get_json() or {}
            building_id = data.get("building_id")

            if building_id:
                metrics = train_model(building_id, force_retrain=True)
                return jsonify({"building_id": building_id, "metrics": metrics})
            else:
                results = train_all_models()
                return jsonify({"results": {str(k): v for k, v in results.items()}})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/model/feature-importance")
    def api_feature_importance():
        """Feature importance. Query param: building_id (gerekli)."""
        try:
            building_id = request.args.get("building_id", type=int)
            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            building = Building.query.get(building_id)
            if building is None:
                return jsonify({"error": f"Bina bulunamadı: id={building_id}"}), 404

            importance = get_feature_importance(building_id)
            return jsonify({
                "building_id": building_id,
                "building_name": building.name,
                "features": importance,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/model/analysis")
    def api_model_analysis():
        """Gerçek vs Tahmin hata analizi. Query: building_id (gerekli), hours (opsiyonel)."""
        try:
            building_id = request.args.get("building_id", type=int)
            hours = request.args.get("hours", default=24, type=int)
            
            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            analysis = get_prediction_analysis(building_id, hours=hours)
            return jsonify({
                "building_id": building_id,
                "analysis": analysis
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # EXPLAINABLE AI
    # -------------------------------------------------------------------

    @app.route("/api/model/explain")
    def api_model_explain():
        """Grup bazlı feature importance (XAI). Query param: building_id."""
        try:
            building_id = request.args.get("building_id", type=int)
            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            building = Building.query.get(building_id)
            if building is None:
                return jsonify({"error": f"Bina bulunamadı: id={building_id}"}), 404

            from modules.ml_model import get_grouped_feature_importance
            grouped = get_grouped_feature_importance(building_id)
            return jsonify({
                "building_id": building_id,
                "building_name": building.name,
                **grouped,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # OPTİMİZASYON ÖNERİLERİ
    # -------------------------------------------------------------------

    @app.route("/api/recommendations")
    def api_recommendations():
        """Optimizasyon önerileri. Query params: building_id (opsiyonel), status (opsiyonel)."""
        try:
            building_id = request.args.get("building_id", type=int)
            status = request.args.get("status")

            query = OptimizationRecommendation.query

            if building_id is not None:
                query = query.filter_by(building_id=building_id)
            if status is not None:
                query = query.filter_by(status=status)

            recs = query.order_by(OptimizationRecommendation.created_at.desc()).all()

            return jsonify({
                "recommendations": [r.to_dict() for r in recs],
                "count": len(recs),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/recommendations/generate", methods=["POST"])
    def api_generate_recommendations():
        """Yeni öneriler üret. Body: {building_id} veya tümü."""
        try:
            data = request.get_json() or {}
            building_id = data.get("building_id")

            if building_id:
                result = generate_recommendations(building_id)
                return jsonify(result)
            else:
                results = generate_all_recommendations()
                return jsonify({
                    "results": {str(k): v for k, v in results.items()},
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/recommendations/<int:rec_id>/apply", methods=["POST"])
    def api_apply_recommendation(rec_id):
        """Öneriyi uygula (status → applied)."""
        try:
            rec = OptimizationRecommendation.query.get(rec_id)
            if rec is None:
                return jsonify({"error": f"Öneri bulunamadı: id={rec_id}"}), 404

            rec.status = "applied"
            db.session.commit()

            return jsonify({
                "message": f"Öneri #{rec_id} uygulandı.",
                "recommendation": rec.to_dict(),
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/recommendations/<int:rec_id>/dismiss", methods=["POST"])
    def api_dismiss_recommendation(rec_id):
        """Öneriyi reddet (status → dismissed)."""
        try:
            rec = OptimizationRecommendation.query.get(rec_id)
            if rec is None:
                return jsonify({"error": f"Öneri bulunamadı: id={rec_id}"}), 404

            rec.status = "dismissed"
            db.session.commit()

            return jsonify({
                "message": f"Öneri #{rec_id} reddedildi.",
                "recommendation": rec.to_dict(),
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # ANOMALİLER
    # -------------------------------------------------------------------

    @app.route("/api/anomalies")
    def api_anomalies():
        """Anomali tespiti. Query params: building_id (gerekli), threshold (varsayılan 2.5)."""
        try:
            building_id = request.args.get("building_id", type=int)
            threshold = request.args.get("threshold", 2.5, type=float)

            if building_id is None:
                return jsonify({"error": "building_id parametresi gerekli."}), 400

            building = Building.query.get(building_id)
            if building is None:
                return jsonify({"error": f"Bina bulunamadı: id={building_id}"}), 404

            anomalies = detect_anomalies(building_id, threshold=threshold)
            return jsonify({
                "building_id": building_id,
                "building_name": building.name,
                "threshold": threshold,
                "anomalies": anomalies,
                "count": len(anomalies),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # SENARYO SİMÜLASYONU
    # -------------------------------------------------------------------

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        """Ne olurdu senaryosu. Body: {building_id, hvac_reduction_percent, ...}."""
        try:
            data = request.get_json()
            if not data or "building_id" not in data:
                return jsonify({"error": "building_id gerekli."}), 400

            result = simulate_optimization_scenario(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------
    # CSV EXPORT
    # -------------------------------------------------------------------

    @app.route("/api/export/report")
    def api_export_report():
        """CSV rapor indirme. Query params: building_id, days (varsayılan 30)."""
        try:
            import csv
            import io

            building_id = request.args.get("building_id", type=int)
            days = request.args.get("days", 30, type=int)

            cutoff = datetime.utcnow() - timedelta(days=days)

            query = EnergyReading.query.filter(EnergyReading.timestamp >= cutoff)
            if building_id:
                query = query.filter_by(building_id=building_id)

            readings = query.order_by(EnergyReading.timestamp).all()

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "id", "building_id", "timestamp", "electricity_kwh",
                "heating_kwh", "cooling_kwh", "total_kwh", "occupancy_rate",
            ])
            for r in readings:
                writer.writerow([
                    r.id, r.building_id, r.timestamp.isoformat(),
                    r.electricity_kwh, r.heating_kwh, r.cooling_kwh,
                    r.total_kwh, r.occupancy_rate,
                ])

            from flask import Response
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename=energy_report_{days}d.csv"},
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Doğrudan çalıştırma
# ---------------------------------------------------------------------------
# Gunicorn import için modül seviyesinde app oluştur
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)
