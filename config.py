"""
Kampüs Enerji Optimizasyon Sistemi — Yapılandırma
==================================================
Flask uygulama ayarları ve veritabanı yapılandırması.
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Temel yapılandırma sınıfı."""

    # Güvenlik
    SECRET_KEY = os.environ.get("SECRET_KEY", "kampus-enerji-gizli-anahtar-2026")

    # Veritabanı — SQLite
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'campus_energy.db')}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Geliştirme
    DEBUG = True

    # Kampüs bilgileri
    CAMPUS_NAME = os.environ.get("CAMPUS_NAME", "İstinye Üniversitesi")
    OPENMETEO_LAT = float(os.environ.get("OPENMETEO_LAT", 41.0082))
    OPENMETEO_LON = float(os.environ.get("OPENMETEO_LON", 28.9784))
