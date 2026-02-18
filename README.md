# ⚡ Kampüs Enerji Optimizasyon Sistemi (EnerjiOS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Socket.IO](https://img.shields.io/badge/Socket.IO-Real--Time-010101?style=for-the-badge&logo=socket.io&logoColor=white)
![License](https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge)

**Üniversite kampüsleri için yapay zeka destekli enerji tüketim tahmini, anomali tespiti ve optimizasyon platformu.**

[Özellikler](#-özellikler) • [Kurulum](#-kurulum) • [Kullanım](#-kullanım) • [Mimari](#-sistem-mimarisi) • [Teknolojiler](#-teknolojiler)

</div>

---

## 📋 Proje Hakkında

EnerjiOS, üniversite kampüslerindeki binaların enerji tüketimini **gerçek zamanlı** olarak izleyen, **makine öğrenmesi** ile gelecekteki tüketimi tahmin eden ve **otomatik optimizasyon stratejileri** üreten kapsamlı bir web platformudur.

Sistem, 5 farklı bina (Mühendislik, Kütüphane, Spor Salonu, İdari Bina, Yurt) için:
- 📊 **Saatlik/günlük enerji tüketim tahmini** yapar
- 🔍 **Anomali tespiti** ile anormal tüketim kalıplarını yakalar
- 💡 **Doluluk, tarife ve hava durumuna göre** tasarruf önerileri üretir
- 🌱 **Karbon ayak izi hesaplaması** ile çevresel etkiyi ölçer

---

## ✨ Özellikler

### 🏠 Gösterge Paneli (Dashboard)
- Gerçek zamanlı KPI kartları (toplam tüketim, tasarruf potansiyeli, karbon emisyonu)
- Bina bazında anlık tüketim grafiği
- WebSocket ile canlı veri akışı
- Tahmin vs. gerçek tüketim karşılaştırması

### 📈 Analitik
- Saatlik enerji tüketim ısı haritası (heatmap)
- Bina karşılaştırmalı tüketim analizi
- Trend analizi ve tarihsel veriler
- İnteraktif Plotly grafikleri

### 🤖 AI Tahmin Motoru
- Random Forest algoritması ile enerji tüketim tahmini
- 24 saat ve 7 günlük öngörü
- Explainable AI (XAI) paneli — özellik katkı yüzdeleri
- Model performans metrikleri (R², MAE, RMSE)
- Time Series Cross-Validation ile model doğrulama

### ⚙️ Optimizasyon Motoru
- Doluluk oranına göre enerji optimizasyonu
- Tarife bazlı maliyet stratejileri
- Hava durumuna duyarlı akıllı öneriler
- Backtesting analizi — son 30 günlük geriye dönük simülasyon
- Bina ve kampüs bazında tasarruf hesaplama

### 🔔 Gerçek Zamanlı Uyarılar
- Z-Score ile anomali tespiti
- Eşik aşımı bildirimleri
- Socket.IO ile anlık push bildirimleri

### 🏢 Bina Yönetimi
- 5 bina için detaylı enerji profili
- Bina bazında optimizasyon önerileri
- Karşılaştırmalı performans analizi

### 🌍 Çevresel Etki
- CO₂ emisyon hesaplaması (0.43 kg/kWh)
- Potansiyel azaltım miktarı
- Kampüs bazında çevresel raporlama

---

## 🏗 Sistem Mimarisi

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Veri       │───▶│  Zamanlayıcı │───▶│  ML Model   │
│  Toplama    │    │  (Scheduler) │    │  (Random    │
│  (Sensörler)│    │              │    │   Forest)   │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
┌─────────────┐   ┌──────────────┐     ┌──────▼──────┐
│  Dashboard  │◀──│  Uyarılar &  │◀────│ Optimizatör │
│  (Flask +   │   │  WebSocket   │     │  (Strateji  │
│   Chart.js) │   │  (Socket.IO) │     │   Üretici)  │
└─────────────┘   └──────────────┘     └─────────────┘
```

---

## 📁 Proje Yapısı

```
TÜBİTAK/
├── app.py                  # Flask uygulama (ana sunucu + API endpointleri)
├── config.py               # Yapılandırma ayarları
├── compare_models.py       # Model karşılaştırma scripti
├── requirements.txt        # Python bağımlılıkları
│
├── modules/
│   ├── database.py         # SQLAlchemy veritabanı modelleri ve sorguları
│   ├── ml_model.py         # ML model eğitimi, tahmin, XAI, cross-validation
│   ├── optimizer.py        # Enerji optimizasyon motoru + backtesting
│   └── scheduler.py        # APScheduler görev zamanlayıcısı
│
├── data/
│   └── seed_data.py        # Başlangıç veri oluşturma scripti
│
├── models/                 # Eğitilmiş ML model dosyaları (.pkl)
│
├── templates/
│   ├── base.html           # Ana şablon (sidebar, navbar, tema)
│   ├── dashboard.html      # Gösterge paneli sayfası
│   ├── analytics.html      # Analitik ve raporlama sayfası
│   ├── predictions.html    # AI tahmin ve XAI paneli
│   ├── system_overview.html# Sistem genel bakış ve mimari
│   ├── buildings.html      # Bina yönetimi
│   └── settings.html       # Ayarlar sayfası
│
└── static/
    └── css/
        └── style.css       # Ana stil dosyası (koyu tema)
```

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.10 veya üzeri
- pip (Python paket yöneticisi)

### Adımlar

```bash
# 1. Repoyu klonlayın
git clone https://github.com/Fatih1453fb/T-B-TAK.git
cd T-B-TAK

# 2. Sanal ortam oluşturun
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Uygulamayı başlatın
python3 app.py
```

Uygulama varsayılan olarak **http://127.0.0.1:5001** adresinde çalışır.

> **Not:** İlk çalıştırmada veritabanı otomatik oluşturulur ve örnek veriler seed edilir. ML modelleri de otomatik olarak eğitilir.

---

## 💻 Kullanım

| Sayfa | URL | Açıklama |
|-------|-----|----------|
| Gösterge Paneli | `/` | Ana dashboard, KPI'lar ve gerçek zamanlı grafikler |
| Analitik | `/analytics` | Isı haritası, trend analizi, bina karşılaştırmaları |
| Tahminler | `/predictions` | AI tahmin sonuçları ve XAI açıklamaları |
| Sistem Genel Bakış | `/system_overview` | Sistem mimarisi ve performans metrikleri |
| Binalar | `/buildings` | Bina detayları ve enerji profilleri |
| Ayarlar | `/settings` | Uygulama yapılandırma |

---

## 🛠 Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| **Backend** | Python, Flask, Flask-SocketIO |
| **ML/AI** | Scikit-Learn (Random Forest), Pandas, NumPy |
| **Veritabanı** | SQLite, SQLAlchemy |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 |
| **Grafikler** | Chart.js, Plotly.js |
| **Gerçek Zamanlı** | Socket.IO, WebSocket |
| **Zamanlama** | APScheduler |
| **Deployment** | Gunicorn |

---

## 📊 API Endpointleri

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/api/realtime` | GET | Gerçek zamanlı tüketim verisi |
| `/api/predictions` | GET | AI tahmin sonuçları |
| `/api/comparison` | GET | Bina karşılaştırma verisi |
| `/api/model/explain` | GET | XAI özellik katkı yüzdeleri |
| `/api/optimizer/backtest` | GET | Optimizatör backtesting sonuçları |
| `/api/anomalies` | GET | Tespit edilen anomaliler |

---

## 👤 Geliştirici

**Fatih Kurucay**
- GitHub: [@Fatih1453fb](https://github.com/Fatih1453fb)

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır.
