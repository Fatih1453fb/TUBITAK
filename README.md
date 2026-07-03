# ⚡ Kampüs Enerji Optimizasyon Sistemi (EnerjiOS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Socket.IO](https://img.shields.io/badge/Socket.IO-Real--Time-010101?style=for-the-badge&logo=socket.io&logoColor=white)
![License](https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge)

**Üniversite kampüsleri için yapay zeka destekli enerji tüketim tahmini, anomali tespiti ve optimizasyon platformu.**

[Özellikler](#-özellikler) • [Test Sonuçları](#-test-sonuçları) • [Kurulum](#-kurulum) • [Kullanım](#-kullanım) • [Mimari](#-sistem-mimarisi) • [Teknolojiler](#-teknolojiler)

</div>

---

## 📋 Proje Hakkında

EnerjiOS, üniversite kampüslerindeki binaların enerji tüketimini **gerçek zamanlı** olarak izleyen, **makine öğrenmesi** ile gelecekteki tüketimi tahmin eden ve **otomatik optimizasyon stratejileri** üreten kapsamlı bir web platformudur.

Sistem, 5 farklı bina (A Blok Laboratuvarları, B Blok Derslikler, C Blok Ofisler, Kütüphane, Spor Merkezi) için:
- 📊 **Saatlik/günlük enerji tüketim tahmini** yapar
- 🔍 **Anomali tespiti** ile anormal tüketim kalıplarını yakalar
- 💡 **Doluluk, tarife ve hava durumuna göre** tasarruf önerileri üretir
- 🌱 **Karbon ayak izi hesaplaması** ile çevresel etkiyi ölçer

> 🎓 Bu proje, **TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destek Programı** kapsamında desteklenmiştir.
> *Proje Yürütücüsü: Fatih Kuruçay · Danışman: Dr. Öğr. Üyesi Peren Jerfi Canatalay · İstinye Üniversitesi*

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
- İnteraktif Chart.js grafikleri

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
- CO₂ emisyon hesaplaması (0,43 kg/kWh — Türkiye şebeke ortalaması)
- Potansiyel azaltım miktarı
- Kampüs bazında çevresel raporlama

---

## 📊 Test Sonuçları

Karşılaştırmalı model değerlendirmesinde üç algoritma test edilmiştir: **Doğrusal Regresyon** (Model A, baseline), **Karar Ağacı** (Model B) ve **Random Forest** (Model C). Time Series Cross-Validation ve hiperparametre optimizasyonu sonucunda en kararlı genelleme performansını Random Forest göstermiş ve sistemin tahmin katmanı olarak seçilmiştir. Karşılaştırma `compare_models.py` scripti ile yeniden üretilebilir.

### Tahmin Modeli Performansı (Random Forest — bina bazlı test sonuçları)

| Bina | Test R² | MAE (kWh) | RMSE (kWh) | Değerlendirme |
|------|---------|-----------|------------|---------------|
| A Blok Laboratuvarları | 0,8280 | 3,08 | 3,88 | Kararlı |
| B Blok Derslikler | 0,8570 | 3,50 | 4,43 | Kararlı |
| C Blok Ofisler | 0,8353 | 1,17 | 1,46 | Kararlı |
| Kütüphane | 0,7855 | 0,67 | 0,84 | Hafif aşırı öğrenme |
| Spor Merkezi | 0,7639 | 1,18 | 1,45 | Hafif aşırı öğrenme |
| **Ortalama** | **0,8139** | — | — | — |

### Optimizatör Backtesting (30 Gün)

| Metrik | Değer |
|--------|-------|
| Toplam gerçek tüketim | 145.732 kWh |
| Simüle edilmiş tüketim | 135.975 kWh |
| Potansiyel tasarruf | **9.756 kWh (%6,7)** |
| CO₂ azaltımı | **4.195 kg** |

### Anomali Tespiti (Z-skoru, eşik = 2,5)

| Metrik | Ortalama Değer |
|--------|----------------|
| Kesinlik (Precision) | 0,94 |
| Duyarlılık (Recall) | 0,31 |
| F1 Skoru | 0,47 |

> Yüksek kesinlik öncelikli, muhafazakâr tespit yaklaşımı benimsenmiştir. HVAC duyarlılık analizleri (%5–%20 kısıtlama senaryoları) ve bulguların ayrıntılı tartışması için TÜBİTAK 2209-A proje sonuç raporuna bakınız.

---

## 🏗 Sistem Mimarisi

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  Veri Toplama  │───▶│  Zamanlayıcı   │───▶│    ML Model    │
│ (Web API'leri) │    │  (APScheduler) │    │ (Random Forest)│
└────────────────┘    └────────────────┘    └────────┬───────┘
                                                     │
┌────────────────┐    ┌────────────────┐    ┌────────▼───────┐
│   Dashboard    │◀───│   Uyarılar &   │◀───│  Optimizatör   │
│(Flask+Chart.js)│    │   WebSocket    │    │   (Strateji    │
│                │    │  (Socket.IO)   │    │    Üretici)    │
└────────────────┘    └────────────────┘    └────────────────┘
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
- Python 3.11 veya üzeri
- pip (Python paket yöneticisi)

### Adımlar

```bash
# 1. Repoyu klonlayın
git clone https://github.com/Fatih1453fb/TUBITAK.git
cd TUBITAK

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

> **Not:** İlk çalıştırmada veritabanı otomatik oluşturulur ve örnek veriler seed edilir; bu, platformun herkes tarafından hemen çalıştırılıp incelenebilmesi (demo) içindir. Proje kapsamındaki asıl veri seti, web kaynaklarından (açık meteoroloji API'leri, kampüs bilgi sistemleri, enerji platformları) 5 dakikalık aralıklarla toplanmıştır. ML modelleri ilk çalıştırmada otomatik olarak eğitilir.

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
| **Grafikler** | Chart.js |
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

**Fatih Kuruçay**
- GitHub: [@Fatih1453fb](https://github.com/Fatih1453fb)

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır.
