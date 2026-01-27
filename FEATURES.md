# 🎯 Patron Dijital Asistan - Özellikler

## ✨ Temel Özellikler

### 1. 📊 Excel Analizi
- **Otomatik kolon tespiti**: Tarih ve sayısal kolonları bulur
- **Veri validasyonu**: En az 3 veri noktası kontrolü
- **Trend analizi**: Son 3 ayın yükseliş/düşüş/sabit trendini hesaplar
- **Risk skoru**: Volatiliteye göre düşük/orta/yüksek risk atar

### 2. 🔮 Prophet ile Tahmin
**Neden Prophet?**
- Facebook tarafından geliştirildi (production-ready)
- Mevsimsellik otomatiği (haftalık, yıllık)
- Trend değişim noktalarını (changepoints) bulur
- Tatil efektlerini modelleyebilir
- %95 güven aralığı verir

**Nasıl Çalışır?**
```python
Prophet(
    daily_seasonality=False,      # Günlük mevsimsellik yok
    weekly_seasonality=True,      # Haftalık var (örn: hafta sonu düşüşü)
    yearly_seasonality=True,      # Yıllık var (örn: sezon)
    changepoint_prior_scale=0.05  # Trend değişim hassasiyeti
)
```

**Çıktılar:**
- 4 aylık tahmin
- Alt sınır (pessimistic)
- Üst sınır (optimistic)
- En olası değer (yhat)

### 3. 📈 İnteraktif Grafikler
**Frontend (Recharts):**
- Area chart (alan grafiği)
- Responsive (mobilde de çalışır)
- Tooltip hover ile detay
- Gerçekleşen vs Tahmin karşılaştırması

**Renk Kodları:**
- 🔵 Mavi: Geçmiş veriler
- 🟢 Yeşil (kesikli): Prophet tahmini
- ⚫ Gri (noktalı): Güven aralığı

### 4. 🤖 Claude CFO Yorumu
**Otomatik Prompt Oluşturma:**
```
Sen deneyimli bir CFO'sun.
Analiz Özeti:
- Kolon: Gelir
- Trend: yükseliş (%12.5)
- Risk: orta
- Tahmin: [veri]

1. Genel Gidişat
2. Risk Durumu
3. 30-60-90-120 gün önerileri
```

Kullanıcı kopyalayıp Claude.ai'a yapıştırır.

## 🔧 Teknik Detaylar

### Backend Stack
```
FastAPI     : REST API
Pandas      : Veri işleme
NumPy       : Matematiksel hesaplamalar
Prophet     : Time-series forecast
Plotly      : Grafik (backend tarafında)
```

### Frontend Stack
```
Next.js 14  : React framework
Recharts    : Grafik kütüphanesi
TailwindCSS : Styling
```

### Algoritma Detayları

**Trend Hesaplama:**
```python
first_half_avg = mean(values[:len/2])
second_half_avg = mean(values[len/2:])
trend_pct = (second_half - first_half) / first_half * 100

if trend_pct > 5:  trend = "yükseliş"
elif trend_pct < -5: trend = "düşüş"
else: trend = "sabit"
```

**Risk Skoru:**
```python
volatility = std(values) / mean(values)

if volatility > 0.3: risk = "yüksek"
elif volatility > 0.15: risk = "orta"
else: risk = "düşük"
```

## 💰 Maliyet Analizi

### Ücretsiz Tier Limitleri

**Render (Backend):**
- ✅ 750 saat/ay (24/7 çalışır)
- ✅ 512 MB RAM (Prophet çalışır)
- ⚠️ Cold start (15 dk hareketsizlikten sonra)
- ✅ Otomatik HTTPS

**Vercel (Frontend):**
- ✅ 100 GB bandwidth
- ✅ Sınırsız request
- ✅ Auto-scaling
- ✅ Edge network (hızlı)

**Toplam Maliyet: 0₺/ay** 🎉

### Ölçeklenme Stratejisi

**Ücretsiz tier yeterli mi?**

| Kullanıcı/Gün | Analiz/Gün | Backend Yük | Sonuç |
|---------------|------------|-------------|-------|
| 10 | 50 | Minimal | ✅ Yeterli |
| 100 | 500 | Düşük | ✅ Yeterli |
| 1000 | 5000 | Orta | ⚠️ Cold start sıkıntı |
| 10000+ | 50000+ | Yüksek | ❌ Ücretli plan gerek |

**Cold Start Sorunu:**
- Render free tier 15 dk sonra uyur
- İlk istek 30-60 saniye sürer
- Çözüm: Cron job ile 10 dk'da bir ping at

## 🚀 Performans

### Prophet Model Eğitim Süresi
- 50 veri noktası: ~3 saniye
- 100 veri noktası: ~5 saniye
- 365 veri noktası: ~10 saniye

### API Response Time
```
Excel Upload      : <1 saniye
Prophet Forecast  : 3-10 saniye (veri boyutuna göre)
Total Response    : 5-15 saniye
```

### Frontend Render
```
Grafik Render     : <100ms (Recharts)
Page Load         : <500ms (Vercel CDN)
```

## 🔒 Güvenlik

### Veri Güvenliği
- ✅ Dosyalar RAM'de işlenir (disk'e yazılmaz)
- ✅ İşlem bitince hemen silinir
- ✅ HTTPS zorunlu (Render + Vercel)
- ❌ Henüz authentication yok (MVP)

### CORS
```python
allow_origins=["*"]  # MVP için tüm originler
# Prod'da: ["https://your-domain.com"]
```

## 📊 Örnek Kullanım Senaryoları

### 1. Gelir Tahmini
**Input:** 1 yıllık haftalık gelir verisi  
**Output:** 4 aylık tahmin + trend + risk  
**Kullanım:** Bütçe planlama

### 2. Gider Kontrolü
**Input:** Aylık operasyonel giderler  
**Output:** Hangi ay yükselecek?  
**Kullanım:** Maliyet optimizasyonu

### 3. Nakit Akışı
**Input:** Günlük banka bakiyesi  
**Output:** 120 gün sonra nakit durumu  
**Kullanım:** Likidite yönetimi

## 🎨 UI/UX İyileştirme Fikirleri

### Şu An
- ✅ Responsive design
- ✅ Drag & drop
- ✅ Renkli risk göstergeleri
- ✅ İnteraktif grafik

### Gelecek
- [ ] Dark mode
- [ ] Çoklu dil (EN/TR)
- [ ] Karşılaştırma modu (bu ay vs geçen ay)
- [ ] PDF export
- [ ] Email rapor gönderme

## 🐛 Bilinen Sınırlamalar

1. **Veri Kalitesi**: Eksik veriler hata verir (şimdilik)
2. **Cold Start**: İlk istek yavaş (Render free tier)
3. **Prophet Hız**: Büyük veri setleri (1000+ satır) yavaş
4. **Hafıza**: Render 512 MB limit (çok büyük Excel crash)
5. **Manuel Claude**: API entegrasyonu yok (şimdilik)

## 📚 Referanslar

- [Prophet Docs](https://facebook.github.io/prophet/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Recharts Gallery](https://recharts.org/en-US/)
- [Vercel Deploy Guide](https://vercel.com/docs)
- [Render Free Tier](https://render.com/docs/free)
