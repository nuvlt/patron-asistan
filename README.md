# 📊 Patron Dijital Asistan - MVP

Finansal Excel verilerini analiz eden ve CFO perspektifinden yorum üreten araç.

## 🎯 Özellikler

✅ Excel dosyası yükleme ve analiz  
✅ Otomatik trend analizi (son 3 ay)  
✅ **Prophet ile akıllı forecast** (Facebook'un AI algoritması)  
✅ **İnteraktif grafikler** (Recharts ile)  
✅ Risk seviyesi belirleme  
✅ Claude için hazır prompt oluşturma  

## 🏗️ Mimari

- **Frontend**: Next.js 14 (Vercel'de ücretsiz deploy)
- **Backend**: FastAPI (Render free tier)
- **Analiz**: pandas + numpy
- **Forecast**: Prophet (Facebook)
- **Grafik**: Recharts (React) + Plotly (backend)
- **AI Yorum**: Claude.ai (manuel veya API)

## 🚀 Kurulum

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend şu adreste çalışır: http://localhost:8000

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Frontend şu adreste çalışır: http://localhost:3000

## 📝 Kullanım

1. **Test verisi oluştur** (opsiyonel):
```bash
python create_sample_data.py
```

2. **Frontend'i aç**: http://localhost:3000

3. **Excel dosyasını yükle** (tarih + sayısal kolonlar içermeli)

4. **Analiz Et** butonuna bas

5. **Claude prompt'u kopyala** ve Claude.ai'a yapıştır

## 📦 Deploy

### Backend (Render)

1. GitHub'a push yap
2. Render.com'da "New Web Service"
3. Repo'yu bağla
4. Build: `pip install -r requirements.txt`
5. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel)

```bash
cd frontend
vercel
```

veya GitHub'a push yap, Vercel otomatik deploy eder.

**Önemli**: Frontend'de `.env.local` dosyası oluştur:
```
NEXT_PUBLIC_API_URL=https://your-render-app.onrender.com
```

## 🧪 Test

Örnek Excel formatı:
```
Tarih       | Gelir   | Gider  | Net_Kar
2024-01-01  | 50000   | 35000  | 15000
2024-01-08  | 52000   | 36000  | 16000
...
```

## 🔄 Roadmap (Gelecek Özellikler)

- [ ] Claude API entegrasyonu (otomatik yorum)
- [x] ~~Prophet ile gelişmiş forecast~~ ✅ EKLENDİ
- [x] ~~Grafik görselleştirme~~ ✅ EKLENDİ
- [ ] PDF rapor export
- [ ] Çoklu kolon karşılaştırma
- [ ] Kullanıcı girişi ve veri saklama

## 🧠 Prophet Nedir?

Facebook tarafından geliştirilen açık kaynak zaman serisi tahmin kütüphanesi:
- Mevsimselliği otomatik tespit eder
- Trend değişimlerini yakalar
- Tatil ve özel günleri hesaba katar
- %95 güven aralığı verir
- **Tamamen ücretsiz!**

## 📄 Lisans

MIT

## 🤝 Katkıda Bulunma

Pull request'ler açıktır!
