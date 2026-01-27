# 🚀 Deployment Rehberi

## Backend (Render.com - ÜCRETSİZ)

### 1. GitHub Repo Oluştur
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/KULLANICI_ADIN/patron-asistan.git
git push -u origin main
```

### 2. Render'da Deploy

1. https://render.com → "New Web Service"
2. GitHub repo'nu bağla
3. Ayarlar:
   - **Name**: patron-asistan-api
   - **Root Directory**: `backend`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

4. "Create Web Service" → Deploy başlar (5-10 dk)

5. URL'i kopyala: `https://patron-asistan-api.onrender.com`

### ⚠️ Render Free Tier Notları:
- İlk istek yavaş olabilir (cold start)
- 15 dk hareketsizlikten sonra uyur
- Ayda 750 saat ücretsiz (yeterli)

---

## Frontend (Vercel - ÜCRETSİZ)

### 1. Vercel CLI ile Deploy

```bash
cd frontend
npm install -g vercel
vercel login
vercel
```

### 2. Environment Variable Ayarla

Vercel dashboard → Project → Settings → Environment Variables

```
NEXT_PUBLIC_API_URL = https://patron-asistan-api.onrender.com
```

### 3. Redeploy
```bash
vercel --prod
```

**VEYA GitHub ile Otomatik Deploy:**

1. GitHub'a push
2. https://vercel.com/new
3. Import repo
4. Environment variable ekle
5. Deploy

---

## ✅ Test Etme

1. Frontend URL'ine git: `https://patron-asistan.vercel.app`
2. Örnek Excel dosyasını yükle
3. "Analiz Et" butonuna tıkla
4. Claude prompt'u kopyala → Claude.ai'a yapıştır

---

## 🔧 Sorun Giderme

### Backend çalışmıyor:
```bash
# Render logs'u kontrol et
# Dashboard → Service → Logs
```

### Frontend API'ye bağlanamıyor:
```bash
# .env.local dosyasını kontrol et
# CORS hatası varsa backend'de allow_origins ayarını kontrol et
```

### CORS hatası:
Backend `main.py` dosyasında:
```python
allow_origins=["https://your-frontend.vercel.app"]
```

---

## 💰 Maliyetler

- **Render Free**: 0₺/ay (750 saat)
- **Vercel Hobby**: 0₺/ay (100GB bandwidth)
- **Toplam**: 0₺/ay ✅

---

## 📈 Sonraki Adımlar

1. ✅ MVP deploy edildi
2. 🔄 Kullanıcı feedback'i topla
3. 📊 Grafik özelliği ekle
4. 🤖 Claude API entegre et
5. 💳 Ücretli plan için Stripe ekle
