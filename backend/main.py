from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json
from typing import Optional
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Patron Dijital Asistan API")

# CORS ayarları (Vercel frontend için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prod'da spesifik domain yapılabilir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_excel(df: pd.DataFrame, target_column: str):
    """Excel verisini analiz et"""
    
    # Tarih kolonu bul
    date_col = None
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'tarih' in col.lower() or 'date' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        # İlk kolonu tarih olarak dene
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            date_col = df.columns[0]
        except:
            raise HTTPException(status_code=400, detail="Tarih kolonu bulunamadı")
    
    # Veriyi sırala
    df = df.sort_values(by=date_col)
    
    # Son 3 ay analizi
    latest_date = df[date_col].max()
    three_months_ago = latest_date - timedelta(days=90)
    recent_data = df[df[date_col] >= three_months_ago]
    
    if len(recent_data) < 3:
        return {
            "success": False,
            "message": "Yeterli veri yok (en az 3 veri noktası gerekli)",
            "can_forecast": False
        }
    
    # Trend analizi
    values = recent_data[target_column].values
    trend = "sabit"
    trend_pct = 0
    
    if len(values) >= 2:
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if first_half > 0:
            trend_pct = ((second_half - first_half) / first_half) * 100
            
            if trend_pct > 5:
                trend = "yükseliş"
            elif trend_pct < -5:
                trend = "düşüş"
    
    # Risk seviyesi belirleme
    volatility = np.std(values) / (np.mean(values) + 0.0001)
    
    if volatility > 0.3:
        risk_level = "yüksek"
    elif volatility > 0.15:
        risk_level = "orta"
    else:
        risk_level = "düşük"
    
    return {
        "success": True,
        "date_range": f"{recent_data[date_col].min().strftime('%Y-%m-%d')} - {recent_data[date_col].max().strftime('%Y-%m-%d')}",
        "trend": trend,
        "trend_percentage": round(trend_pct, 2),
        "risk_level": risk_level,
        "data_points": len(recent_data),
        "average_value": round(np.mean(values), 2),
        "can_forecast": len(df) >= 10  # En az 10 veri noktası
    }

def prophet_forecast(df: pd.DataFrame, target_column: str, date_col: str, periods: int = 4):
    """Prophet ile gelişmiş forecast (yedek: sklearn)"""
    
    try:
        # Prophet formatına dönüştür
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_column]
        })
        
        # Model oluştur ve eğit
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True if len(prophet_df) > 365 else False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        # Gelecek tahminleri
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        # Son periods kadar tahmin al
        future_forecast = forecast.tail(periods)
        
        forecasts = []
        for idx, row in future_forecast.iterrows():
            forecasts.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "value": round(max(0, row['yhat']), 2),
                "lower": round(max(0, row['yhat_lower']), 2),
                "upper": round(max(0, row['yhat_upper']), 2),
                "month": len(forecasts) + 1
            })
        
        # Grafik verisi hazırla
        historical = prophet_df.tail(12)
        
        chart_data = {
            "historical": {
                "dates": historical['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "values": historical['y'].tolist()
            },
            "forecast": {
                "dates": [f['date'] for f in forecasts],
                "values": [f['value'] for f in forecasts],
                "lower": [f['lower'] for f in forecasts],
                "upper": [f['upper'] for f in forecasts]
            }
        }
        
        return {
            "success": True,
            "forecasts": forecasts,
            "method": "prophet",
            "chart_data": chart_data
        }
    
    except Exception as e:
        # Prophet başarısız olursa sklearn'e düş
        print(f"Prophet failed, falling back to sklearn: {str(e)}")
        return smart_forecast(df, target_column, date_col, periods)

def smart_forecast(df: pd.DataFrame, target_column: str, date_col: str, periods: int = 4):
    """Akıllı forecast (Linear Regression + Seasonality) - Prophet yedek"""
    
    try:
        # Tarihleri sayıya çevir
        df = df.sort_values(by=date_col)
        df['days'] = (df[date_col] - df[date_col].min()).dt.days
        
        # Train data
        X = df[['days']].values
        y = df[target_column].values
        
        # Linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Mevsimsellik tespit (haftalık pattern)
        df['weekday'] = df[date_col].dt.dayofweek
        weekly_avg = df.groupby('weekday')[target_column].mean()
        overall_avg = df[target_column].mean()
        seasonal_factor = (weekly_avg / overall_avg).to_dict()
        
        # Tahminler
        last_date = df[date_col].max()
        last_days = df['days'].max()
        
        forecasts = []
        for i in range(1, periods + 1):
            # Tahmin tarihi
            forecast_date = last_date + timedelta(days=30 * i)
            forecast_days = last_days + (30 * i)
            
            # Temel tahmin
            base_prediction = model.predict([[forecast_days]])[0]
            
            # Mevsimsellik ekle
            weekday = forecast_date.weekday()
            seasonal_adjustment = seasonal_factor.get(weekday, 1.0)
            forecast_value = base_prediction * seasonal_adjustment
            
            # Güven aralığı (basit yaklaşım)
            std_error = np.std(y - model.predict(X))
            confidence_interval = 1.96 * std_error  # %95
            
            forecasts.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "value": round(max(0, forecast_value), 2),
                "lower": round(max(0, forecast_value - confidence_interval), 2),
                "upper": round(max(0, forecast_value + confidence_interval), 2),
                "month": i
            })
        
        # Grafik verisi hazırla
        historical = df.tail(12)
        
        chart_data = {
            "historical": {
                "dates": historical[date_col].dt.strftime('%Y-%m-%d').tolist(),
                "values": historical[target_column].tolist()
            },
            "forecast": {
                "dates": [f['date'] for f in forecasts],
                "values": [f['value'] for f in forecasts],
                "lower": [f['lower'] for f in forecasts],
                "upper": [f['upper'] for f in forecasts]
            }
        }
        
        # Model performans metrikleri
        r2_score = model.score(X, y)
        
        return {
            "success": True,
            "forecasts": forecasts,
            "method": "linear_regression_fallback",
            "r2_score": round(r2_score, 3),
            "chart_data": chart_data
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Forecast yapılamadı: {str(e)}"
        }

@app.get("/")
def root():
    return {
        "message": "Patron Dijital Asistan API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    target_column: Optional[str] = None
):
    """Excel dosyasını analiz et"""
    
    try:
        # Dosyayı oku
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Kolon listesi
        columns = df.columns.tolist()
        
        # Hedef kolon belirlenmemişse, sayısal ilk kolonu al
        if target_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise HTTPException(status_code=400, detail="Sayısal kolon bulunamadı")
            target_column = numeric_cols[0]
        
        # Analiz
        analysis = analyze_excel(df, target_column)
        
        if not analysis["success"]:
            return JSONResponse(content=analysis, status_code=400)
        
        # Forecast yap (eğer yapılabiliyorsa)
        forecast_result = None
        date_col = None
        
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'tarih' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                date_col = df.columns[0]
            except:
                pass
        
        if analysis["can_forecast"] and date_col:
            forecast_result = prophet_forecast(df, target_column, date_col)
        
        # Claude prompt için hazırla
        claude_input = {
            "target_column": target_column,
            "date_range": analysis["date_range"],
            "trend_summary": f"{analysis['trend']} ({analysis['trend_percentage']}%)",
            "forecast_result": forecast_result if forecast_result else "Forecast için yeterli veri yok",
            "risk_level": analysis["risk_level"],
            "average_value": analysis["average_value"],
            "data_points": analysis["data_points"]
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "forecast": forecast_result,
            "columns": columns,
            "target_column": target_column,
            "claude_prompt_data": claude_input
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

@app.post("/interpret")
async def interpret_with_claude(analysis_data: dict):
    """
    Claude API ile yorumlama (opsiyonel)
    MVP'de frontend'den manuel olarak da yapılabilir
    """
    
    # Bu endpoint'i Claude API key'iniz olunca aktive edebilirsiniz
    return {
        "message": "Bu özellik yakında eklenecek",
        "manual_prompt": f"""
Sen deneyimli bir CFO'sun.
Ben sana bir şirketin finansal analiz sonuçlarını veriyorum.

Analiz Özeti:
- Analiz edilen kolon: {analysis_data.get('target_column')}
- Zaman aralığı: {analysis_data.get('date_range')}
- Son 3 ay trendi: {analysis_data.get('trend_summary')}
- 3 aylık tahmin sonucu: {analysis_data.get('forecast_result')}
- Risk seviyesi: {analysis_data.get('risk_level')}

Lütfen şu başlıklarla cevap ver:
1. Genel Gidişat
2. Risk Durumu
3. Önümüzdeki 30-60-90-120 gün için öneriler
"""
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
