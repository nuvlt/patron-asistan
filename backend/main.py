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
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Patron Dijital Asistan API v2")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_wide_format(df: pd.DataFrame) -> bool:
    """Excel'in wide format olup olmadığını kontrol et"""
    
    # Çok kolon var mı? (>15 kolon)
    if len(df.columns) <= 15:
        return False
    
    # İlk kolon text, geri kalanlar sayısal mı?
    try:
        first_col_text = df.iloc[:, 0].dtype == 'object'
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        
        return first_col_text and numeric_ratio > 0.7
    except:
        return False

def analyze_wide_format(df: pd.DataFrame) -> dict:
    """
    Wide format analizi
    Satırlarda: Ürün/Oyun/Kategori
    Kolonlarda: Dönemler (aylar, yıllar)
    """
    
    try:
        id_column = df.columns[0]
        items = df[id_column].dropna().tolist()
        
        if len(items) == 0:
            return {
                "success": False,
                "message": "Analiz edilecek öğe bulunamadı"
            }
        
        # En yüksek toplam değeri bul
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                "success": False,
                "message": "Sayısal veri bulunamadı"
            }
        
        # Her satırın toplamını hesapla
        df['_total'] = df[numeric_cols].sum(axis=1, skipna=True)
        max_idx = df['_total'].idxmax()
        target_item = df.iloc[max_idx][id_column]
        
        # Target item'ın verisini al
        target_row = df.iloc[max_idx]
        values = target_row[numeric_cols].values
        
        # NaN ve 0'ları temizle
        clean_values = [float(v) for v in values if pd.notna(v) and v != 0]
        
        if len(clean_values) < 3:
            return {
                "success": False,
                "message": f"{target_item} için yeterli veri yok (min 3 dönem gerekli)"
            }
        
        # Son 12 dönemi al (veya mevcut kadarını)
        recent_count = min(12, len(clean_values))
        recent_values = clean_values[-recent_count:]
        
        # Trend hesapla
        first_half = np.mean(recent_values[:len(recent_values)//2])
        second_half = np.mean(recent_values[len(recent_values)//2:])
        
        if first_half > 0:
            trend_pct = ((second_half - first_half) / first_half) * 100
        else:
            trend_pct = 0
        
        if trend_pct > 10:
            trend = "güçlü yükseliş"
        elif trend_pct > 5:
            trend = "yükseliş"
        elif trend_pct < -10:
            trend = "güçlü düşüş"
        elif trend_pct < -5:
            trend = "düşüş"
        else:
            trend = "sabit"
        
        # Risk hesapla
        volatility = np.std(recent_values) / (np.mean(recent_values) + 0.0001)
        
        if volatility > 0.4:
            risk_level = "yüksek"
        elif volatility > 0.2:
            risk_level = "orta"
        else:
            risk_level = "düşük"
        
        # Dönem bilgisi
        date_range = f"Son {len(clean_values)} dönem"
        
        return {
            "success": True,
            "target_column": target_item,
            "date_range": date_range,
            "trend": trend,
            "trend_percentage": round(trend_pct, 2),
            "risk_level": risk_level,
            "average_value": round(np.mean(recent_values), 2),
            "data_points": len(clean_values),
            "can_forecast": False,  # Wide format için forecast şimdilik kapalı
            "format_type": "wide_format",
            "available_items": items[:20]
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Wide format analiz hatası: {str(e)}"
        }

def analyze_excel(df: pd.DataFrame, target_column: str):
    """Standart long format analiz (eski kod)"""
    
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
        "target_column": target_column,
        "date_range": f"{recent_data[date_col].min().strftime('%Y-%m-%d')} - {recent_data[date_col].max().strftime('%Y-%m-%d')}",
        "trend": trend,
        "trend_percentage": round(trend_pct, 2),
        "risk_level": risk_level,
        "data_points": len(recent_data),
        "average_value": round(np.mean(values), 2),
        "can_forecast": len(df) >= 10,  # En az 10 veri noktası
        "format_type": "long_format",
        "date_column": date_col
    }

def prophet_forecast(df: pd.DataFrame, target_column: str, date_col: str, periods: int = 4):
    """Prophet ile forecast"""
    
    try:
        # Prophet formatına dönüştür
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_column]
        })
        
        # Model oluştur
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True if len(prophet_df) > 365 else False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        # Tahminler
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
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
        
        # Grafik verisi
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
        return {
            "success": False,
            "message": f"Prophet forecast yapılamadı: {str(e)}"
        }

@app.get("/")
def root():
    return {
        "message": "Patron Dijital Asistan API v2",
        "version": "2.0.0",
        "status": "active",
        "features": ["wide_format", "long_format", "prophet_forecast"]
    }

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    target_column: Optional[str] = None
):
    """Excel dosyasını akıllıca analiz et - hem wide hem long format"""
    
    try:
        # Dosyayı oku
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Format tespiti
        is_wide = is_wide_format(df)
        
        # Analiz
        if is_wide:
            # Wide format (oyun bazlı satışlar gibi)
            analysis = analyze_wide_format(df)
            forecast_result = None  # Şimdilik wide format'ta forecast yok
            
        else:
            # Long format (standart zaman serisi)
            columns = df.columns.tolist()
            
            # Hedef kolon belirlenmemişse, sayısal ilk kolonu al
            if target_column is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise HTTPException(status_code=400, detail="Sayısal kolon bulunamadı")
                target_column = numeric_cols[0]
            
            analysis = analyze_excel(df, target_column)
            
            if not analysis["success"]:
                return JSONResponse(content=analysis, status_code=400)
            
            # Forecast (eğer long format ise)
            forecast_result = None
            if analysis.get("can_forecast") and analysis.get("date_column"):
                forecast_result = prophet_forecast(df, target_column, analysis["date_column"])
        
        # Frontend için response hazırla
        claude_prompt_data = {
            "target_column": analysis.get("target_column", "Bilinmiyor"),
            "date_range": analysis.get("date_range", "Bilinmiyor"),
            "trend_summary": f"{analysis.get('trend', 'sabit')} ({analysis.get('trend_percentage', 0)}%)",
            "forecast_result": forecast_result if forecast_result else "Wide format için forecast henüz desteklenmiyor",
            "risk_level": analysis.get("risk_level", "bilinmiyor"),
            "average_value": analysis.get("average_value", 0),
            "data_points": analysis.get("data_points", 0),
            "format_type": analysis.get("format_type", "unknown")
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "forecast": forecast_result,
            "columns": df.columns.tolist() if not is_wide else analysis.get("available_items", [])[:10],
            "target_column": analysis.get("target_column"),
            "claude_prompt_data": claude_prompt_data,
            "format_detected": "wide" if is_wide else "long"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
