from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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

def parse_period_headers(df: pd.DataFrame) -> dict:
    """
    Wide format kolon başlıklarını parse et ve tarihlere çevir
    Örnek: 'Ağu YTD Act' → Ağustos 2024
    """
    
    # Türkçe ay isimleri
    ay_map = {
        'oca': 1, 'ocak': 1,
        'şub': 2, 'şubat': 2,
        'mar': 3, 'mart': 3,
        'nis': 4, 'nisan': 4,
        'may': 5, 'mayıs': 5,
        'haz': 6, 'haziran': 6,
        'tem': 7, 'temmuz': 7,
        'ağu': 8, 'ağustos': 8,
        'eyl': 9, 'eylül': 9,
        'eki': 10, 'ekim': 10,
        'kas': 11, 'kasım': 11,
        'ara': 12, 'aralık': 12
    }
    
    period_info = []
    current_year = 2022  # Başlangıç yılı
    
    for col in df.columns[1:]:  # İlk kolon (id) hariç
        col_lower = str(col).lower().replace('\\n', ' ')
        
        # Yıl kontrolü
        for year in range(2022, 2030):
            if str(year) in col_lower:
                current_year = year
                break
        
        # Ay kontrolü
        found_month = None
        for ay_key, ay_num in ay_map.items():
            if ay_key in col_lower:
                found_month = ay_num
                break
        
        if found_month:
            # Tarih nesnesi oluştur
            try:
                date_obj = datetime(current_year, found_month, 1)
                period_info.append({
                    'column': col,
                    'date': date_obj,
                    'display': date_obj.strftime('%b %Y')  # Kas 2024
                })
            except:
                period_info.append({
                    'column': col,
                    'date': None,
                    'display': str(col)
                })
    
    return period_info

def is_wide_format(df: pd.DataFrame) -> bool:
    """Excel'in wide format olup olmadığını kontrol et"""
    
    # Çok az kolon varsa kesinlikle long format
    if len(df.columns) <= 10:
        return False
    
    # Çok kolon var mı? (>15 kolon)
    if len(df.columns) < 15:
        return False
    
    # İlk kolonda tarih var mı kontrol et
    try:
        first_col = df.iloc[:, 0]
        # İlk kolonu tarihe çevirmeyi dene
        dates = pd.to_datetime(first_col, errors='coerce')
        valid_dates = dates.notna().sum()
        
        # Eğer ilk kolonun %30'undan fazlası tarihse, bu long format
        if valid_dates / len(first_col) > 0.3:
            return False
    except:
        pass
    
    # İlk kolon text, geri kalanlar çoğunlukla sayısal mı?
    try:
        first_col_text = df.iloc[:, 0].dtype == 'object'
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_ratio = len(numeric_cols) / len(df.columns)
        
        # İlk kolon text VE kolonların %60'ından fazlası sayısal ise wide format
        return first_col_text and numeric_ratio > 0.6
    except:
        return False

def analyze_wide_format(df: pd.DataFrame) -> dict:
    """
    Wide format analizi - Gerçek tarihlerle
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
        
        # Kolon başlıklarını parse et
        period_info = parse_period_headers(df)
        
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
        
        # Sadece tarih bilgisi olan kolonları al
        dated_values = []
        dated_periods = []
        
        for info in period_info:
            if info['date'] and info['column'] in df.columns:
                val = target_row[info['column']]
                if pd.notna(val) and val != 0:
                    try:
                        dated_values.append(float(val))
                        dated_periods.append(info)
                    except:
                        pass
        
        if len(dated_values) < 3:
            return {
                "success": False,
                "message": f"{target_item} için yeterli veri yok (min 3 dönem gerekli)"
            }
        
        # Anomali tespiti (outlier detection)
        values_array = np.array(dated_values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        # Z-score ile outlier tespiti
        z_scores = np.abs((values_array - mean_val) / (std_val + 0.0001))
        outliers = z_scores > 3  # 3 sigma dışı
        
        # Outlier'ları temizle (forecast için)
        clean_values_for_forecast = [v for i, v in enumerate(dated_values) if not outliers[i]]
        
        # Son 12 dönemi al (analiz için - outlier'lar dahil)
        recent_count = min(12, len(dated_values))
        recent_values = dated_values[-recent_count:]
        recent_periods = dated_periods[-recent_count:]
        
        # Trend hesapla (outlier'sız)
        if len(clean_values_for_forecast) >= 6:
            recent_clean = clean_values_for_forecast[-6:]
            first_half = np.mean(recent_clean[:len(recent_clean)//2])
            second_half = np.mean(recent_clean[len(recent_clean)//2:])
        else:
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
        volatility = np.std(clean_values_for_forecast) / (np.mean(clean_values_for_forecast) + 0.0001)
        
        if volatility > 0.4:
            risk_level = "yüksek"
        elif volatility > 0.2:
            risk_level = "orta"
        else:
            risk_level = "düşük"
        
        # Tarih aralığı
        if len(dated_periods) > 0:
            start_date = dated_periods[0]['date'].strftime('%b %Y')
            end_date = dated_periods[-1]['date'].strftime('%b %Y')
            date_range = f"{start_date} - {end_date}"
        else:
            date_range = f"Son {len(dated_values)} dönem"
        
        return {
            "success": True,
            "target_column": target_item,
            "date_range": date_range,
            "trend": trend,
            "trend_percentage": round(trend_pct, 2),
            "risk_level": risk_level,
            "average_value": round(np.mean(clean_values_for_forecast), 2),
            "data_points": len(dated_values),
            "can_forecast": len(clean_values_for_forecast) >= 6,
            "format_type": "wide_format",
            "available_items": items[:20],
            "period_data": clean_values_for_forecast,  # Outlier'sız veriler
            "period_info": dated_periods,  # Tarih bilgileri
            "outliers_detected": int(sum(outliers))
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Wide format analiz hatası: {str(e)}"
        }

def wide_format_forecast(period_data: list, period_info: list, periods: int = 4):
    """Wide format için forecast - gerçek tarihlerle"""
    
    try:
        n = len(period_data)
        
        # Linear regression
        X = np.array(range(n)).reshape(-1, 1)
        y = np.array(period_data)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Son tarih
        if len(period_info) > 0:
            last_date = period_info[-1]['date']
        else:
            last_date = datetime.now()
        
        # Tahminler - relativedelta ile doğru ay hesabı
        forecasts = []
        for i in range(1, periods + 1):
            future_x = n + i - 1
            forecast_value = model.predict([[future_x]])[0]
            
            # Güven aralığı
            std_error = np.std(y - model.predict(X))
            
            # Gelecek tarih (relativedelta ile ay bazlı)
            future_date = last_date + relativedelta(months=i)
            
            forecasts.append({
                "date": future_date.strftime('%b %Y'),  # Oca 2027
                "value": round(max(0, forecast_value), 2),
                "lower": round(max(0, forecast_value - std_error * 1.96), 2),
                "upper": round(max(0, forecast_value + std_error * 1.96), 2),
                "month": i
            })
        
        # Grafik verisi
        historical_count = min(12, len(period_data))
        historical = period_data[-historical_count:]
        historical_periods = period_info[-historical_count:] if len(period_info) >= historical_count else []
        
        chart_data = {
            "historical": {
                "dates": [p['display'] for p in historical_periods] if historical_periods else [f"Dönem {i+1}" for i in range(len(historical))],
                "values": historical
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
            "method": "linear_regression_with_dates",
            "chart_data": chart_data
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Wide format forecast hatası: {str(e)}"
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
    """Prophet ile forecast - hata varsa sklearn'e düş"""
    
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
        
        # Stan backend sorunu için suppress warnings
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        
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
        # Prophet başarısız olursa sklearn ile basit tahmin yap
        print(f"Prophet failed: {str(e)}, falling back to sklearn")
        return sklearn_fallback_forecast(df, target_column, date_col, periods)

def sklearn_fallback_forecast(df: pd.DataFrame, target_column: str, date_col: str, periods: int = 4):
    """Prophet çalışmazsa sklearn ile basit forecast"""
    
    try:
        # Tarihleri sayıya çevir
        df = df.sort_values(by=date_col)
        df['days'] = (df[date_col] - df[date_col].min()).dt.days
        
        # Linear regression
        X = df[['days']].values
        y = df[target_column].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Tahminler
        last_date = df[date_col].max()
        last_days = df['days'].max()
        
        forecasts = []
        for i in range(1, periods + 1):
            forecast_date = last_date + timedelta(days=30 * i)
            forecast_days = last_days + (30 * i)
            
            forecast_value = model.predict([[forecast_days]])[0]
            
            # Basit güven aralığı
            std_error = np.std(y - model.predict(X))
            
            forecasts.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "value": round(max(0, forecast_value), 2),
                "lower": round(max(0, forecast_value - std_error * 1.96), 2),
                "upper": round(max(0, forecast_value + std_error * 1.96), 2),
                "month": i
            })
        
        # Grafik verisi
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
        
        return {
            "success": True,
            "forecasts": forecasts,
            "method": "linear_regression_fallback",
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
        
        # Önce ham oku - boş satırları tespit et
        df_raw = pd.read_excel(io.BytesIO(contents), header=None)
        
        # İlk boş olmayan satırı bul
        first_data_row = 0
        for idx, row in df_raw.iterrows():
            if not row.isnull().all():
                first_data_row = idx
                break
        
        # Eğer ilk satır boşsa veya çok az veri varsa, bir sonraki satırdan başla
        if df_raw.iloc[first_data_row].isnull().sum() > len(df_raw.columns) * 0.5:
            first_data_row += 1
        
        # Düzgün header ile oku
        df = pd.read_excel(io.BytesIO(contents), skiprows=first_data_row)
        
        # Format tespiti
        is_wide = is_wide_format(df)
        
        # Analiz
        if is_wide:
            # Wide format (oyun bazlı satışlar gibi)
            analysis = analyze_wide_format(df)
            
            # Wide format için forecast
            forecast_result = None
            if analysis.get("can_forecast") and analysis.get("period_data"):
                forecast_result = wide_format_forecast(
                    analysis["period_data"],
                    analysis.get("period_info", [])
                )
            
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
            "forecast_result": forecast_result if forecast_result else "Forecast için yeterli veri yok",
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
