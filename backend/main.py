from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json
from typing import Optional, Tuple, List
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

def smart_excel_reader(file_content) -> Tuple[pd.DataFrame, str, dict]:
    """
    Akıllı Excel okuyucu - farklı formatları otomatik tespit eder
    Returns: (DataFrame, format_type, metadata)
    """
    
    try:
        # Önce tüm sheet'leri kontrol et
        excel_file = pd.ExcelFile(io.BytesIO(file_content))
        sheet_names = excel_file.sheet_names
        
        # İlk sheet'i oku (header=None ile)
        df_raw = pd.read_excel(io.BytesIO(file_content), header=None)
        
        # Boş satırları tespit et ve atla
        non_empty_rows = []
        for idx, row in df_raw.iterrows():
            if not row.isnull().all():
                non_empty_rows.append(idx)
        
        if len(non_empty_rows) == 0:
            raise Exception("Excel dosyası boş")
        
        # İlk veri olan satırı bul
        first_data_row = non_empty_rows[0]
        
        # Header satırını belirle (genelde ilk non-empty satır)
        df = pd.read_excel(io.BytesIO(file_content), skiprows=first_data_row)
        
        # Format tespiti
        format_info = detect_format(df)
        
        metadata = {
            "sheets": sheet_names,
            "rows": len(df),
            "columns": len(df.columns),
            "first_data_row": first_data_row
        }
        
        return df, format_info["type"], metadata
    
    except Exception as e:
        # Fallback
        df = pd.read_excel(io.BytesIO(file_content))
        return df, "standard", {"error": str(e)}

def detect_format(df: pd.DataFrame) -> dict:
    """Excel formatını tespit et"""
    
    # 1. Wide Format Kontrolü (çok kolon, satırlarda ürün/kategori)
    if len(df.columns) > 15:
        # İlk kolon text mi, geri kalanı sayısal mı?
        first_col_text = df.iloc[:, 0].dtype == 'object'
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if first_col_text and len(numeric_cols) > 5:
            return {
                "type": "wide_format",
                "id_column": df.columns[0],
                "value_columns": numeric_cols.tolist(),
                "description": "Satırlarda ürünler/kategoriler, kolonlarda dönemler"
            }
    
    # 2. Long Format Kontrolü (tarih kolonlu)
    for col in df.columns[:5]:  # İlk 5 kolonu kontrol et
        try:
            dates = pd.to_datetime(df[col], errors='coerce')
            valid_dates = dates.notna().sum()
            if valid_dates / len(df) > 0.3:  # %30'dan fazlası tarih
                return {
                    "type": "long_format",
                    "date_column": col,
                    "value_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "description": "Standart format - satırlarda tarihler"
                }
        except:
            continue
    
    # 3. Standard Format (sayısal kolonlar var ama net tarih yok)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return {
            "type": "standard",
            "value_columns": numeric_cols.tolist(),
            "description": "Standart tablo formatı"
        }
    
    return {
        "type": "unknown",
        "description": "Format belirlenemedi"
    }

def convert_wide_to_long(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """
    Wide format'ı long format'a çevir
    Örnek: Oyun adları satırlarda, aylar kolonlarda → Her satır bir oyun-ay kombinasyonu
    """
    
    # Tarih/dönem gibi görünen kolonları bul
    period_cols = []
    for col in df.columns:
        if col == id_column:
            continue
        col_str = str(col).lower()
        # Ay isimleri, yıllar, dönemler
        keywords = ['oca', 'şub', 'mar', 'nis', 'may', 'haz', 'tem', 'ağu', 'eyl', 'eki', 'kas', 'ara',
                   'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                   '2022', '2023', '2024', '2025', '2026', 'ytd', 'q1', 'q2', 'q3', 'q4', 'fy']
        
        if any(kw in col_str for kw in keywords) or pd.api.types.is_numeric_dtype(df[col]):
            period_cols.append(col)
    
    # Melt işlemi
    df_long = df.melt(
        id_vars=[id_column],
        value_vars=period_cols,
        var_name='period',
        value_name='value'
    )
    
    # Boş değerleri temizle
    df_long = df_long.dropna(subset=['value'])
    
    # Değerleri sayısal yap
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value'])
    
    # Sıralama için indeks
    df_long = df_long.reset_index(drop=True)
    df_long['time_index'] = range(len(df_long))
    
    return df_long

def analyze_data(df: pd.DataFrame, format_type: str, target_item: Optional[str] = None) -> dict:
    """
    Veriyi analiz et - format tipine göre
    """
    
    try:
        if format_type == "wide_format":
            return analyze_wide_format(df, target_item)
        elif format_type == "long_format":
            return analyze_long_format(df, target_item)
        else:
            return analyze_standard_format(df, target_item)
    except Exception as e:
        raise Exception(f"Analiz hatası: {str(e)}")

def analyze_wide_format(df: pd.DataFrame, target_item: Optional[str] = None) -> dict:
    """Wide format analizi (oyun bazlı satışlar gibi)"""
    
    id_column = df.columns[0]
    items = df[id_column].tolist()
    
    # Eğer target belirtilmemişse, en yüksek toplam değere sahip olanı seç
    if target_item is None or target_item not in items:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sums = df[numeric_cols].sum(axis=1)
        target_idx = sums.idxmax()
        target_item = df.iloc[target_idx][id_column]
    
    # Target item'ın verisini al
    target_row = df[df[id_column] == target_item].iloc[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    values = target_row[numeric_cols].values
    
    # NaN ve 0 değerleri temizle
    clean_values = [v for v in values if pd.notna(v) and v != 0]
    
    if len(clean_values) < 3:
        return {
            "success": False,
            "message": f"{target_item} için yeterli veri yok"
        }
    
    # Trend analizi (son değerler)
    recent_count = min(12, len(clean_values))
    recent_values = clean_values[-recent_count:]
    
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
    
    # Risk analizi
    volatility = np.std(recent_values) / (np.mean(recent_values) + 0.0001)
    
    if volatility > 0.4:
        risk_level = "yüksek"
    elif volatility > 0.2:
        risk_level = "orta"
    else:
        risk_level = "düşük"
    
    # Özet istatistikler
    total = sum(clean_values)
    average = np.mean(recent_values)
    max_val = max(recent_values)
    min_val = min(recent_values)
    
    return {
        "success": True,
        "target_item": target_item,
        "format": "wide_format",
        "trend": trend,
        "trend_percentage": round(trend_pct, 2),
        "risk_level": risk_level,
        "volatility": round(volatility, 3),
        "stats": {
            "total": round(total, 2),
            "average": round(average, 2),
            "max": round(max_val, 2),
            "min": round(min_val, 2),
            "data_points": len(clean_values)
        },
        "can_forecast": len(clean_values) >= 6,
        "available_items": items[:20]  # İlk 20 item
    }

def analyze_long_format(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """Long format analizi (standart zaman serisi)"""
    
    # Burası eski analyze_excel fonksiyonu gibi çalışır
    # Şimdilik basitleştirilmiş
    return {
        "success": True,
        "format": "long_format",
        "message": "Long format desteği yakında gelecek"
    }

def analyze_standard_format(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """Standard format analizi"""
    
    return {
        "success": True,
        "format": "standard",
        "message": "Standard format desteği yakında gelecek"
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
    target_item: Optional[str] = None
):
    """Excel dosyasını akıllıca analiz et"""
    
    try:
        # Dosyayı oku
        contents = await file.read()
        
        # Akıllı okuyucu
        df, format_type, metadata = smart_excel_reader(contents)
        
        # Analiz
        analysis_result = analyze_data(df, format_type, target_item)
        
        if not analysis_result["success"]:
            return JSONResponse(content=analysis_result, status_code=400)
        
        # Claude prompt hazırla
        claude_prompt_data = {
            "analyzed_item": analysis_result.get("target_item", "Genel"),
            "format_type": format_type,
            "trend_summary": f"{analysis_result['trend']} ({analysis_result['trend_percentage']}%)",
            "risk_level": analysis_result["risk_level"],
            "stats": analysis_result.get("stats", {})
        }
        
        return {
            "success": True,
            "format_detected": format_type,
            "metadata": metadata,
            "analysis": analysis_result,
            "claude_prompt_data": claude_prompt_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
