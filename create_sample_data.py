import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Örnek finansal veri oluştur
dates = pd.date_range(start='2024-01-01', end='2025-01-27', freq='W')
np.random.seed(42)

# Gelir verisi (yükseliş trendi ile)
base_revenue = 50000
trend = np.linspace(0, 20000, len(dates))
noise = np.random.normal(0, 3000, len(dates))
revenue = base_revenue + trend + noise

# Gider verisi (daha az volatil)
base_expense = 35000
expense_trend = np.linspace(0, 8000, len(dates))
expense_noise = np.random.normal(0, 2000, len(dates))
expense = base_expense + expense_trend + expense_noise

# DataFrame oluştur
df = pd.DataFrame({
    'Tarih': dates,
    'Gelir': revenue.round(2),
    'Gider': expense.round(2),
    'Net_Kar': (revenue - expense).round(2)
})

# Excel'e kaydet
df.to_excel('ornek_finansal_veri.xlsx', index=False)
print("✅ Örnek Excel dosyası oluşturuldu: ornek_finansal_veri.xlsx")
print(f"📊 {len(df)} satır veri içeriyor")
print(f"📅 Tarih aralığı: {df['Tarih'].min().date()} - {df['Tarih'].max().date()}")
