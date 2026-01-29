'use client';

import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [claudePrompt, setClaudePrompt] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.xlsx') && !selectedFile.name.endsWith('.xls')) {
        setError('Lütfen Excel dosyası (.xlsx veya .xls) yükleyin');
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Lütfen bir dosya seçin');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analiz başarısız');
      }

      setResult(data);
      
      // Claude prompt'u oluştur
      const promptData = data.claude_prompt_data;
      const prompt = `Sen deneyimli bir CFO'sun.
Ben sana bir şirketin finansal analiz sonuçlarını veriyorum.

Analiz Özeti:
- Analiz edilen kolon: ${promptData.target_column}
- Zaman aralığı: ${promptData.date_range}
- Son 3 ay trendi: ${promptData.trend_summary}
- 3 aylık tahmin sonucu: ${JSON.stringify(promptData.forecast_result, null, 2)}
- Risk seviyesi: ${promptData.risk_level}

Lütfen şu başlıklarla cevap ver:
1. Genel Gidişat
2. Risk Durumu
3. Önümüzdeki 30-60-90-120 gün için öneriler`;
      
      setClaudePrompt(prompt);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(claudePrompt);
    alert('Claude prompt kopyalandı! Claude.ai\'a yapıştırabilirsiniz.');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            📊 Patron Dijital Asistan
          </h1>
          <p className="text-gray-600">
            Excel dosyanızı yükleyin, finansal analizinizi alın
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Excel Dosyası Seçin
              </label>
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-50 file:text-indigo-700
                  hover:file:bg-indigo-100
                  cursor-pointer"
              />
              {file && (
                <p className="mt-2 text-sm text-green-600">
                  ✓ {file.name} seçildi
                </p>
              )}
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg
                font-semibold hover:bg-indigo-700 disabled:bg-gray-400
                disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analiz ediliyor...' : 'Analiz Et'}
            </button>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Analiz Sonuçları */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
                📈 Analiz Sonuçları
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Analiz Edilen Kolon</p>
                  <p className="text-xl font-semibold text-gray-900">
                    {result.target_column}
                  </p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Veri Aralığı</p>
                  <p className="text-xl font-semibold text-gray-900">
                    {result.analysis.date_range}
                  </p>
                </div>

                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Son 3 Ay Trendi</p>
                  <p className="text-xl font-semibold text-gray-900">
                    {result.analysis.trend === 'yükseliş' && '📈'} 
                    {result.analysis.trend === 'düşüş' && '📉'}
                    {result.analysis.trend === 'sabit' && '➡️'}
                    {' '}
                    {result.analysis.trend} ({result.analysis.trend_percentage}%)
                  </p>
                </div>

                <div className={`p-4 rounded-lg ${
                  result.analysis.risk_level === 'yüksek' ? 'bg-red-50' :
                  result.analysis.risk_level === 'orta' ? 'bg-yellow-50' :
                  'bg-green-50'
                }`}>
                  <p className="text-sm text-gray-600">Risk Seviyesi</p>
                  <p className="text-xl font-semibold text-gray-900">
                    {result.analysis.risk_level === 'yüksek' && '🔴'}
                    {result.analysis.risk_level === 'orta' && '🟡'}
                    {result.analysis.risk_level === 'düşük' && '🟢'}
                    {' '}
                    {result.analysis.risk_level}
                  </p>
                </div>
              </div>
            </div>

            {/* Forecast Sonuçları */}
            {result.forecast && result.forecast.success && (
              <div className="bg-white rounded-lg shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">
                  🔮 Prophet AI ile 4 Aylık Tahmin
                </h2>
                
                {/* Grafik */}
                {result.forecast.chart_data && (
                  <div className="mb-8 bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-700 mb-4">
                      📈 Trend ve Tahmin Grafiği
                    </h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <AreaChart
                        data={[
                          ...result.forecast.chart_data.historical.dates.map((date, idx) => ({
                            date: date.includes('Dönem') ? date : new Date(date).toLocaleDateString('tr-TR', { month: 'short', year: 'numeric' }),
                            Gerçekleşen: result.forecast.chart_data.historical.values[idx],
                            type: 'historical'
                          })),
                          ...result.forecast.chart_data.forecast.dates.map((date, idx) => ({
                            date: date.includes('Dönem') ? date : new Date(date).toLocaleDateString('tr-TR', { month: 'short', year: 'numeric' }),
                            Tahmin: result.forecast.chart_data.forecast.values[idx],
                            'Alt Sınır': result.forecast.chart_data.forecast.lower[idx],
                            'Üst Sınır': result.forecast.chart_data.forecast.upper[idx],
                            type: 'forecast'
                          }))
                        ]}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <defs>
                          <linearGradient id="colorGerceklesen" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="colorTahmin" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="date" 
                          angle={-45} 
                          textAnchor="end" 
                          height={80}
                          tick={{ fontSize: 11 }}
                        />
                        <YAxis />
                        <Tooltip 
                          formatter={(value) => value?.toLocaleString('tr-TR')}
                          contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px' }}
                        />
                        <Legend />
                        <Area 
                          type="monotone" 
                          dataKey="Gerçekleşen" 
                          stroke="#3b82f6" 
                          fillOpacity={1} 
                          fill="url(#colorGerceklesen)" 
                          strokeWidth={2}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="Tahmin" 
                          stroke="#10b981" 
                          fillOpacity={1} 
                          fill="url(#colorTahmin)" 
                          strokeWidth={2}
                          strokeDasharray="5 5"
                        />
                        <Area 
                          type="monotone" 
                          dataKey="Alt Sınır" 
                          stroke="#6b7280" 
                          fill="none" 
                          strokeWidth={1}
                          strokeDasharray="2 2"
                        />
                        <Area 
                          type="monotone" 
                          dataKey="Üst Sınır" 
                          stroke="#6b7280" 
                          fill="none" 
                          strokeWidth={1}
                          strokeDasharray="2 2"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                    <p className="text-sm text-gray-600 mt-4 text-center">
                      💡 Mavi alan: Geçmiş veriler | Yeşil alan: Prophet tahmini | Gri çizgiler: Tahmin aralığı
                    </p>
                  </div>
                )}
                
                {/* Sayısal Tahminler */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-700 mb-3">
                    📊 Detaylı Tahminler
                  </h3>
                  {result.forecast.forecasts.map((f, idx) => (
                    <div key={idx} className="bg-gradient-to-r from-green-50 to-emerald-50 p-5 rounded-lg border border-green-200">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-bold text-gray-700 text-lg">
                          {f.date.includes('Dönem') 
                            ? f.date 
                            : `${idx + 1}. Ay - ${new Date(f.date).toLocaleDateString('tr-TR', { month: 'long', year: 'numeric' })}`
                          }
                        </span>
                        <span className="text-2xl font-bold text-green-700">
                          {f.value.toLocaleString('tr-TR')} ₺
                        </span>
                      </div>
                      <div className="flex justify-between text-sm text-gray-600">
                        <span>Alt Sınır: {f.lower.toLocaleString('tr-TR')} ₺</span>
                        <span>Üst Sınır: {f.upper.toLocaleString('tr-TR')} ₺</span>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="mt-6 bg-blue-50 border border-blue-200 p-4 rounded-lg">
                  <p className="text-sm text-blue-800">
                    ℹ️ <strong>Prophet Nedir?</strong> Facebook tarafından geliştirilen, mevsimsellik ve trend değişimlerini otomatik tespit eden gelişmiş bir tahmin algoritmasıdır. Sklearn yedek olarak kullanılır.
                  </p>
                </div>
              </div>
            )}

            {!result.analysis.can_forecast && (
              <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-6 py-4 rounded-lg">
                ⚠️ Forecast için yeterli veri yok (en az 10 veri noktası gerekli)
              </div>
            )}

            {/* Claude Prompt */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">
                  🤖 CFO Yorumu için Claude Prompt
                </h2>
                <button
                  onClick={copyToClipboard}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg
                    hover:bg-green-700 transition-colors text-sm font-semibold"
                >
                  📋 Kopyala
                </button>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
                  {claudePrompt}
                </pre>
              </div>
              
              <p className="mt-4 text-sm text-gray-600">
                👆 Bu prompt'u kopyalayıp Claude.ai'a yapıştırarak CFO yorumunu alabilirsiniz
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
