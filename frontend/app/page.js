'use client';

import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [claudePrompt, setClaudePrompt] = useState('');
  const [cfoComment, setCfoComment] = useState('');
  const [commentLoading, setCommentLoading] = useState(false);
  
  // Filtreler
  const [selectedCategory, setSelectedCategory] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

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
    
    // Filtreler varsa ekle
    if (selectedCategory) {
      formData.append('selected_category', selectedCategory);
    }
    if (startDate) {
      formData.append('start_date', startDate);
    }
    if (endDate) {
      formData.append('end_date', endDate);
    }

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

      // Otomatik CFO yorumu al
      setCommentLoading(true);
      try {
        const commentResponse = await fetch(`${API_URL}/generate-cfo-comment`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            trend: data.analysis.trend,
            trend_percentage: data.analysis.trend_percentage,
            risk_level: data.analysis.risk_level,
            target_column: data.analysis.target_column
          }),
        });
        
        const commentData = await commentResponse.json();
        if (commentData.success) {
          setCfoComment(commentData.comment);
        }
      } catch (err) {
        console.error('CFO yorumu alınamadı:', err);
      } finally {
        setCommentLoading(false);
      }

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

            {/* Filtreler - Sonuç geldikten sonra göster */}
            {result && result.available_categories && result.available_categories.length > 0 && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <h3 className="font-semibold text-gray-900 mb-4">🎯 Filtreleme Seçenekleri</h3>
                
                {/* Kategori Seçici */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Kategori/Oyun Seç
                  </label>
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">Otomatik (En yüksek)</option>
                    {result.available_categories.map((cat, idx) => (
                      <option key={idx} value={cat}>{cat}</option>
                    ))}
                  </select>
                </div>

                {/* Tarih Aralığı */}
                {result.available_dates && result.available_dates.length > 0 && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Başlangıç Tarihi
                      </label>
                      <select
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                      >
                        <option value="">Tümü</option>
                        {result.available_dates.map((d, idx) => (
                          <option key={idx} value={d.date}>{d.display}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Bitiş Tarihi
                      </label>
                      <select
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                      >
                        <option value="">Tümü</option>
                        {result.available_dates.map((d, idx) => (
                          <option key={idx} value={d.date}>{d.display}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}

                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded-lg
                    font-semibold hover:bg-green-700 disabled:bg-gray-400
                    disabled:cursor-not-allowed transition-colors"
                >
                  🔄 Filtreleri Uygula ve Tekrar Analiz Et
                </button>
              </div>
            )}

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
                
                {/* Birim açıklaması */}
                <div className="mb-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                  <p className="text-sm text-blue-800">
                    💡 <strong>Değerler:</strong> {result.format_detected === 'wide' 
                      ? 'Milyon TL cinsinden aylık toplam cirolar' 
                      : 'TL cinsinden değerler'}
                  </p>
                </div>
                
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
                            date: date,
                            Gerçekleşen: result.forecast.chart_data.historical.values[idx],
                            type: 'historical'
                          })),
                          ...result.forecast.chart_data.forecast.dates.map((date, idx) => ({
                            date: date,
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
                      💡 Mavi alan: Geçmiş veriler (Milyon TL) | Yeşil alan: Tahmin (Milyon TL) | Gri çizgiler: Güven aralığı (%95)
                    </p>
                  </div>
                )}
                
                {/* Sayısal Tahminler */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-700 mb-3">
                    📊 Detaylı Tahminler
                    <span className="text-sm font-normal text-gray-500 ml-2">
                      (Milyon TL)
                    </span>
                  </h3>
                  {result.forecast.forecasts.map((f, idx) => (
                    <div key={idx} className="bg-gradient-to-r from-green-50 to-emerald-50 p-5 rounded-lg border border-green-200">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-bold text-gray-700 text-lg">
                          {f.date}
                        </span>
                        <span className="text-2xl font-bold text-green-700">
                          {f.value.toLocaleString('tr-TR')} Milyon ₺
                        </span>
                      </div>
                      <div className="flex justify-between text-sm text-gray-600">
                        <span>Alt Sınır: {f.lower.toLocaleString('tr-TR')} Milyon ₺</span>
                        <span>Üst Sınır: {f.upper.toLocaleString('tr-TR')} Milyon ₺</span>
                      </div>
                      <div className="mt-2 text-xs text-gray-500 italic">
                        Gerçek değer: ~{(f.value * 1000000).toLocaleString('tr-TR')} ₺
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

            {/* CFO Yorumu */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">
                  🤖 CFO Yorumu
                </h2>
              </div>
              
              {commentLoading ? (
                <div className="flex items-center justify-center p-8">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
                  <span className="ml-4 text-gray-600">Analiz ediliyor...</span>
                </div>
              ) : cfoComment ? (
                <div className="prose max-w-none">
                  <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                    {cfoComment.split('\n').map((line, idx) => {
                      if (line.startsWith('## ')) {
                        return <h2 key={idx} className="text-xl font-bold text-gray-900 mt-6 mb-3">{line.replace('## ', '')}</h2>;
                      } else if (line.startsWith('**') && line.endsWith('**')) {
                        return <p key={idx} className="font-bold text-gray-800 mt-4 mb-2">{line.replace(/\*\*/g, '')}</p>;
                      } else if (line.includes('**')) {
                        const parts = line.split('**');
                        return (
                          <p key={idx} className="text-gray-700 mb-2">
                            {parts.map((part, i) => i % 2 === 1 ? <strong key={i}>{part}</strong> : part)}
                          </p>
                        );
                      } else if (line.startsWith('⚠️') || line.startsWith('✅') || line.startsWith('📊')) {
                        return <p key={idx} className="text-gray-700 mb-3 pl-4 border-l-2 border-blue-500">{line}</p>;
                      } else if (line.trim() === '---') {
                        return <hr key={idx} className="my-6 border-gray-300" />;
                      } else if (line.trim() === '') {
                        return <br key={idx} />;
                      } else {
                        return <p key={idx} className="text-gray-700 mb-2">{line}</p>;
                      }
                    })}
                  </div>
                </div>
              ) : (
                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    ⚠️ Otomatik yorum oluşturulamadı. Manuel prompt kullanabilirsiniz.
                  </p>
                </div>
              )}
              
              {/* Manuel Prompt (opsiyonel) */}
              <details className="mt-6">
                <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-900">
                  Manuel Claude Prompt'unu göster
                </summary>
                <div className="mt-4 bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                    {claudePrompt}
                  </pre>
                  <button
                    onClick={copyToClipboard}
                    className="mt-4 bg-green-600 text-white px-4 py-2 rounded-lg
                      hover:bg-green-700 transition-colors text-sm font-semibold"
                  >
                    📋 Kopyala
                  </button>
                </div>
              </details>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
