Finansal piyasa tahmini ve risk analizi için geliştirilmiş sofistike ensemble Monte Carlo simülasyon sistemi. Bu sistem, çoklu stokastik modelleri Bayesian optimizasyonu ile birleştirerek gerçekçi fiyat projeksiyonları ve kapsamlı risk metrikleri üretir.

## Temel Özellikler

### Çoklu Model Ensemble Sistemi
- **Geometric Brownian Motion (GBM)**: Geleneksel fiyat hareketi modellemesi
- **GARCH Volatility Model**: Zamanla değişen volatilite ve kümeleme etkileri
- **Heston Stochastic Volatility**: Ortalama geri dönüşlü dinamik volatilite modellemesi
- **Jump Diffusion Process**: Piyasa şoku ve boşluk olayı simülasyonu
- **Mean Reversion Model**: Fiyat denge modellemesi

### İleri Düzey Analitikler
- **Bayesian Parameter Optimization**: Otomatik parametre ayarlama
- **Volatility Regime Detection**: Piyasa koşullarına uyum
- **Multiple Timeframe Analysis**: Çoklu zaman dilimi doğrulaması
- **Dynamic Position Sizing**: Risk ayarlı pozisyon büyüklüğü belirleme

### Risk Yönetimi
- **Value at Risk (VaR)**: %95 güven seviyesinde maksimum kayıp
- **Expected Shortfall (CVaR)**: Koşullu risk değeri
- **Maximum Drawdown**: Maksimum çekilme analizi
- **Confidence Intervals**: Güven aralıkları ile tahmin dağılımı


## Model Detayları

### Ensemble Ağırlıkları
Sistem, her modelin tarihsel performansına göre dinamik ağırlıklar belirler:
- Heston Modeli: ~%21-22
- GBM: ~%19-20
- GARCH: ~%19-20
- Jump Diffusion: ~%19-20
- Mean Reversion: ~%19-20

### Validation Metrikleri
- **MAPE (Mean Absolute Percentage Error)**: %16-18
- **Coverage Rate**: %98+
- **Direction Accuracy**: %75+

## Çıktılar ve Raporlar

Sistem aşağıdaki detaylı raporları sağlar:

### 1. Tahmin Sonuçları
- 1 yıllık fiyat projeksiyonları
- Güven aralıkları
- Kazanç/kayıp olasılıkları

### 2. Hedef Analizi
- Hedef fiyat seviyeleri
- Beklenen ulaşma süreleri
- Zaman bazlı olasılıklar

### 3. Risk Metrikleri
- Value at Risk
- Expected Shortfall
- Maksimum çekilme riski

### 4. Yatırım Önerileri
- Optimal pozisyon büyüklüğü
- Stop-loss seviyeleri
- Çıkış stratejileri

## Performans

### Backtest Sonuçları
- **Ortalama Getiri**: %19.9
- **Başarı Oranı**: %75.0
- **Profit Factor**: 7.74
- **Yön Doğruluğu**: %75.0

### Gerçekçilik Kontrolleri
- Tarihsel veri backtest'i
- Çoklu model validasyonu
- Risk metrikleri doğrulaması
- Piyasa rejimi adaptasyonu

## Geliştirme ve Katkı

### Planlanan Özellikler
- [ ] Fundamental analiz entegrasyonu
- [ ] Sentiment analizi
- [ ] Reel zamanlı veri akışları
- [ ] Portföy optimizasyonu
