# -*- coding: utf-8 -*-
"""BACKTEST SİSTEMİ - TAM VERSİYON"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 10)
sns.set_style('whitegrid')

print("BACKTEST SİSTEMİ BAŞLATILIYOR...")

# Basitleştirilmiş Ensemble Model (hızlı çalışması için)
class FastEnsembleMonteCarlo:
    def __init__(self):
        self.model_weights = {'gbm': 0.25, 'heston': 0.35, 'jump': 0.20, 'mean_reversion': 0.20}

    def ensemble_predict(self, data, days=126, simulations=2000):
        """Hızlı ensemble tahmini"""
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])

        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        # Basit ensemble (hızlı hesaplama için)
        ensemble_paths = np.zeros((days, simulations))

        # GBM modeli
        gbm_paths = np.zeros((days, simulations))
        gbm_paths[0] = last_price
        for t in range(1, days):
            shock = np.random.normal(mu * dt, sigma * np.sqrt(dt), simulations)
            gbm_paths[t] = gbm_paths[t-1] * np.exp(shock)

        # Heston benzeri (volatilite değişken)
        heston_paths = np.zeros((days, simulations))
        heston_paths[0] = last_price
        vol_path = sigma**2
        for t in range(1, days):
            # Volatilite değişimi
            vol_shock = np.random.normal(0, 0.1)
            vol_path = max(vol_path + 1.5*(sigma**2 - vol_path)*dt + 0.2*np.sqrt(vol_path)*np.sqrt(dt)*vol_shock, 0.0001)

            price_shock = np.random.normal((mu - 0.5*vol_path)*dt, np.sqrt(vol_path)*np.sqrt(dt), simulations)
            heston_paths[t] = heston_paths[t-1] * np.exp(price_shock)

        # Ensemble birleştirme
        ensemble_paths = (gbm_paths * self.model_weights['gbm'] +
                         heston_paths * self.model_weights['heston'] +
                         gbm_paths * self.model_weights['jump'] * 1.1 +  # Jump etkisi
                         gbm_paths * self.model_weights['mean_reversion'] * 0.9)  # Mean reversion etkisi

        return ensemble_paths, last_price

class BacktestEngine:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.portfolio_history = []
        self.trade_history = []
        self.ensemble = FastEnsembleMonteCarlo()

    def run_backtest(self, data, test_periods=8):
        """Basit ve hızlı backtest"""
        print("🎯 BACKTEST ÇALIŞTIRILIYOR...")
        print("=" * 50)

        capital = self.initial_capital
        portfolio_value = capital

        for period in range(test_periods):
            # Train/test split
            split_index = int(len(data) * (0.4 + period * 0.07))
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:split_index+90]  # 3 aylık test

            if len(test_data) < 30:
                continue

            print(f"📊 Dönem {period+1}: {len(train_data)} gün eğitim, {len(test_data)} gün test")

            try:
                # Model tahmini
                predictions, current_price = self.ensemble.ensemble_predict(train_data, days=len(test_data), simulations=1000)
                final_predictions = predictions[-1, :]

                # Tahmin analizi
                mean_prediction = np.mean(final_predictions)
                expected_return = (mean_prediction / current_price - 1)
                profit_probability = np.mean(final_predictions > current_price) * 100

                # Trade kararı
                if expected_return > 0.2 and profit_probability > 60:
                    # AL işlemi
                    entry_price = current_price
                    position_size = 0.1  # %10 pozisyon
                    position_value = capital * position_size
                    shares = position_value / entry_price

                    # SATış zamanı (test periyodu sonu)
                    exit_price = test_data['Close'].iloc[-1]
                    trade_return = (exit_price - entry_price) / entry_price

                    # Trade kaydı
                    self.trade_history.append({
                        'period': period + 1,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'duration': len(test_data),
                        'expected_return': expected_return,
                        'profit_probability': profit_probability
                    })

                    # Portfolio güncelleme
                    trade_pnl = position_value * trade_return
                    capital += trade_pnl
                    portfolio_value = capital

                    status = "✅" if trade_return > 0 else "❌"
                    print(f"   {status} Trade: {entry_price:.2f} → {exit_price:.2f} TL (%{trade_return*100:.1f})")

                self.portfolio_history.append(portfolio_value)

            except Exception as e:
                print(f"   ❌ Hata: {e}")
                continue

        return self.trade_history

    def generate_performance_report(self):
        """Performans raporu"""
        print("\n" + "=" * 60)
        print("📊 BACKTEST PERFORMANS RAPORU")
        print("=" * 60)

        if not self.trade_history:
            print("❌ Yeterli trade yok!")
            return

        returns = [t['return'] for t in self.trade_history]
        total_return = (self.portfolio_history[-1] - self.initial_capital) / self.initial_capital * 100

        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]

        win_rate = len(winning_trades) / len(returns) * 100
        avg_return = np.mean(returns) * 100
        avg_win = np.mean(winning_trades) * 100 if winning_trades else 0
        avg_loss = np.mean(losing_trades) * 100 if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else np.inf

        print(f"\n🎯 GENEL PERFORMANS:")
        print(f"   • Başlangıç: {self.initial_capital:,.0f} TL")
        print(f"   • Son Portföy: {self.portfolio_history[-1]:,.0f} TL")
        print(f"   • Toplam Getiri: %{total_return:.1f}")

        print(f"\n📈 TRADE İSTATİSTİKLERİ:")
        print(f"   • Toplam Trade: {len(returns)}")
        print(f"   • Başarı Oranı: %{win_rate:.1f}")
        print(f"   • Ortalama Getiri: %{avg_return:.1f}")
        print(f"   • Ort. Kazanç: %{avg_win:.1f}")
        print(f"   • Ort. Kayıp: %{avg_loss:.1f}")
        print(f"   • Profit Factor: {profit_factor:.2f}")

        print(f"\n🔍 MODEL DOĞRULUK:")
        prediction_accuracy = []
        for trade in self.trade_history:
            predicted_profit = trade['expected_return'] > 0
            actual_profit = trade['return'] > 0
            if predicted_profit == actual_profit:
                prediction_accuracy.append(1)
            else:
                prediction_accuracy.append(0)

        accuracy = np.mean(prediction_accuracy) * 100
        print(f"   • Yön Tahmin Doğruluğu: %{accuracy:.1f}")

        # Trade detayları
        print(f"\n💎 SON 5 TRADE:")
        for trade in self.trade_history[-5:]:
            status = "✅" if trade['return'] > 0 else "❌"
            print(f"   • {status} %{trade['return']*100:5.1f} (Beklenen: %{trade['expected_return']*100:.1f})")

        # Görselleştirme
        self._plot_results()

        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'profit_factor': profit_factor
        }

    def _plot_results(self):
        """Sonuçları görselleştir"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Portfolio değeri
        ax1.plot(self.portfolio_history, linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', label='Başlangıç')
        ax1.set_title('Portföy Değeri Gelişimi')
        ax1.set_ylabel('Portföy Değeri (TL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Trade getirileri
        returns = [t['return'] * 100 for t in self.trade_history]
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.bar(range(1, len(returns)+1), returns, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_title('Trade Getirileri')
        ax2.set_xlabel('Trade No')
        ax2.set_ylabel('Getiri (%)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Ana backtest fonksiyonu
def run_comprehensive_backtest():
    """Kapsamlı backtest çalıştır"""
    print("BACKTEST SİSTEMİ")
    print("=" * 60)

    # Veri yükle
    data = yf.download("NATEN.IS", period="5y", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    print(f"✅ {len(data)} günlük veri yüklendi")

    # Backtest çalıştır
    backtest = BacktestEngine(initial_capital=100000)
    trades = backtest.run_backtest(data)

    if trades:
        performance = backtest.generate_performance_report()

        print(f"\n" + "=" * 60)
        print("🎯 BACKTEST SONUÇ DEĞERLENDİRMESİ:")

        if performance['total_return'] > 20:
            print("✅ SİSTEM ÇOK BAŞARILI - Yüksek getiri")
        elif performance['total_return'] > 0:
            print("✅ SİSTEM BAŞARILI - Pozitif getiri")
        else:
            print("⚠️  SİSTEM GELİŞTİRİLMELİ - Negatif getiri")

        if performance['win_rate'] > 60:
            print("✅ YÜKSEK KAZANMA ORANI - İstikrarlı")
        elif performance['win_rate'] > 50:
            print("✅ MAKUL KAZANMA ORANI - Kullanılabilir")
        else:
            print("⚠️  DÜŞÜK KAZANMA ORANI - Riskli")

        if performance['accuracy'] > 70:
            print("✅ MODEL DOĞRULUĞU YÜKSEK - Güvenilir")
        elif performance['accuracy'] > 60:
            print("✅ MODEL DOĞRULUĞU MAKUL - Kullanılabilir")
        else:
            print("⚠️  MODEL DOĞRULUĞU DÜŞÜK - Geliştirilmeli")

    return backtest

# Çalıştır
if __name__ == "__main__":
    backtest_results = run_comprehensive_backtest()