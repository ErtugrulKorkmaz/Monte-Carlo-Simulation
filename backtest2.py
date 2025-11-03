# -*- coding: utf-8 -*-
"""HAMPTON-STYLE + GÜN TAHMİNİ SİSTEMİ"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 10)
sns.set_style('whitegrid')

print("GÜN TAHMİNİ SİSTEMİ BAŞLATILIYOR...")



class EnhancedEnsembleMonteCarlo:
    def __init__(self):
        self.models = {
            'gbm': self._gbm_model,
            'garch': self._garch_model,
            'heston': self._heston_model,
            'jump_diffusion': self._jump_model,
            'mean_reversion': self._mean_reversion_model
        }
        self.model_weights = None

    def _gbm_model(self, data, days=252, simulations=5000):
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])
        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        paths = np.zeros((days, simulations))
        paths[0] = last_price

        for t in range(1, days):
            shock = np.random.normal(mu * dt, sigma * np.sqrt(dt), simulations)
            paths[t] = paths[t-1] * np.exp(shock)

        return paths, last_price, {'mu': mu, 'sigma': sigma}

    def _garch_model(self, data, days=252, simulations=5000):
        try:
            from arch import arch_model
            prices = data['Close'].values
            returns = np.log(prices[1:] / prices[:-1]) * 100
            garch = arch_model(returns, vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off')
            conditional_vol = garch_fit.conditional_volatility / 100
            last_price = prices[-1]
            mu = np.mean(returns) / 100 * 252
            dt = 1/252

            paths = np.zeros((days, simulations))
            paths[0] = last_price

            for t in range(1, days):
                vol_t = conditional_vol[t % len(conditional_vol)] if t < len(conditional_vol) else conditional_vol[-1]
                shock = np.random.normal(mu * dt, vol_t * np.sqrt(dt), simulations)
                paths[t] = paths[t-1] * np.exp(shock)

            return paths, last_price, {'params': garch_fit.params}
        except ImportError:
            return self._gbm_model(data, days, simulations)

    def _heston_model(self, data, days=252, simulations=5000):
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])
        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        kappa = 1.5
        theta = sigma**2 * 0.8
        xi = 0.2
        rho = -0.3

        paths = np.zeros((days, simulations))
        vol_paths = np.zeros((days, simulations))
        paths[0] = last_price
        vol_paths[0] = theta

        for t in range(1, days):
            z1 = np.random.normal(0, 1, simulations)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, simulations)
            vol_paths[t] = np.maximum(vol_paths[t-1] + kappa * (theta - vol_paths[t-1]) * dt + xi * np.sqrt(np.maximum(vol_paths[t-1], 0.0001)) * np.sqrt(dt) * z2, 0.0001)
            price_shock = (mu - 0.5 * vol_paths[t]) * dt + np.sqrt(vol_paths[t]) * np.sqrt(dt) * z1
            paths[t] = paths[t-1] * np.exp(price_shock)

        return paths, last_price, {'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho}

    def _jump_model(self, data, days=252, simulations=5000):
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])
        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        lambda_jump = 4.0
        jump_mean = 0.0
        jump_std = 0.08

        paths = np.zeros((days, simulations))
        paths[0] = last_price

        for t in range(1, days):
            normal_shock = np.random.normal(mu * dt, sigma * np.sqrt(dt), simulations)
            jump_shock = np.zeros(simulations)
            jump_events = np.random.poisson(lambda_jump * dt, simulations)

            for i in range(simulations):
                if jump_events[i] > 0:
                    direction = np.random.choice([-1, 1])
                    jump_shock[i] = direction * np.sum(np.random.normal(jump_mean, jump_std, jump_events[i]))

            total_shock = normal_shock + jump_shock
            paths[t] = paths[t-1] * np.exp(total_shock)

        return paths, last_price, {'lambda_jump': lambda_jump, 'jump_std': jump_std}

    def _mean_reversion_model(self, data, days=252, simulations=5000):
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])
        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        long_term_mean = np.mean(prices[-252:]) if len(prices) > 252 else np.mean(prices)
        alpha = 0.1

        paths = np.zeros((days, simulations))
        paths[0] = last_price

        for t in range(1, days):
            current_prices = paths[t-1]
            mean_reversion = alpha * (long_term_mean - current_prices) / current_prices * dt
            random_shock = np.random.normal(mu * dt, sigma * np.sqrt(dt), simulations)
            total_shock = mean_reversion + random_shock
            paths[t] = current_prices * np.exp(total_shock)

        return paths, last_price, {'long_term_mean': long_term_mean, 'alpha': alpha}

    def calculate_model_weights(self, data, validation_periods=5):
        model_performances = {}

        for model_name, model_func in self.models.items():
            try:
                errors = []
                for i in range(validation_periods):
                    split_idx = int(len(data) * (0.6 + i * 0.08))
                    train_data = data.iloc[:split_idx]
                    test_data = data.iloc[split_idx:split_idx+63]

                    if len(test_data) < 21:
                        continue

                    predictions, last_price, params = model_func(train_data, days=len(test_data), simulations=2000)
                    pred_means = np.mean(predictions, axis=1)
                    mape = np.mean(np.abs((pred_means - test_data['Close'].values) / test_data['Close'].values)) * 100
                    errors.append(mape)

                if errors:
                    model_performances[model_name] = np.mean(errors)

            except Exception as e:
                continue

        if model_performances:
            total_performance = sum(1/p for p in model_performances.values())
            self.model_weights = {name: (1/performance)/total_performance for name, performance in model_performances.items()}

        return self.model_weights

    def ensemble_predict(self, data, days=252, simulations=10000):
        if self.model_weights is None:
            self.calculate_model_weights(data)

        all_predictions = {}
        ensemble_paths = np.zeros((days, simulations))

        for model_name, model_func in self.models.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                try:
                    predictions, last_price, params = model_func(data, days=days, simulations=simulations)
                    all_predictions[model_name] = predictions
                    ensemble_paths += predictions * weight
                except Exception as e:
                    continue

        return ensemble_paths, last_price, all_predictions

    def calculate_days_to_target(self, price_paths, current_price, target_prices):
        """Hedef fiyatlara ulaşmak için gereken günleri hesapla"""
        days_to_target = {}

        for target_name, target_price in target_prices.items():
            days_distribution = []

            for sim in range(price_paths.shape[1]):
                path = price_paths[:, sim]
                # Hedef fiyata ulaşan ilk günü bul
                target_days = np.where(path >= target_price)[0]
                if len(target_days) > 0:
                    days_distribution.append(target_days[0])

            if days_distribution:
                days_to_target[target_name] = {
                    'target_price': target_price,
                    'average_days': np.mean(days_distribution),
                    'median_days': np.median(days_distribution),
                    'min_days': np.min(days_distribution),
                    'max_days': np.max(days_distribution),
                    'probability_90_days': np.mean(np.array(days_distribution) <= 90) * 100,
                    'probability_180_days': np.mean(np.array(days_distribution) <= 180) * 100
                }

        return days_to_target


class HamptonStyleWithDaysSystem:
    def __init__(self):
        self.ensemble = EnhancedEnsembleMonteCarlo()

    def comprehensive_analysis_with_days(self, ticker="NATEN.IS", total_capital=100000):
        """Gün tahmini eklenmiş kapsamlı analiz"""
        print("GÜN TAHMİNİ ANALİZİ")
        print("=" * 60)

        # Veri yükle
        data = yf.download(ticker, period="5y", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        print(f"✅ {len(data)} günlük veri yüklendi - {ticker}")

        # Ensemble tahmini
        ensemble_predictions, last_price, individual_predictions = self.ensemble.ensemble_predict(data)
        final_prices = ensemble_predictions[-1, :]

        # Hedef fiyatları belirle
        target_prices = {
            '%10_kazanc': last_price * 1.10,
            '%20_kazanc': last_price * 1.20,
            '%30_kazanc': last_price * 1.30,
            '%50_kazanc': last_price * 1.50,
            '%75_kazanc': last_price * 1.75,
            '%100_kazanc': last_price * 2.00,
            '%150_kazanc': last_price * 2.50
        }

        # Gün tahminlerini hesapla
        print("\n⏰ HEDEF FİYAT GÜN TAHMİNLERİ HESAPLANIYOR...")
        days_predictions = self.ensemble.calculate_days_to_target(ensemble_predictions, last_price, target_prices)

        # Temel istatistikler
        mean_prediction = np.mean(final_prices)
        median_prediction = np.median(final_prices)
        confidence_95 = np.percentile(final_prices, [2.5, 97.5])
        profit_probability = np.mean(final_prices > last_price) * 100

        # Sonuçları göster
        print(f"\n🎯 1 YILLIK TAHMİN SONUÇLARI:")
        print(f"   • Mevcut Fiyat: {last_price:.2f} TL")
        print(f"   • Ortalama Tahmin: {mean_prediction:.2f} TL (%{((mean_prediction/last_price)-1)*100:.1f})")
        print(f"   • Medyan Tahmin: {median_prediction:.2f} TL")
        print(f"   • %95 Güven Aralığı: {confidence_95[0]:.2f} - {confidence_95[1]:.2f} TL")
        print(f"   • Kazanç Olasılığı: %{profit_probability:.1f}")

        # Geliştirilmiş hedef analizi (gün tahmini ile)
        print(f"\n🎯 HEDEF FİYAT ve GÜN TAHMİNLERİ:")
        print("=" * 80)
        print(f"{'Hedef':<12} {'Fiyat (TL)':<12} {'Ort. Gün':<10} {'Medyan Gün':<12} {'90g Olasılık':<14} {'180g Olasılık':<14} {'Getiri':<10}")
        print("-" * 80)

        for target_name, prediction in days_predictions.items():
            getiri_yuzde = (prediction['target_price'] / last_price - 1) * 100
            print(f"{target_name:<12} {prediction['target_price']:<12.2f} {prediction['average_days']:<10.0f} {prediction['median_days']:<12.0f} %{prediction['probability_90_days']:<12.1f} %{prediction['probability_180_days']:<13.1f} %{getiri_yuzde:<8.1f}")

        # Risk metrikleri
        print(f"\n⚠️  RİSK METRİKLERİ:")
        var_95 = np.percentile((final_prices - last_price) / last_price, 5) * 100
        max_drawdown_risk = abs(min((final_prices - last_price) / last_price)) * 100

        print(f"   • Value at Risk (95%): %{var_95:.1f}")
        print(f"   • Maximum Drawdown Risk: %{max_drawdown_risk:.1f}")

        # Yatırım önerisi
        print(f"\n💎 OPTİMAL YATIRIM STRATEJİSİ:")

        # En iyi risk/geteri hedefini bul
        best_target = None
        best_score = -np.inf

        for target_name, prediction in days_predictions.items():
            if 'kazanc' in target_name:
                # Skorlama: getiri/gün verimliliği
                getiri = (prediction['target_price'] / last_price - 1) * 100
                efficiency_score = getiri / max(prediction['median_days'], 1)
                risk_adjusted_score = efficiency_score * (prediction['probability_90_days'] / 100)

                if risk_adjusted_score > best_score and getiri > 20:
                    best_score = risk_adjusted_score
                    best_target = (target_name, prediction, getiri)

        if best_target:
            target_name, prediction, getiri = best_target
            print(f"   ✅ EN İYİ HEDEF: {target_name}")
            print(f"   🎯 FİYAT: {prediction['target_price']:.2f} TL (%{getiri:.1f} kazanç)")
            print(f"   ⏰ BEKLENEN SÜRE: {prediction['median_days']:.0f} gün")
            print(f"   📊 3 AY İÇİNDE OLASILIK: %{prediction['probability_90_days']:.1f}")
            print(f"   🛡️  STOP: {last_price * 0.85:.2f} TL (%15 kayıp)")

            # Pozisyon büyüklüğü
            confidence = prediction['probability_90_days'] / 100
            if confidence > 0.7:
                position_size = 0.10
            elif confidence > 0.5:
                position_size = 0.07
            else:
                position_size = 0.05

            position_value = total_capital * position_size
            print(f"   💰 POZİSYON: %{position_size*100:.1f} ({position_value:,.0f} TL)")

        # Zaman bazlı olasılıklar
        print(f"\n📅 ZAMAN BAZLI OLASILIKLAR:")
        time_probabilities = self._calculate_time_based_probabilities(ensemble_predictions, last_price)
        for days, prob in time_probabilities.items():
            print(f"   • {days:3} gün içinde kar olasılığı: %{prob:.1f}")

        return {
            'current_price': last_price,
            'mean_prediction': mean_prediction,
            'median_prediction': median_prediction,
            'days_predictions': days_predictions,
            'time_probabilities': time_probabilities
        }

    def _calculate_time_based_probabilities(self, price_paths, current_price):
        """Zaman bazlı kar olasılıklarını hesapla"""
        time_points = [30, 60, 90, 120, 180, 240]  # gün
        probabilities = {}

        for days in time_points:
            if days < price_paths.shape[0]:
                prices_at_day = price_paths[days, :]
                prob_profit = np.mean(prices_at_day > current_price) * 100
                probabilities[f"{days}"] = prob_profit

        return probabilities



def main():
    print(" GÜN TAHMİNİ SİSTEMİ ÇALIŞTIRILIYOR...")
    print("=" * 60)

    system = HamptonStyleWithDaysSystem()

    try:
        results = system.comprehensive_analysis_with_days("NATEN.IS", total_capital=100000)

        print(f"\n" + "=" * 60)
        print("ANALİZ TAMAMLANDI!")
        print("🎯 ARTIK HEDEFLERE ULAŞMA SÜRELERİNİ DE BİLİYORSUNUZ!")

        return results

    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()