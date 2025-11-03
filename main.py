# -*- coding: utf-8 -*-
"""HAMPTON-STYLE İLERİ DÜZEY MONTE CARLO SİSTEMİ - TÜM IMPROVEMENT'LAR"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 10)
sns.set_style('whitegrid')

print("SİSTEM BAŞLATILIYOR...")


class EnsembleMonteCarlo:
    """Çoklu model ensemble sistemi"""

    def __init__(self):
        self.models = {
            'gbm': self._gbm_model,
            'garch': self._garch_model,
            'heston': self._heston_model,
            'jump_diffusion': self._jump_model,
            'mean_reversion': self._mean_reversion_model
        }
        self.model_weights = None
        self.performance_history = {}

    def _gbm_model(self, data, days=252, simulations=5000):
        """Geometric Brownian Motion"""
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
        """GARCH Volatilite Modeli"""
        try:
            from arch import arch_model

            prices = data['Close'].values
            returns = np.log(prices[1:] / prices[:-1]) * 100

            # GARCH(1,1)
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
        """Heston Stokastik Volatilite Modeli"""
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])

        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        # Heston parametreleri
        kappa = 1.5  # Mean reversion speed
        theta = sigma**2 * 0.8  # Long-term variance
        xi = 0.2  # Vol of vol
        rho = -0.3  # Correlation

        paths = np.zeros((days, simulations))
        vol_paths = np.zeros((days, simulations))

        paths[0] = last_price
        vol_paths[0] = theta

        for t in range(1, days):
            z1 = np.random.normal(0, 1, simulations)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, simulations)

            vol_paths[t] = np.maximum(
                vol_paths[t-1] + kappa * (theta - vol_paths[t-1]) * dt +
                xi * np.sqrt(np.maximum(vol_paths[t-1], 0.0001)) * np.sqrt(dt) * z2,
                0.0001
            )

            price_shock = (mu - 0.5 * vol_paths[t]) * dt + np.sqrt(vol_paths[t]) * np.sqrt(dt) * z1
            paths[t] = paths[t-1] * np.exp(price_shock)

        return paths, last_price, {'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho}

    def _jump_model(self, data, days=252, simulations=5000):
        """Jump Diffusion Model"""
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])

        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        # Jump parametreleri
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
        """Mean Reversion Model"""
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])

        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        last_price = prices[-1]
        dt = 1/252

        # Mean reversion
        long_term_mean = np.mean(prices[-252:]) if len(prices) > 252 else np.mean(prices)
        alpha = 0.1  # Mean reversion strength

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
        """Model performansına göre ağırlık hesapla"""
        print("📊 Ensemble model ağırlıkları hesaplanıyor...")

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

                    mape = mean_absolute_error(test_data['Close'].values, pred_means) / np.mean(test_data['Close'].values) * 100
                    errors.append(mape)

                if errors:
                    model_performances[model_name] = np.mean(errors)
                    print(f"   • {model_name:20} MAPE: %{np.mean(errors):.1f}")

            except Exception as e:
                continue

        # Performansı tersine çevir (düşük hata = yüksek ağırlık)
        if model_performances:
            total_performance = sum(1/p for p in model_performances.values())
            self.model_weights = {name: (1/performance)/total_performance for name, performance in model_performances.items()}

            print("\n🎯 MODEL AĞIRLIKLARI:")
            for name, weight in self.model_weights.items():
                print(f"   • {name:20} %{weight*100:.1f}")

        return self.model_weights

    def ensemble_predict(self, data, days=252, simulations=10000):
        """Ensemble tahmini"""
        if self.model_weights is None:
            self.calculate_model_weights(data)

        all_predictions = {}
        ensemble_paths = np.zeros((days, simulations))

        print("\n🔮 ENSEMBLE TAHMİN HESAPLANIYOR...")

        for model_name, model_func in self.models.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                try:
                    predictions, last_price, params = model_func(data, days=days, simulations=simulations)
                    all_predictions[model_name] = predictions

                    # Weighted contribution
                    ensemble_paths += predictions * weight
                    print(f"   • {model_name:20} %{weight*100:.1f} katkı")

                except Exception as e:
                    print(f"   • {model_name:20} HATA: {e}")
                    continue

        return ensemble_paths, last_price, all_predictions


class VolatilityRegimeDetector:
    """Volatilite rejimi tespit sistemi"""

    def __init__(self):
        self.regime_history = []
        self.regime_thresholds = {
            'low_vol': 0.15,    # %15 altı volatilite
            'normal_vol': 0.35, # %15-35 arası
            'high_vol': 0.60,   # %35-60 arası
            'extreme_vol': 1.0  # %60 üstü
        }

    def detect_regime(self, data, window=63):
        """Mevcut volatilite rejimini tespit et"""
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])

        # Rolling volatilite
        if len(returns) >= window:
            recent_returns = returns[-window:]
            current_vol = np.std(recent_returns) * np.sqrt(252)
        else:
            current_vol = np.std(returns) * np.sqrt(252)

        # Rejim tespiti
        if current_vol < self.regime_thresholds['low_vol']:
            regime = 'low_volatility'
        elif current_vol < self.regime_thresholds['normal_vol']:
            regime = 'normal_volatility'
        elif current_vol < self.regime_thresholds['high_vol']:
            regime = 'high_volatility'
        else:
            regime = 'extreme_volatility'

        self.regime_history.append({
            'date': data.index[-1] if hasattr(data, 'index') else len(data),
            'volatility': current_vol,
            'regime': regime
        })

        print(f"📊 VOLATILITE REJİMİ: {regime} (%{current_vol*100:.1f})")
        return regime, current_vol

    def get_regime_based_parameters(self, regime, base_mu, base_sigma):
        """Rejime göre parametre ayarı"""
        regime_multipliers = {
            'low_volatility': {'mu': 1.0, 'sigma': 0.8},
            'normal_volatility': {'mu': 1.0, 'sigma': 1.0},
            'high_volatility': {'mu': 0.9, 'sigma': 1.2},
            'extreme_volatility': {'mu': 0.7, 'sigma': 1.5}
        }

        multiplier = regime_multipliers.get(regime, regime_multipliers['normal_volatility'])

        adjusted_mu = base_mu * multiplier['mu']
        adjusted_sigma = base_sigma * multiplier['sigma']

        print(f"   • Getiri çarpanı: {multiplier['mu']} -> %{adjusted_mu*100:.1f}")
        print(f"   • Volatilite çarpanı: {multiplier['sigma']} -> %{adjusted_sigma*100:.1f}")

        return adjusted_mu, adjusted_sigma


class BayesianParameterOptimizer:
    """Bayesian parametre optimizasyonu"""

    def __init__(self):
        self.parameter_history = []

    def optimize_parameters(self, data, objective_function, n_iter=50):
        """Basit Bayesian optimizasyon"""
        print("🎯 Bayesian parametre optimizasyonu...")

        best_params = None
        best_score = -np.inf

        # Parametre space'i
        mu_range = np.linspace(0.1, 2.0, 20)  # Getiri çarpanları
        sigma_range = np.linspace(0.5, 2.0, 20)  # Volatilite çarpanları

        for i in range(n_iter):
            # Rastgele parametre seç
            mu_mult = np.random.choice(mu_range)
            sigma_mult = np.random.choice(sigma_range)

            try:
                score = objective_function(data, mu_mult, sigma_mult)

                if score > best_score:
                    best_score = score
                    best_params = {'mu_mult': mu_mult, 'sigma_mult': sigma_mult}

                    print(f"   • Iterasyon {i+1}: mu={mu_mult:.2f}, sigma={sigma_mult:.2f}, score={score:.4f}")

            except Exception as e:
                continue

        print(f"✅ EN İYİ PARAMETRELER: mu={best_params['mu_mult']:.2f}, sigma={best_params['sigma_mult']:.2f}")
        return best_params

    def sharpe_objective(self, data, mu_mult, sigma_mult):
        """Sharpe oranı maksimizasyonu"""
        prices = data['Close'].values
        returns = np.log(prices[1:] / prices[:-1])

        base_mu = np.mean(returns) * 252
        base_sigma = np.std(returns) * np.sqrt(252)

        adjusted_mu = base_mu * mu_mult
        adjusted_sigma = base_sigma * sigma_mult

        # Sharpe ratio (basitleştirilmiş)
        sharpe = adjusted_mu / adjusted_sigma if adjusted_sigma > 0 else 0

        return sharpe


class MultiTimeframeAnalyzer:
    """Çoklu zaman dilimi analizi"""

    def __init__(self):
        self.timeframes = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63
        }

    def analyze_timeframes(self, data):
        """Çoklu zaman dilimlerinde analiz"""
        print("🕒 Çoklu Zaman Dilimi Analizi...")

        results = {}
        prices = data['Close']

        for tf_name, tf_days in self.timeframes.items():
            # Zaman dilimine göre resample
            if tf_days > 1:
                tf_prices = prices.iloc[::tf_days]
            else:
                tf_prices = prices

            if len(tf_prices) > 10:
                tf_returns = np.log(tf_prices / tf_prices.shift(1)).dropna()

                mu = np.mean(tf_returns) * (252 / tf_days)
                sigma = np.std(tf_returns) * np.sqrt(252 / tf_days)

                results[tf_name] = {
                    'mu': mu,
                    'sigma': sigma,
                    'trend': 'BULL' if mu > 0 else 'BEAR',
                    'volatility_regime': 'HIGH' if sigma > 0.4 else 'LOW'
                }

                print(f"   • {tf_name:10} Getiri: %{mu*100:6.1f} Vol: %{sigma*100:5.1f} Trend: {results[tf_name]['trend']}")

        return results

    def get_consensus_signal(self, timeframe_results):
        """Zaman dilimleri konsensüs sinyali"""
        bullish_count = sum(1 for tf in timeframe_results.values() if tf['trend'] == 'BULL')
        total_timeframes = len(timeframe_results)

        consensus = bullish_count / total_timeframes if total_timeframes > 0 else 0.5

        if consensus >= 0.75:
            signal = "STRONG BULL"
        elif consensus >= 0.6:
            signal = "BULL"
        elif consensus <= 0.25:
            signal = "STRONG BEAR"
        elif consensus <= 0.4:
            signal = "BEAR"
        else:
            signal = "NEUTRAL"

        print(f"🎯 KONSENSUS SİNYAL: {signal} (%{consensus*100:.0f} bullish)")
        return signal, consensus


class DynamicPositionSizer:
    """Dinamik pozisyon büyüklüğü belirleme"""

    def __init__(self, max_position_size=0.1):  # Maksimum %10 pozisyon
        self.max_size = max_position_size
        self.risk_free_rate = 0.08  # %8 risksiz getiri

    def calculate_position_size(self, expected_return, volatility, confidence, total_capital=100000):
        """Risk-adjusted pozisyon büyüklüğü"""
        # Kelly Criterion benzeri formül
        if volatility > 0:
            # Temel Kelly: f = (mu - r) / sigma^2
            kelly_fraction = (expected_return - self.risk_free_rate) / (volatility ** 2)
        else:
            kelly_fraction = 0

        # Conservative adjustment
        conservative_fraction = kelly_fraction * 0.5  # Half-Kelly

        # Confidence adjustment
        confidence_adjusted = conservative_fraction * confidence

        # Cap at maximum position size
        final_fraction = min(confidence_adjusted, self.max_size)
        final_fraction = max(final_fraction, 0)  # No shorting

        position_value = total_capital * final_fraction

        print(f"💰 POZİSYON BÜYÜKLÜĞÜ ANALİZİ:")
        print(f"   • Beklenen getiri: %{expected_return*100:.1f}")
        print(f"   • Volatilite: %{volatility*100:.1f}")
        print(f"   • Kelly fraksiyonu: %{kelly_fraction*100:.1f}")
        print(f"   • Güven ayarlı: %{confidence_adjusted*100:.1f}")
        print(f"   • Son pozisyon: %{final_fraction*100:.1f} → {position_value:,.0f} TL")

        return final_fraction, position_value


class HamptonStyleAdvancedSystem:
    """Tüm improvement'ları içeren ana sistem"""

    def __init__(self):
        self.ensemble = EnsembleMonteCarlo()
        self.regime_detector = VolatilityRegimeDetector()
        self.parameter_optimizer = BayesianParameterOptimizer()
        self.timeframe_analyzer = MultiTimeframeAnalyzer()
        self.position_sizer = DynamicPositionSizer()

        self.performance_history = []

    def comprehensive_analysis(self, ticker="NATEN.IS", total_capital=100000):
        """Kapsamlı analiz ve tahmin"""
        print("ANALİZ")
        print("=" * 60)

        # Veri yükle
        data = yf.download(ticker, period="5y", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        print(f"✅ {len(data)} günlük veri yüklendi - {ticker}")

        # 1. Çoklu Zaman Dilimi Analizi
        print("\n1. 🔄 ÇOKLU ZAMAN DİLİMİ ANALİZİ")
        timeframe_results = self.timeframe_analyzer.analyze_timeframes(data)
        consensus_signal, consensus_strength = self.timeframe_analyzer.get_consensus_signal(timeframe_results)

        # 2. Volatilite Rejimi Tespiti
        print("\n2. 📊 VOLATİLİTE REJİMİ TESPİTİ")
        current_regime, current_vol = self.regime_detector.detect_regime(data)

        # 3. Bayesian Parametre Optimizasyonu
        print("\n3. 🎯 BAYESIAN PARAMETRE OPTİMİZASYONU")
        best_params = self.parameter_optimizer.optimize_parameters(
            data, self.parameter_optimizer.sharpe_objective
        )

        # 4. Ensemble Model Tahmini
        print("\n4. 🔮 ENSEMBLE MODEL TAHMİNİ")
        ensemble_predictions, last_price, individual_predictions = self.ensemble.ensemble_predict(data)

        # 5. Sonuç Analizi
        print("\n5. 📈 SONUÇ ANALİZİ ve POZİSYON BELİRLEME")
        final_prices = ensemble_predictions[-1, :]

        # Temel istatistikler
        mean_prediction = np.mean(final_prices)
        median_prediction = np.median(final_prices)
        confidence_95 = np.percentile(final_prices, [2.5, 97.5])
        profit_probability = np.mean(final_prices > last_price) * 100

        # Beklenen getiri ve volatilite
        expected_return = (mean_prediction / last_price - 1)
        expected_volatility = np.std(final_prices) / last_price

        print(f"\n🎯 1 YILLIK TAHMİN SONUÇLARI:")
        print(f"   • Mevcut Fiyat: {last_price:.2f} TL")
        print(f"   • Ortalama Tahmin: {mean_prediction:.2f} TL (%{expected_return*100:.1f})")
        print(f"   • Medyan Tahmin: {median_prediction:.2f} TL")
        print(f"   • %95 Güven Aralığı: {confidence_95[0]:.2f} - {confidence_95[1]:.2f} TL")
        print(f"   • Kazanç Olasılığı: %{profit_probability:.1f}")
        print(f"   • Konsensüs Sinyal: {consensus_signal}")
        print(f"   • Volatilite Rejimi: {current_regime}")

        # 6. Pozisyon Büyüklüğü Belirleme
        confidence = consensus_strength * (profit_probability / 100)
        position_size, position_value = self.position_sizer.calculate_position_size(
            expected_return, expected_volatility, confidence, total_capital
        )

        # 7. Hedef Fiyat Analizi
        print(f"\n🎯 HEDEF FİYAT ANALİZİ:")
        targets = [1.10, 1.20, 1.30, 1.50, 2.00]
        for target in targets:
            target_price = last_price * target
            prob = np.mean(final_prices >= target_price) * 100
            print(f"   • %{(target-1)*100:.0f} Kazanç ({target_price:.2f} TL): %{prob:.1f} olasılık")

        # 8. Risk Metrikleri
        print(f"\n⚠️  RİSK METRİKLERİ:")
        var_95 = np.percentile((final_prices - last_price) / last_price, 5) * 100
        cvar_95 = np.mean([x for x in (final_prices - last_price) / last_price if x <= np.percentile((final_prices - last_price) / last_price, 5)]) * 100

        print(f"   • Value at Risk (95%): %{var_95:.1f}")
        print(f"   • Expected Shortfall (95%): %{cvar_95:.1f}")
        print(f"   • Maximum Drawdown Risk: %{abs(min((final_prices - last_price) / last_price)) * 100:.1f}")

        # Sonuç özeti
        print(f"\n💎 YATIRIM ÖNERİSİ:")
        if position_size > 0.05:
            print(f"   ✅ POZİSYON AL: %{position_size*100:.1f} ({position_value:,.0f} TL)")
            print(f"   🎯 HEDEF: {median_prediction:.2f} TL (%{((median_prediction/last_price)-1)*100:.1f} kazanç)")
            print(f"   🛡️  STOP: {last_price * 0.85:.2f} TL (%15 kayıp)")
        elif position_size > 0.02:
            print(f"   ⚠️  KÜÇÜK POZİSYON: %{position_size*100:.1f} ({position_value:,.0f} TL)")
            print(f"   🔍 DİKKATLİ İZLE")
        else:
            print(f"   ❌ POZİSYON ALMA: Çok riskli/yetersiz getiri")

        return {
            'current_price': last_price,
            'mean_prediction': mean_prediction,
            'median_prediction': median_prediction,
            'confidence_95': confidence_95,
            'profit_probability': profit_probability,
            'consensus_signal': consensus_signal,
            'volatility_regime': current_regime,
            'position_size': position_size,
            'position_value': position_value,
            'expected_return': expected_return,
            'risk_metrics': {'var_95': var_95, 'cvar_95': cvar_95}
        }


def main():
    print("SİSTEM ÇALIŞTIRILIYOR...")
    print("=" * 60)

    system = HamptonStyleAdvancedSystem()

    try:
        results = system.comprehensive_analysis("NATEN.IS", total_capital=100000)

        print(f"\n" + "=" * 60)
        print(" ANALİZ TAMAMLANDI!")

        return results

    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()