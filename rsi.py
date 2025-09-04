
from itertools import product

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from db_utils import fetch_candle_data



def calculate_rsi(data, period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def evaluate_params(period, lower_bound, upper_bound, data):
    df = data.copy()
    rsi_lookback = period + 1
    df['rsi'] = calculate_rsi(df, period)
    df["signal"] = detect_divergence_convergence_signal(df, lookback=rsi_lookback)
    r = np.log(df['c']).diff().shift(-1)
    sig_rets = df["signal"] * r
    sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum() if sig_rets[sig_rets < 0].abs().sum() != 0 else np.nan
    sig_rets = sig_rets.dropna()
    if sig_rets.std() != 0:
        sharpe_ratio = (sig_rets.mean() / sig_rets.std())
        annualized_sharpe_ratio = sharpe_ratio * np.sqrt(365)
    else:
        sharpe_ratio = annualized_sharpe_ratio = 0

    cum_r = sig_rets.cumsum()
    initial_portfolio = 10000
    portfolio = initial_portfolio * np.exp(cum_r)

    return {
        "period": period,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "profit_factor": sig_pf,
        "sharpe_ratio": sharpe_ratio,
        "annualized_sharpe_ratio": annualized_sharpe_ratio,
        "cumulative_return": cum_r,
        "portfolio_return": portfolio,
    }


def find_optimal_rsi_params(data, rsi_periods, rsi_lower_bounds, rsi_upper_bounds):
    """Find the optimal RSI period for the given data."""
    results = []
    param_combos = list(product(rsi_periods, rsi_lower_bounds, rsi_upper_bounds))
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(evaluate_params)(period, lower, upper, data)
        for period, lower, upper in tqdm(param_combos, desc="Optimizing RSI Params")
    )
    # Return the result with the maximum annualized_sharpe_ratio
    return max(results, key=lambda x: x['annualized_sharpe_ratio'])

# def get_rsi_signal(ohlc: pd.DataFrame, period: int, lower_bound: int, upper_bound: int):
#     """Get RSI signal for the given parameters."""
#     df = ohlc.copy()
#     df['rsi'] = calculate_rsi(df, period)
#     df["signal"] = np.where(df['rsi'] <= lower_bound, 1, np.where(df['rsi'] >= upper_bound, -1, 0))
#     return df["signal"].ffill()

def detect_divergence_convergence_signal(data, lookback=5):
    """Detect RSI divergence and convergence with price."""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['c']
    signals['rsi'] = data['rsi']
    signals['divergence'] = np.zeros(len(signals))  # 1 for bullish, -1 for bearish
    signals['convergence'] = np.zeros(len(signals))  # 1 for bullish, -1 for bearish

    for i in range(lookback, len(data)):
        # Get price and RSI highs/lows in lookback period
        price_window = data['c'].iloc[i-lookback:i+1]
        rsi_window = signals['rsi'].iloc[i-lookback:i+1]
        
        price_high_idx = price_window.idxmax()
        price_low_idx = price_window.idxmin()
        rsi_high_idx = rsi_window.idxmax()
        rsi_low_idx = rsi_window.idxmin()

        # Bullish Divergence: Lower price low but higher RSI low
        if (price_window[price_low_idx] < price_window.shift(1).min()) and \
           (rsi_window[rsi_low_idx] > rsi_window.shift(1).min()):
            signals.iloc[i, signals.columns.get_loc('divergence')] = 1
        
        # Bearish Divergence: Higher price high but lower RSI high
        if (price_window[price_high_idx] > price_window.shift(1).max()) and \
           (rsi_window[rsi_high_idx] < rsi_window.shift(1).max()):
            signals.iloc[i, signals.columns.get_loc('divergence')] = -1
        
        # Bullish Convergence: Higher price high and higher RSI high
        if (price_window[price_high_idx] > price_window.shift(1).max()) and \
           (rsi_window[rsi_high_idx] > rsi_window.shift(1).max()):
            signals.iloc[i, signals.columns.get_loc('convergence')] = 1
        
        # Bearish Convergence: Lower price low and lower RSI low
        if (price_window[price_low_idx] < price_window.shift(1).min()) and \
           (rsi_window[rsi_low_idx] < rsi_window.shift(1).min()):
            signals.iloc[i, signals.columns.get_loc('convergence')] = -1

    # Combine divergence and convergence signals
    signals['signal'] = signals['divergence'] + signals['convergence']
    signals['signal'] = signals['signal'].replace({2: 1, -2: -1})  # Convert to 1 for bullish, -1 for bearish
    signals['signal'] = signals['signal'].replace({-2: -1})

    return signals['signal']#.ffill()

def walkforward_rsi(
    ohlc: pd.DataFrame, 
    train_lookback: int = 60 * 24 * 30, 
    train_step: int = 24 * 30,
    rsi_periods=range(5, 50),
    rsi_lower_bounds=range(10, 35),
    rsi_upper_bounds=range(65, 90)
):
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            opt_params = find_optimal_rsi_params(
                ohlc.iloc[i-train_lookback:i], 
                rsi_periods, 
                rsi_lower_bounds, 
                rsi_upper_bounds
            )
            # tmp_signal = get_rsi_signal(
            #     ohlc, 
            #     opt_params["period"], 
            #     opt_params["lower_bound"], 
            #     opt_params["upper_bound"]
            # )
            next_train += train_step
        
        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


if __name__ == "__main__":
    print(f"Sharpe Ratio Factor: 365: {np.sqrt(365*24*60)}, 252: {np.sqrt(252*24*60)}")

    # MongoDB connection details
    MONGO_URI = "mongodb://localhost:27017/"
    COIN = "HYPE"
    DB_NAME = f"{COIN}_data"
    COLLECTION_NAME = "candles_1m"

    # Fetch candle data
    candle_data = fetch_candle_data(MONGO_URI, DB_NAME, COLLECTION_NAME)

    # Ensure data is sorted by time and clean up
    candle_data = candle_data[['t', 'c']].dropna()

    # Find  RSI
    rsi_periods = range(20, 31) 
    rsi_lower_bounds = range(10, 25)
    rsi_upper_bounds = range(80, 90)
    result = find_optimal_rsi_params(candle_data, rsi_periods, rsi_lower_bounds, rsi_upper_bounds)

    optimal_period = result['period']
    profit_factor = result['profit_factor']
    plt.figure(figsize=(12, 6))
    
    plt.plot(result['cumulative_return'].values)
    plt.title('Cumulative Returns for All RSI Parameter Sets')
    plt.xlabel('Time Index')
    plt.ylabel('Cumulative Return')
    plt.text(
        0.02, 0.95,
        f"Sharpe Ratio: {result['annualized_sharpe_ratio']:.2f}\nProfit Factor: {result['profit_factor']:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('all_cumulative_returns.png')
    plt.show()

    plt.plot(result['portfolio_return'].values)
    plt.title('Cumulative Returns for All RSI Parameter Sets')
    plt.xlabel('Time Index')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('all_cumulative_returns.png')
    plt.show()


    print(f"Optimal Scenario Parameters: {result}")

