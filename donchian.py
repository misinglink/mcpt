import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def donchian_breakout(ohlc: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    upper = ohlc['c'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['c'].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc['c'] > upper] = 1
    signal.loc[ohlc['c'] < lower] = -1
    signal = signal.ffill()
    return signal

def optimize_donchian(ohlc: pd.DataFrame):

    best_pf = 0
    best_lookback = -1
    r = np.log(ohlc['c']).diff().shift(-1)
    for lookback in range(12, 169):
        signal = donchian_breakout(ohlc, lookback)
        sig_rets = signal * r
        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf

def walkforward_donch(
        ohlc: pd.DataFrame, 
        train_lookback: int = 60 * 24 * 30, 
        train_step: int = 24 * 30
    ):

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None
    
    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_lookback, _ = optimize_donchian(ohlc.iloc[i-train_lookback:i])
            tmp_signal = donchian_breakout(ohlc, best_lookback)
            next_train += train_step
        
        wf_signal[i] = tmp_signal.iloc[i]
    
    return wf_signal

if __name__ == '__main__':

    # df = pd.read_csv('data/HYPE_price_history_coinlore.csv')
    # df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].replace({'\\$': ''}, regex=True).astype(float)
    # df["date"] = pd.to_datetime(df["date"], format='%m/%d/%Y')
    # df = df.set_index('date')
    # df.index = df.index.astype('datetime64[s]')

    # BTC load in   
    df = pd.read_csv('data/BTC_price_history_coinmarketcap.csv', sep=';')
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    df = df.set_index('t')




    # df = df[(df.index.year >= 2016) & (df.index.year < 2020)] 
    best_lookback, best_real_pf = optimize_donchian(df)

    print(f"Best Lookback: {best_lookback}, Best Real Profit Factor: {best_real_pf}")
    
    signal = donchian_breakout(df, best_lookback) 

    df['r'] = np.log(df['close']).diff().shift(-1)
    df['donch_r'] = df['r'] * signal

    plt.style.use("dark_background")
    df['donch_r'].cumsum().plot(color='red')
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel('Cumulative Log Return')
    plt.show()


