import numpy as np
import pandas as pd
from typing import List, Union

from pymongo import MongoClient
import matplotlib.pyplot as plt

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]
    
    n_bars = len(ohlc[0])


    perm_index = start_index + 1
    perm_n = n_bars - perm_index


    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['o', 'h', 'l', 'c']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['o'] - log_bars['c'].shift()).to_numpy()
        
        # Get prices relative to this bars open
        r_h = (log_bars['h'] - log_bars['o']).to_numpy()
        r_l = (log_bars['l'] - log_bars['o']).to_numpy()
        r_c = (log_bars['c'] - log_bars['o']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) seprately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['o', 'h', 'l', 'c']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['o', 'h', 'l', 'c'])

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]

if __name__ == '__main__':
    COIN = "TRUMP"
    
    client = MongoClient("mongodb://localhost:37017/")
    db = client[F"{COIN}_data"]
    collection = db["candles_1m"]

    # Load all documents from the collection into a DataFrame
    real_df = pd.DataFrame(list(collection.find()))

    perm_df = get_permutation(real_df)
    real_r = np.log(real_df['c']).diff() 
    perm_r = np.log(perm_df['c']).diff()

    print(f"Mean. REAL: {real_r.mean():14.6f} PERM: {perm_r.mean():14.6f}")
    print(f"Stdd. REAL: {real_r.std():14.6f} PERM: {perm_r.std():14.6f}")
    print(f"Skew. REAL: {real_r.skew():14.6f} PERM: {perm_r.skew():14.6f}")
    print(f"Kurt. REAL: {real_r.kurt():14.6f} PERM: {perm_r.kurt():14.6f}")

    # eth_real = pd.read_csv('data/HYPE_price_history_coinlore.csv')
    # eth_real.index = eth_real.index.astype('datetime64[s]')
    # eth_real = eth_real[(eth_real.index.year >= 2018) & (eth_real.index.year < 2020)]
    # eth_real_r = np.log(eth_real['close']).diff()
    
    # # print("") 

    # permed = get_permutation([btc_real, eth_real])
    # btc_perm = permed[0]
    # eth_perm = permed[1]

    # perm_profit = perm_log_close.cumsum()

    # eth_perm_r = np.log(eth_perm['close']).diff()
    # print(f"BTC&ETH Correlation REAL:/ {btc_real_r.corr(eth_real_r):5.3f} PERM: {btc_perm_r.corr(eth_perm_r):5.3f}")

    plt.style.use("dark_background")    
    np.log(real_df['c']).diff().cumsum().plot(color='blue', label=f'{COIN}USD')
    np.log(perm_df['c']).diff() .cumsum().plot(color='gray', label=f'{COIN}USD')
    # np.log(eth_perm['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    plt.title(f"{COIN} permuted data vs real log data LOG return")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.show()



########## REAL PRICES #########
    plt.style.use("dark_background")    
    real_df["c"].plot(color='blue', label=f'{COIN}USD')
    # np.log(eth_real['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    
    # plt.ylabel("Cumulative Log Return")
    # plt.title("Real HYPEUSD")
    # plt.legend()
    # plt.show()

    perm_df["c"].plot(color='gray', label=f'{COIN}USD')
    # np.log(eth_perm['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    plt.title(f"{COIN} permuted prices vs real log data")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


