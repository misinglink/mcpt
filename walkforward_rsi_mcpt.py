from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from tqdm import tqdm

from bar_permute import get_permutation
from donchian import  walkforward_donch
from rsi import walkforward_rsi

client = MongoClient("mongodb://localhost:37017/")
db = client["BTC_data"]
collection = db["candles_1m"]

start_day = "2025-05-25"
start_day_s = datetime.strptime(start_day, "%Y-%m-%d")
start_day_ms = int(start_day_s.timestamp() * 100)  # assuming 't' is in ms
# Load all documents from the collection into a DataFrame
df = pd.DataFrame(list(collection.find({"t": {"$gte": start_day_ms}})))
print(f"Amount of data {len(df)} minutes.")
df['r'] = np.log(df['c']).diff().shift(-1)

train_window = len(df) - (60 * 5)

df['rsi_wf_signal'] = walkforward_donch(df, train_lookback=train_window)

rsi_rets = df['rsi_wf_signal'] * df['r']
real_wf_pf = rsi_rets[rsi_rets > 0].sum() / rsi_rets[rsi_rets < 0].abs().sum()

n_permutations = 200
perm_better_count = 1
permuted_pfs = []
print("Walkforward MCPT")
for perm_i in tqdm(range(1, n_permutations)):
    wf_perm = get_permutation(df, start_index=train_window)
    
    wf_perm['r'] = np.log(wf_perm['c']).diff().shift(-1) 
    wf_perm_sig = walkforward_rsi(wf_perm, train_lookback=train_window)
    perm_rets = wf_perm['r'] * wf_perm_sig
    perm_pf = perm_rets[perm_rets > 0].sum() / perm_rets[perm_rets < 0].abs().sum()
    
    if perm_pf >= real_wf_pf:
        perm_better_count += 1

    permuted_pfs.append(perm_pf)


walkforward_mcpt_pval = perm_better_count / n_permutations
print(f"Walkforward MCPT P-Value: {walkforward_mcpt_pval}")


plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Permutations')
plt.axvline(real_wf_pf, color='red', label='Real')
plt.xlabel("Profit Factor")
plt.title(f"Walkforward MCPT. P-Value: {walkforward_mcpt_pval}")
plt.grid(False)
plt.legend()
plt.show()

