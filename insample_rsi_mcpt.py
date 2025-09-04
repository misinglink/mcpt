from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymongo import MongoClient

from bar_permute import get_permutation
from rsi import find_optimal_rsi_params
    
client = MongoClient("mongodb://localhost:27017/")
db = client["HYPE_data"]
collection = db["candles_1m"]

start_day = "2025-05-20"
start_day_s = datetime.strptime(start_day, "%Y-%m-%d")
start_day_ms = int(start_day_s.timestamp() * 100)  # assuming 't' is in ms
# Load all documents from the collection into a DataFrame
df = pd.DataFrame(list(collection.find({"t": {"$gte": start_day_ms}})))
print(f"Amount of data {len(df)}")

# Ensure the index is a datetime index
df['t'] = pd.to_datetime(df['t'], unit='ms')
df = df.set_index('t')

# train_df = df[(df.index.year >= 2016) & (df.index.year < 2020)]
## rsi params 
rsi_periods = range(20, 50)
rsi_lower_bounds = range(10, 25)
rsi_upper_bounds = range(75, 90)
opt_real_params = find_optimal_rsi_params(df, rsi_periods, rsi_lower_bounds, rsi_upper_bounds)
# opt_real_params = find_optimal_rsi_params(df)

print(opt_real_params)
print(
    "In-sample PF:", opt_real_params["profit_factor"],
    "Best Period:", opt_real_params.get("period"),
    "RSI Period Lower Bound:", opt_real_params.get("lower_bound"),
    "RSI Period Upper Bound:", opt_real_params.get("upper_bound")
)


n_permutations = 1000
perm_better_count_pf = 0
perm_better_count_sr = 0
perm_both_better_count = 0
permuted_optimals = []
print("In-Sample MCPT")
for perm_i in tqdm(range(1, n_permutations)):
    train_perm = get_permutation(df)
    opt_perm_params = find_optimal_rsi_params(train_perm, rsi_periods, rsi_lower_bounds, rsi_upper_bounds)

    with open("insample_mcpt_results.txt", "a+") as f:
        f.write(f"{opt_perm_params}\n")

    pf_better = opt_perm_params["profit_factor"] >= opt_real_params["profit_factor"]
    sr_better = opt_perm_params["sharpe_ratio"] >= opt_real_params["sharpe_ratio"]
    if pf_better:
        perm_better_count_pf += 1
    if sr_better:
        perm_better_count_sr += 1
    if pf_better and sr_better:
        perm_both_better_count += 1
    

    permuted_optimals.append(opt_perm_params)

insample_mcpt_pval_pf = perm_better_count_pf / n_permutations
insample_mcpt_pval_sr = perm_better_count_sr / n_permutations
insample_mcpt_pval = perm_both_better_count / n_permutations
print(f"In-sample MCPT Profit Factor P-Value: {insample_mcpt_pval_pf}")
print(f"In-sample MCPT Sharpe Ratio P-Value: {insample_mcpt_pval_sr}")
print(f"In-sample MCPT Both P-Value: {insample_mcpt_pval}")

pfs = [opt["profit_factor"] for opt in permuted_optimals]
plt.style.use('dark_background')
pd.Series(pfs).hist(color='blue', label='Permutations')
plt.axvline(opt_real_params["profit_factor"], color='red', label='Real')
plt.xlabel("Profit Factor")
plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval_pf}")
plt.grid(False)
plt.legend()
plt.show()

srs = [opt["sharpe_ratio"] for opt in permuted_optimals]
plt.style.use('dark_background')
pd.Series(srs).hist(color='blue', label='Permutations')
plt.axvline(opt_real_params["sharpe_ratio"], color='red', label='Real')
plt.xlabel("Sharpe Ratio")
plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval_sr}")
plt.grid(False)
plt.legend()
plt.show()


