import pandas as pd
import matplotlib.pyplot as plt

# List the CSV files from 5 runs
files = [
    './results/naive_with_lr_ab/0.001/data_lr_0.001_1.csv',
    './results/naive_with_lr_ab/0.001/data_lr_0.001_2.csv',
    './results/naive_with_lr_ab/0.001/data_lr_0.001_3.csv',
    './results/naive_with_lr_ab/0.001/data_lr_0.001_4.csv',
    './results/naive_with_lr_ab/0.001/data_lr_0.001_5.csv'
]

dfs = [pd.read_csv(f) for f in files]

env_steps = dfs[0]['env_step']

returns_ep = pd.concat([df['Episode_Return'] for df in dfs], axis=1)
mean_ep = returns_ep.mean(axis=1)

returns_ep_smooth = pd.concat([df['Episode_Return_smooth'] for df in dfs], axis=1)
mean_ep_smooth = returns_ep_smooth.mean(axis=1)
std_ep_smooth = returns_ep_smooth.std(axis=1)

aggregated = pd.DataFrame({
    'Episode_Return': mean_ep,
    'Episode_Return_smooth': mean_ep_smooth,
    'env_step': env_steps
})

aggregated.to_csv('./results/naive_with_lr_ab/0.001/0.001.csv', index=False)
print("File saved")

plt.figure(figsize=(10, 6))
plt.plot(aggregated['env_step'], aggregated['Episode_Return_smooth'], label='Mean Smoothed Return')
plt.fill_between(
    aggregated['env_step'], 
    aggregated['Episode_Return_smooth'] - std_ep_smooth,
    aggregated['Episode_Return_smooth'] + std_ep_smooth,
    alpha=0.2, label='Std Dev'
)
plt.xlabel("Environment Steps")
plt.ylabel("Episode Return (Smoothed)")
plt.title("DQN Training, LR = 0.001 (Mean of 5 Runs)")
plt.legend()
plt.show()
