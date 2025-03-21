import pandas as pd
import matplotlib.pyplot as plt

my_result = pd.read_csv('./results/naive_with_lr_ab/0.001/0.001.csv')

# baseline dataset
baseline_results = pd.read_csv('./results/tn_er/tn_er.csv')  

# Plot comparison
plt.figure(figsize=(12, 8))
plt.plot(my_result['env_step'], my_result['Episode_Return_smooth'], label='Naive Data')
plt.plot(baseline_results['env_step'], baseline_results['Episode_Return_smooth'], label='TN ER Data')
plt.xlabel('Environment Steps')
plt.ylabel('Smoothed Return')
plt.title('Comparison of DQN tn_er and naive performance')
plt.legend()
plt.show()