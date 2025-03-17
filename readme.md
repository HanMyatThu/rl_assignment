# RL Assignment - Deep Q-Learning for CartPole

This repository contains the implementation of a Deep Q-Network (DQN) agent for solving the CartPole environment as part of the RL Assignment. The project explores both theoretical and experimental aspects of Deep Q-Learning, including:

- Derivation of the DQN loss function from the Bellman equation.
- Discussion on why tabular update rules are inadequate when using neural network function approximation.
- Implementation of a naive DQN (without Experience Replay (ER) and Target Networks (TN)) with hyperparameter ablation studies.
- Comparison of four configurations: Naive, Only TN, Only ER, and TN & ER.
- Aggregation of experimental results over multiple runs to generate robust learning curves.

## Repository Structure

- dataset -> baseline data
- results -> results photo, csv files from each runs
- dqn.py -> dqn function
- compute_average.py -> compute the mean value from 5 runs
- compare.py -> compare two results


## Requirements

Ensure you have Python 3 installed. Install the required packages using:

```bash
pip install -r requirements.txt


## Instructions

Naive DQN Hyperparameter Ablation
To study the effect of different learning rates on a naive DQN (without ER and TN) with an update-to-data ratio of 1, run:

```bash
python dqn_experiments.py --mode naive --learning_rate 0.0001 --update_to_data 1
python dqn_experiments.py --mode naive --learning_rate 0.0005 --update_to_data 1
python dqn_experiments.py --mode naive --learning_rate 0.001 --update_to_data 1
```

# ER/TN Configuration Comparison
To compare the inclusion of Target Networks and Experience Replay, run:

python dqn_experiments.py --mode only_tn
python dqn_experiments.py --mode only_er
python dqn_experiments.py --mode tn_er

# Data Comparison
modify the source file, change file names, and run
python compare.py

# Computing mean value

python compute_average.py