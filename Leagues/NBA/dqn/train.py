import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
from dqn import Agent
from envs.plonn_nba_env import NBAPlonnENV

# Load and clean NBA Betting Dataset
def load_and_clean_data(filepath, sheet_name):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Trim column names
    df.columns = df.columns.astype(str).str.strip()
    
    # Keep only relevant columns
    columns_to_keep = ['home', 'away', "dk lines", "home pred", "away pred", "pred total", "Total Actual",  "Accuracy %"]
    df = df[columns_to_keep].copy()
    
    # One-hot encode team names
    home_teams = pd.get_dummies(df["home"], prefix="home")
    away_teams = pd.get_dummies(df["away"], prefix="away")
    df = pd.concat([df, home_teams, away_teams], axis=1)
    df.drop(columns=["home", "away"], inplace=True)
    # Fill missing values in Accuracy % with the mean accuracy
        # Normalize numeric values for stable training
    for col in ["dk lines", "home pred", "away pred", "pred total", "Total Actual"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Compute cumulative win rate (W/G) as a constant
    
    
    
    
    return df  # Ensure it returns a DataFrame

# Filepath to the dataset
file_path = "data/nba.xlsx"
sheet_name = "30dw"

# Load cleaned dataset as DataFrame
data = load_and_clean_data(file_path, sheet_name)

# Load NBA Betting Environment
env = NBAPlonnENV(data)  # Ensure we pass a DataFrame
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize DQN Agent
agent = Agent(state_size, action_size)

# Training Parameters
episodes = 323 * 10 # Number of games
update_target_every = 5  # Update target network every X episodes

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        # Prevent out-of-bounds errors
        
        if env.current_index >= len(env.data) - 1:
            print(f"Out of data at index {env.current_index}, max index is {len(env.data) - 1}. Ending episode.")
            done = True  # End the episode if we run out of data
            break

        next_state, reward, done, _ = env.step(action)

        reward = max(-1, min(1, reward))

        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    # Update target network periodically
    if episode % update_target_every == 0:
        agent.update_target_network()
    
    print(f"Episode {episode+1}/{episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.4f}")

# Save the trained model
agent.q_network.save("dqn_nba_betting_model.h5")
print("Training completed and model saved!")
