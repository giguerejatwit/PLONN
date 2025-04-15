import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class NBAPlonnENV(gym.Env):
    def __init__(self, data):
        super(NBAPlonnENV, self).__init__()
        
        # Ensure column names are stripped of extra spaces
        if isinstance(data, pd.DataFrame):
            data.columns = data.columns.astype(str).str.strip()
        self.data = data.reset_index(drop=True)
        
        self.current_index = 0  # Track current game
        
        # Determine the number of team features (one-hot encoded teams)
        team_feature_size = len([col for col in data.columns if col.startswith("home_") or col.startswith("away_")])
        
        # State: Predicted scores, DraftKings Lines, Team One-Hot Encodings
        state_size = 5 + team_feature_size  # 5 existing features + team features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        
        # Actions: [0] = No Bet, [1] = Over, [2] = Under
        self.action_space = spaces.Discrete(3)
    
    def reset(self):
        self.current_index = 0  # Restart from first game
        return self._get_observation(), {}
    
    def _get_observation(self):
        row = self.data.iloc[self.current_index]
        
        # Extract numeric features
        numeric_features = np.array([
            row['dk lines'], 
            row['home pred'], 
            row['away pred'], 
            row['pred total'], 
            
        ], dtype=np.float32)
        
        # Extract team one-hot encoded features
        team_features = row[[col for col in self.data.columns if col.startswith("home_") or col.startswith("away_")]].values.astype(np.float32)
        
        # Concatenate numeric features with team features
        return np.concatenate([numeric_features, team_features])
    
    def step(self, action):
        row = self.data.iloc[self.current_index]
        actual_total = row['Total Actual']
        dk_line = row['dk lines']
        reward = 0

        if action == 1 and actual_total > dk_line:
            reward = 2  # Correct Over Bet
        elif action == 2 and actual_total < dk_line:
            reward = 2  # Correct Under Bet
        elif action == 0:
            reward = -0.1  # Penalty for No Bet
        else:
            reward = -0.5  # Wrong Bet
        
        self.current_index += 1
        done = self.current_index >= len(self.data)
        
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        pass
