import os
import torch
import random
import numpy as np
from collections import deque
from environment import SnakeGameAI, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.record = 0
        self.total_score = 0

        model_path = './model/model.pth'
        record_path = './model/record.txt'

        if os.path.exists(model_path):
            try:
                # 1. Load the pre-trained neural network weights
                # weights_only=True is standard practice for secure local loading
                self.model.load_state_dict(torch.load(model_path, weights_only=True)) # load brain
                print("Model weights loaded successfully inside Agent!")

                # 2. Check if we have a matching high-score tracker file
                if os.path.exists(record_path):
                    with open(record_path, 'r') as f:
                        lines = f.read().splitlines()
                        if lines:
                            self.record = int(lines[0].strip())

                            # Optional quality-of-life add: restore game count to preserve Epsilon decay
                            if len(lines) > 1:
                                self.n_games = int(lines[1].strip())

                            if len(lines) > 2:
                                self.total_score = int(lines[2].strip())

                    print(f"Agent Restored: Record={self.record} | Games={self.n_games} | Historical Avg={(self.total_score/max(1, self.n_games)):.2f}")
            except Exception as e:
                print(f"Problem restoring previous training state: {e}")

    def get_state(self, game):
        head = game.head
        
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            
            game.food[0] < head[0], # food left
            game.food[0] > head[0], # food right
            game.food[1] < head[1], # food up
            game.food[1] > head[1]  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move
