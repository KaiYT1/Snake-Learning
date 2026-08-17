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
        self.model = Linear_QNet(17, 256, 3) # input states, nodes, output states
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

    def _get_ray_distances(self, game, direction_str):
        """Casts a ray in a given direction and returns distance to danger & food"""
        x, y = game.head[0], game.head[1]

        # Determine the grid step vector for this absolute direction
        dx, dy = 0, 0
        if direction_str == "RIGHT": dx = BLOCK_SIZE
        elif direction_str == "LEFT": dx = -BLOCK_SIZE
        elif direction_str == "DOWN": dy = BLOCK_SIZE
        elif direction_str == "UP": dy = -BLOCK_SIZE

        distance = 0.0
        danger_dist = 1.0 # 1.0 means no danger seen (out of bounds)
        food_dist = 1.0

        # Max steps across the screen boundary box
        max_steps = max(game.w, game.h) // BLOCK_SIZE

        for step in range(1, max_steps + 1):
            check_pt = [x + (dx * step), y + (dy * step)]
            distance_normalized = step / max_steps

            # Track if food is on this ray path
            if check_pt == game.food and food_dist == 1.0:
                food_dist = distance_normalized

            # Track if a collision hazard is on this ray path
            if game.is_collision(check_pt) and danger_dist == 1.0:
                danger_dist = distance_normalized
                break # Stop casting once we find the closest structural wall/body

        return danger_dist, food_dist

    def get_state(self, game):
        head = game.head

        # 1. Map relative directions (Straight, Right, Left) to absolute grid layout headings
        clock_wise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clock_wise.index(game.direction)
        
        dir_straight = clock_wise[idx]
        dir_right = clock_wise[(idx + 1) % 4]
        dir_left = clock_wise[(idx - 1) % 4]

        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        # 2. Extract long-range distances via Raycasting
        danger_s, food_s = self._get_ray_distances(game, dir_straight)
        danger_r, food_r = self._get_ray_distances(game, dir_right)
        danger_l, food_l = self._get_ray_distances(game, dir_left)
        
        # 3. Retain core basic states for backup stability
        head = game.head
        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        # 4. Construct our 17-element advanced perception array
        state = [
            # Long-range distance perception variables (6 values)
            danger_s, danger_r, danger_l,
            food_s, food_r, food_l,

            # Same as before indicators (11 values)
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
        return np.array(state, dtype=float)

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
