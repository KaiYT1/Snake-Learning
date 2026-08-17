# Snake Game Reinforcement Learning Agent

A local Deep Q-Network (DQN) agent built from scratch in Python to play the classic Snake game using PyTorch, Pygame-CE, and Matplotlib.

## How the Agent Thinks
The agent uses a 2-layer Feed-Forward Neural Network to predict the best action based on an **11-element binary state vector**:
* **Danger Vectors (3):** Is there an obstacle (wall/body) straight ahead, to the right, or to the left relative to the current direction?
* **Heading Orientation (4):** Is the snake currently moving UP, DOWN, LEFT, or RIGHT?
* **Food Proximity (4):** Is the food located above, below, left, or right of the snake's head?

## 🏆 Reward System
* **+10.0** for eating food (positive reinforcement).
* **-10.0** for crashing into a wall or its own body (negative reinforcement).
* **-0.1** step penalty for every frame survived without food (incentivizes the agent to take efficient paths and prevents it from spinning in infinite circles).

## How To Use

### 1. Installation
Clone the repository and install the tracked dependencies inside your virtual environment:
```powershell
python -m pip install -r requirements.txt
```

### 2. Run Training
```powershell
python main.py
```

## Results
A live Matplotlib interface will open during execution to map out the historical game scores and the overall running mean score.
