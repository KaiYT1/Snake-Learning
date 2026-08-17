import torch
import torch.nn as nn
import torch.optim as optim
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Linear layer 1: map 11 inputs to hidden nodes (hidden_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Linear layer 2: map hidden nodes to 3 action outputs (output nodes (output_size))
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply ReLU activation function to the output of the first linear layer (hidden layer)
        x = torch.relu(self.linear1(x))
        # Output as raw prediction values (Q-values)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Create directory if it doesn't exist
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Save the model state dictionary to the specified file
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error loss function for regression tasks

    def train_step(self, state, action, reward, next_state, done):
        # Convert lists/numpy arrays to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # Tensor conversion helper function to ensure correct shape for single samples

        # Handle cases train on single step vs mini-batch
        if len(state.shape) == 1:
            # Add batch dimension if it's single state (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # Convert done to a tuple for consistency

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            # Use Bellman equation: Q_new = R + gamma * max(next_predicted_Q_value)
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            # Find which action was taken
            action_i = torch.argmax(action[i]).item()
            target[i][action_i] = Q_new

        # 2: calc loss and update model weights
        self.optimizer.zero_grad() # Clear gradients from previous step
        loss = self.criterion(target, pred) # Calculate loss between target and predicted Q values
        loss.backward() # Backpropagate the loss
        self.optimizer.step() # Update model weights based on gradients
