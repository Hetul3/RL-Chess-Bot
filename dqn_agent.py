import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, state, action, next_state, reward):
        self.memory.append(Experience(state, action, next_state, reward))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.model = model
        self.target_model = type(model)()
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(100000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    # exploration implementation, model will start off exploring random moves to learn, as it fine tunes, it will start to exploit the best moves it knows
    def select_action(self, state, legal_moves):
        if random.random() > self.epsilon:
            return self.model.get_move(state, legal_moves)
        else:
            return random.choice(legal_moves)
            
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor([(a.from_square * 64 + a.to_square) for a in batch.action])
        reward_batch = torch.tensor(batch.reward)
        
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = torch.zeros(batch_size)
        next_q_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        