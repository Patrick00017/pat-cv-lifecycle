import torch
import torch.nn as nn
import numpy as np
import random

def onehot_encode(state, arms):
    vec = np.zeros(arms)
    vec[state] = 1
    return vec

def sample_action(vec):
    return np.argmax(vec, axis=0)

class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()
    
    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms, arms) # (states, arms)

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = np.random.randint(0, self.arms)
    
    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward
    
class TinyNet(nn.Module):
    def __init__(self, arms):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(arms, 100),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x



if __name__ == '__main__':
    arms = 10
    env = ContextBandit(arms=arms)
    state = env.get_state()
    reward = env.get_reward(1)
    print(f"state: {state}, reward: {reward}")

    model = TinyNet(arms)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    rewards = []
    cur_state = torch.Tensor(onehot_encode(env.get_state(), arms))
    for epoch in range(50000):
        y_pred = model(cur_state)
        y_pred_logits = torch.softmax(y_pred, dim=0).detach().numpy()
        choice = np.random.choice(arms, p=y_pred_logits)
        cur_reward = env.choose_arm(choice)
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice] = cur_reward
        reward = torch.Tensor(one_hot_reward)
        rewards.append(reward)
        loss = criterion(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(onehot_encode(env.get_state(), arms))
    
    for reward in rewards:
        r = reward.detach().numpy()
        print(np.mean(r))