import numpy as np
import random
from collections import defaultdict
from Brick_breaker_game import Game, WIDTH, HEIGHT
import matplotlib.pyplot as plt

epoch_limit = 5000
class QLearningAgent:
    def __init__(self, game, optimistic_start=False, initial_value=20.0): #optimistic q-value
        self.game = game
        self.ACTIONS = [0, 1, 2]
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        self.PADDLE_BINS = 10
        self.BALL_X_BINS = 15
        self.BALL_Y_BINS = 15
        
        self.optimistic_start = optimistic_start
        self.initial_value = initial_value
        
        def default_value():
            if optimistic_start:
                return [initial_value, initial_value, initial_value]
            else:
                return [0., 0., 0.]
        self.Q = defaultdict(default_value)
    
    def discretize_value(self, value, max_value, bins):
        normalized = value / max_value
        bin_index = int(normalized * bins)
        return min(bins - 1, max(0, bin_index))
    
    def get_state(self):
        paddle_pos = self.game.paddle.x
        paddle_max = WIDTH - self.game.paddle.width
        paddle_bin = self.discretize_value(paddle_pos, paddle_max, self.PADDLE_BINS)
        
        ball_x_bin = self.discretize_value(self.game.ball.centerx, WIDTH, self.BALL_X_BINS)
        ball_y_bin = self.discretize_value(self.game.ball.centery, HEIGHT, self.BALL_Y_BINS)
        
        return (paddle_bin, ball_x_bin, ball_y_bin)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        else:
            return np.argmax(self.Q[state])
    
    def update_q(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_epoch(self):
        self.game.reset()
        state = self.get_state()
        total_reward = 0
        
        while not self.game.done:
            action = self.choose_action(state)
            reward, done = self.game.step(action)
            next_state = self.get_state()
            self.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        self.decay_epsilon()
        return total_reward
    
    def train_for_epochs(self, num_epochs=epoch_limit):
        rewards = []
        for epoch in range(num_epochs):
            rewards.append(self.train_epoch())
        return rewards

def run_comparison(num_epochs=epoch_limit):
    
    game_opt = Game()
    agent_opt = QLearningAgent(game_opt, optimistic_start=True, initial_value=10.0)
    optimistic_rewards = agent_opt.train_for_epochs(num_epochs)
    
    
    game_det = Game()
    agent_det = QLearningAgent(game_det, optimistic_start=False)
    deterministic_rewards = agent_det.train_for_epochs(num_epochs)
    
    
    plt.figure(figsize=(10, 6))
    
    
    window_size = 50
    optimistic_ma = np.convolve(optimistic_rewards, np.ones(window_size)/window_size, mode='valid')
    deterministic_ma = np.convolve(deterministic_rewards, np.ones(window_size)/window_size, mode='valid')
    
    epochs = np.arange(1, num_epochs + 1)
    
    plt.plot(epochs[window_size-1:], optimistic_ma, 'b-', label='Optimistic Start', linewidth=2)
    plt.plot(epochs[window_size-1:], deterministic_ma, 'r-', label='Deterministic Start', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Total Reward (Moving Average)')
    plt.title(f'Reward per Epoch: Optimistic vs Deterministic Start\n(Moving Average, window={window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison(num_epochs=epoch_limit)