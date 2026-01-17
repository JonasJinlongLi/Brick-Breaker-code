import numpy as np
import random
from collections import defaultdict
from Brick_breaker_game import Game, WIDTH, HEIGHT
import statistics

class SimpleEpisodeCounter:
    def __init__(self, game):
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
        
        def default_value():
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
    
    def count_to_first_win(self, max_episodes=10000):
        for episode in range(1, max_episodes + 1):
            self.game.reset()
            state = self.get_state()
            
            
            while not self.game.done:
                action = self.choose_action(state)
                reward, done = self.game.step(action)
                next_state = self.get_state()
                self.update_q(state, action, reward, next_state)
                state = next_state
            
            
            if len(self.game.bricks) == 0:
                return episode
            
            
            self.decay_epsilon()
        
        return None

    def reset_q_learning(self):
        self.Q.clear()
        self.epsilon = 1.0

def run_multiple_trials(num_runs, max_episodes_per_run):
    
    episodes_results = []
    successful_runs = 0
    
    for run in range(num_runs):
        game = Game()
        counter = SimpleEpisodeCounter(game)
        
        episodes_to_win = counter.count_to_first_win(max_episodes=max_episodes_per_run)
        
        if episodes_to_win:
            episodes_results.append(episodes_to_win)
            successful_runs += 1
            print(f"Run {run+1}: {episodes_to_win} episodes")
        else:
            print(f"Run {run+1}: No win")
    
    
    if successful_runs > 0:
        mean_episodes = statistics.mean(episodes_results)
        std_episodes = statistics.stdev(episodes_results) if len(episodes_results) > 1 else 0
        print(f"Mean: {mean_episodes:.2f}")
        print(f"Standard deviation: {std_episodes:.2f}")
    else:
        print("No successful runs!")


if __name__ == "__main__":
    # Run n trials and calculate statistics
    run_multiple_trials(num_runs=100, max_episodes_per_run=50000)
