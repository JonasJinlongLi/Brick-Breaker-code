import numpy as np
import random
from collections import defaultdict
import pygame
import sys
from Brick_breaker_game import Game, WIDTH, HEIGHT

ACTIONS = [0, 1, 2]  # stay, left, right
PADDLE_BINS = 10
BALL_X_BINS = 15
BALL_Y_BINS = 15

# RL parameters
alpha = 0.2
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

def default_value():
    return [0., 0., 0.]

Q = defaultdict(default_value)

episode = 0
counter = 0

print(f"State space size: {PADDLE_BINS * BALL_X_BINS * BALL_Y_BINS} possible states")


def discretize_value(value, max_value, bins):
    normalized = value / max_value
    bin_index = int(normalized * bins)
    return min(bins - 1, max(0, bin_index))

def get_state(game):
    paddle_pos = game.paddle.x
    paddle_max = WIDTH - game.paddle.width
    paddle_bin = discretize_value(paddle_pos, paddle_max, PADDLE_BINS)
    
    ball_x_bin = discretize_value(game.ball.centerx, WIDTH, BALL_X_BINS)
    ball_y_bin = discretize_value(game.ball.centery, HEIGHT, BALL_Y_BINS)
    
    return (paddle_bin, ball_x_bin, ball_y_bin)

def choose_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        return np.argmax(Q[state])

def update_q(state, action, reward, next_state):
    best_next = np.max(Q[next_state])
    Q[state][action] += alpha * (
        reward + gamma * best_next - Q[state][action]
    )

def decay_epsilon():
    global epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

def print_state_info(game):
    state = get_state(game)
    print(f"Paddle x: {game.paddle.x}, Ball: ({game.ball.centerx}, {game.ball.centery})")
    print(f"State tuple: {state}")
    print(f"Q-values for state {state}: {Q[state]}")

def run_training(game):
    global episode, counter
    
    running = True
    clock = pygame.time.Clock()
    FPS = 120
    
    screen = game.screen
    
    WHITE = (255, 255, 255)
    BLUE = (50, 150, 255)
    RED = (200, 50, 50)
    BLACK = (0, 0, 0)
    
    game.reset()
    
    while running:
        clock.tick(FPS)
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    print_state_info(game)
                elif event.key == pygame.K_r:
                    game.reset()
                    episode += 1
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Get current state
        state = get_state(game)
        
        # Choose action
        action = choose_action(state)
        
        # Perform action and receive reward
        reward, done = game.step(action)
        
        # Get next state
        next_state = get_state(game)
        
        # Update Q-value
        update_q(state, action, reward, next_state)
        
        if done:
            decay_epsilon()
            counter += 1
            game.reset()
            episode += 1

        pygame.draw.rect(screen, BLUE, game.paddle)
        pygame.draw.ellipse(screen, WHITE, game.ball)
        for brick in game.bricks:
            pygame.draw.rect(screen, RED, brick)
        
        font = pygame.font.Font(None, 24)
        info_text = (
            f"Episode: {episode} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Bricks: {len(game.bricks)} | "
            f"Reward: {reward}"
        )
        text_surface = font.render(info_text, True, WHITE)
        screen.blit(text_surface, (10, 10))

        state_text = f"State: {get_state(game)}"
        state_surface = font.render(state_text, True, WHITE)
        screen.blit(state_surface, (10, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game = Game()
    run_training(game)
