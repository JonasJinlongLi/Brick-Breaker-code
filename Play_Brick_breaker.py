import pygame
import sys
from Brick_breaker_game import Game, WIDTH, HEIGHT

def manual_game():
    game = Game()
    running = True
    episode = 0
    total_reward = 0 
    
    font = pygame.font.Font(None, 24)
    
    while running:
        game.clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    episode += 1
                    total_reward = 0  
                    print(f"Episode {episode} started")
        
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        
        
        step_reward, done = game.step(action)
        total_reward += step_reward 
        
        if done:
            print(f"Episode {episode} completed with total reward: {total_reward}")
            game.reset()
            episode += 1
            total_reward = 0 
        
        game.render()
        
    
        info_text = f"Episode: {episode} | Bricks: {len(game.bricks)} | Total Reward: {total_reward}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        game.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    manual_game()