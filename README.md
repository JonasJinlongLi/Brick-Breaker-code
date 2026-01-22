**This is an overview of the Brick Breaker files. It explains the purpose of each file and what functionality they provide.**


**Brick_breaker_game.py:**
Contains the core implementation of the Brick Breaker game using Pygame. It defines the game environment, including the game window, paddle, ball, and bricks, as well as all movement and collision mechanics. This file is required for the other files to work.


**Play_Brick_breaker.py:**
This file enables manual gameplay of the Brick Breaker game. It handles keyboard input for paddle control.


**Brick_breaker_AI.py:**
This file is a reinforcement learning agent that learns to play the Brick Breaker game using Q-learning.


**data.py:**
This file tracks the number of episodes needed for the agent to win the Brick Breaker game for the first time. The file also provides functionality to run multiple trials, collecting statistics such as the mean and standard deviation of episodes required to achieve a win.


**plot.py:**
This file compares the performance of two Q-learning agents in the Brick Breaker game: one using an optimistic initial Q-value and one using a deterministic start. This file trains each agent for a number of epochs, tracks the total reward per epoch, and generates a moving average plot of rewards to visualize learning progress. This file takes a relatively long time to generate the graph.
