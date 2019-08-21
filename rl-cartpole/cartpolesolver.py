import gym
import numpy as np
import math


#https://dev.to/n1try/cartpole-with-q-learning---first-experiences-with-openai-gym
#https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578


class CartPoleSolver:
    def __init__( self, n_episodes = 1000, min_epsilon = 0.1, min_alpha = 0.1, discount = 1.0, epsilon_decay=0.04, alpha_decay = 0.04, n_win_ticks =195 ) :
        self.env = gym.make('CartPole-v0')
        self.env.reset()
        self.low = [self.env.observation_space.low[0],  -0.5, self.env.observation_space.low[2], -math.radians(50) ]
        self.high = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) ]
        self.min_epsilon = min_epsilon
        self.min_alpha = min_alpha
        self.discount = discount
        self.epsilon = 1
        self.alpha = 0.7
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks =195
        
    def __del__(self): 
        self.env.close()   
           
        
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) * self.alpha_decay)))
        
    def get_next_action( self, curr_state,epsilon ): pass
        
    def run( self): pass
        

