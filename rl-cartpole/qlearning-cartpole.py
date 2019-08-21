import numpy as np
from cartpolesolver import CartPoleSolver
from collections import deque


#https://dev.to/n1try/cartpole-with-q-learning---first-experiences-with-openai-gym
#https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578


class QCartPoleSolver(CartPoleSolver):
    def __init__( self, state_bins=(1,1,6,12 ), n_episodes = 1000, min_epsilon = 0.1, min_alpha = 0.1, discount = 1.0, epsilon_decay=0.04, alpha_decay=0.04, n_win_ticks =195 ) :
        CartPoleSolver.__init__(self, n_episodes, min_epsilon, min_alpha, discount, epsilon_decay, alpha_decay, n_win_ticks)
        self.state_bins = state_bins
        self.Qtable = np.zeros(state_bins + ( self.env.action_space.n,) )
        self.state_bin_size = [(self.high[i]-self.low[i])/self.state_bins[i] for i in range(len(state_bins))]
        print('state_bin_size:{}'.format(self.state_bin_size) )
    
            
    def discretizeState( self, state ):
        discreet_state = [ max( self.low[i], min( self.high[i], state[i]) ) for i in range(len(state))]
        discreet_state = [int(np.floor((discreet_state[i]-self.low[i])/self.state_bin_size[i])) for i in range(len(state))]
        discreet_state = [min( self.state_bins[i]-1, discreet_state[i]) for i in range(len(state))]
        return tuple(discreet_state)
        
    def updateQtable( self, state, action, next_state, reward, alpha):
        self.Qtable[state][action] = ( 1-alpha ) *self.Qtable[state][action] + alpha*(reward + self.discount * max( self.Qtable[next_state]))

    
    def get_next_action( self, curr_state, epsilon ):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Qtable[curr_state])
       
    def run( self):
        solved = False
        for e in range( self.n_episodes ):
            curr_state = self.env.reset()   # reset the environment before every episode
            curr_dstate = self.discretizeState(curr_state)
            done = False # whether  the episode has ended or not
            i = 0
            score = deque(maxlen=100)
            while not done:
                #self.env.render()
                epsilon = self.get_epsilon(e)
                alpha = self.get_alpha(e)
                action = self.get_next_action(curr_dstate, epsilon)
                new_state, reward, done, _ = self.env.step(action)
                new_dstate = self.discretizeState(new_state)
                self.updateQtable(curr_dstate, action,new_dstate, reward,alpha )
                curr_dstate = new_dstate
                i += 1
            score.append(i)
            mean_score = np.mean(score)
            #print('After episode {} mean score is : {}'.format(e, mean_score))
            if mean_score >= self.n_win_ticks and e >= 100:
                print (' Mean reward in last 100 episodes is {} after {} episodes'.format(mean_score,e) )
                if mean_score >= self.n_win_ticks:
                    solved = True
                    print('Game solved after {} episodes'.format(e+1-100))
                    break
        if not solved:
            print('Could not solve after {} episodes'.format(e+1))
        
if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run()
    del solver

