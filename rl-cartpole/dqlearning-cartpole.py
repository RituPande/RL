import numpy as np
import random
from cartpolesolver import CartPoleSolver
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import  LeakyReLU

from keras.optimizers import Adam


#https://gist.github.com/n1try/2a6722407117e4d668921fce53845432


class DQCartPoleSolver(CartPoleSolver):
    def __init__( self, n_episodes = 1000, min_epsilon = 0.1, min_alpha = 0.1, discount = 1.0, epsilon_decay=0.04, alpha_decay=0.04, n_win_ticks =195, batch_size=32, tau=50 ) :
        CartPoleSolver.__init__(self, n_episodes, min_epsilon, min_alpha, discount, epsilon_decay, alpha_decay, n_win_ticks)
        self.QModel = self.create_model();
        self.TargetQModel = self.create_model();
        self.tau = tau # number of episodes after which weights from QModel should be copied to target weights
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = batch_size
        self.STATE_DIM =4
    
           
    def create_model(self ):
        model = Sequential()
        model.add(Dense(32, input_dim=self.STATE_DIM))
        model.add(LeakyReLU())
        model.add(Dense(24))
        model.add(LeakyReLU())
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
        
    
    def get_next_action( self, curr_state,epsilon ):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.QModel.predict(curr_state))
            
    def preprocess_input( self, state ):
        state = np.reshape(state, [1, self.STATE_DIM])
        return state
    
    def add_to_replay_buffer( self, curr_state, action, reward, next_state, done ):
        self.replay_buffer.append((curr_state, action, reward, next_state, done))
        
    def sync_target_model( self ):
        self.TargetQModel.set_weights( self.QModel.get_weights()) 
        
    def replay(self, e):
        x_batch, y_batch = [], []
        mini_batch = random.sample( self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
        
        for i in range( len(mini_batch)):
            curr_state, action, reward, next_state, done = mini_batch[i]
            y_target = self.QModel.predict(curr_state) # get existing Qvalues for the current state
            y_target[0][action] = reward if done else reward + self.discount*np.max(self.TargetQModel.predict(next_state)) # modify the qvalues for the action perfomrmed to get the new target 
            x_batch.append(curr_state[0])
            y_batch.append(y_target[0])
            
        self.QModel.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        
    
            
    def experience(self, e ):
        curr_state = self.env.reset()   # reset the environment before every episode
        curr_state = self.preprocess_input( curr_state)
        i = 0
        done = False # whether  the episode has ended or not
        while not done:
            #self.env.render()
            epsilon = self.get_epsilon(e)
            action = self.get_next_action(curr_state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess_input( next_state )
            self.add_to_replay_buffer( curr_state, action, reward, next_state, done )
            curr_state = next_state
            i += 1
            
        return i
   
    def run( self):
        scores = deque(maxlen=100)
        solved = False
        for e in range( self.n_episodes ):
            s = self.experience( e )
            scores.append(s)
            
            if e > 0 and e % self.tau == 0:
                self.sync_target_model()
            
            self.replay(e)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                print (' Mean reward in last 100 episodes is {} after {} episodes'.format(mean_score,e+1) )
                if mean_score >= self.n_win_ticks:
                    print('Game solved after {} episodes'.format(e+1-100))
                    solved = True
                    break
        
        if not solved:
            print('Could not solve after {} episodes'.format(e+1))
            
       
if __name__ == "__main__":
    solver = DQCartPoleSolver()
    solver.run()
    del solver
        

