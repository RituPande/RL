import numpy as np
from cartpolesolver import CartPoleSolver
from collections import deque
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import  LeakyReLU
import keras.backend as K
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

#https://gist.github.com/n1try/2a6722407117e4d668921fce53845432


class PgCartPoleSolver(CartPoleSolver):
    def __init__( self, n_episodes = 10000, discount = 0.99 , n_win_ticks =195, batch_size=25000, tau=20 ) :
        CartPoleSolver.__init__(self, n_episodes,  discount,  n_win_ticks)
        self.tau = tau # number of episodes used to collect trajectories to estimate advatage estimate
        self.batch_size = batch_size
        self.STATE_DIM = 4
        self.ACTION_DIM = 2
        self.ActorNetwork_train, self.ActorNetwork_predict = self.create_policy_model();
        self.CriticNetwork = self.create_critic_model();

        np.random.seed(1)
        set_random_seed(9)
            
    def pgloss( self, adv):
        def loss(y_true,y_pred):
            val =  K.categorical_crossentropy(y_true, y_pred) * adv 
            return K.mean(val)
        return loss
           
    def create_policy_model(self):
        
        inp = Input(shape=[self.STATE_DIM], name = "input_x")
        adv = Input(shape=[1], name = "advantage")
        x = Dense(32, activation='tanh')(inp)
        x = Dense(24, activation='relu')(x)
        out= Dense(self.ACTION_DIM, activation='softmax')(x)
        
        model_train = Model(inputs=[inp, adv], outputs=out)
        model_train.compile(loss=self.pgloss(adv), optimizer=Adam(lr=1e-2))
        model_predict = Model(inputs=[inp], outputs=out)
        return model_train, model_predict
        
    def create_critic_model(self ):
        model = Sequential()
        model.add(Dense(5, activation='relu', input_dim=self.STATE_DIM))
        model.add(Dense(4, activation= 'relu' ))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=1e-2))
        return model
        
    def get_next_action( self, curr_state ):
        p_actions = self.ActorNetwork_predict.predict(curr_state)
        p_actions = np.squeeze(p_actions, axis=0)
        return np.random.choice( range(self.ACTION_DIM), p=p_actions )
            
    def preprocess_input( self, state ):
        state = np.reshape(state, [1,self.STATE_DIM])
        return state

    def preprocess_action( self, action ):
        a = np.zeros((1,self.ACTION_DIM))
        a[0, action] = 1
        return a

         
    def get_discounted_rewards(self, rewards):
        discounted_rewards = deque(maxlen=500)
        discounted_rewards.clear()
        rewards.reverse()
        for r in rewards:
            if not discounted_rewards:
                v = r
            else:
                v=  r + self.discount*discounted_rewards[0]
            discounted_rewards.appendleft(v)
        
        return discounted_rewards
        
    def experience_trajectory(self, e ):
        curr_state = self.env.reset()   # reset the environment before every episode
        curr_state = self.preprocess_input( curr_state )
        score = 0
        done = False # whether  the episode has ended or not
        states = deque(maxlen=500) 
        rewards = deque(maxlen=500)
        actions = deque(maxlen=500)
        states.clear()
        rewards.clear()
        actions.clear()
        while not done:
            #self.env.render()
            action = self.get_next_action(curr_state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess_input( next_state )
            states.append(curr_state)
            rewards.append(reward) 
            action = self.preprocess_action( action )
            actions.append(action)
            curr_state = next_state
            score += 1
        discounted_rewards = self.get_discounted_rewards(rewards)
        return score, discounted_rewards, states, actions
 
    def get_advatange_estimate( self,states_batch, discounted_rewards_batch ):
        x = np.array(states_batch)
        x = np.squeeze(x, axis=1)
        target =np.array(discounted_rewards_batch)
        target = np.reshape(target, (-1,1))
        pred_baseline = self.CriticNetwork.predict(x) 
        adv = target - pred_baseline
        adv = np.reshape(adv, (-1,1) )
        return adv
        
    def reset_batch_variables( self,states_batch,actions_batch, discounted_rewards_batch ):
        states_batch.clear()
        actions_batch.clear()
        discounted_rewards_batch.clear()
       
        
    def prepare_batch(self, batch ):
        x = np.array(batch)
        x = np.squeeze(x, axis=1)
        return x
    
    def normalize_batch( self, batch, data ):
        mu = np.mean(data)
        std = np.std(data)
        batch -= mu
        batch /= std
        
        
    def score_model( self, n_test_episodes):
         test_scores = deque(maxlen=n_test_episodes)
         test_scores.clear()
        
         for e in range( n_test_episodes):
            curr_state = self.env.reset()   # reset the environment before every episode
            curr_state = self.preprocess_input( curr_state )
            score = 0
            done = False # whether  the episode has ended or not
         
            while not done:
                #self.env.render()
                p_a = self.ActorNetwork_predict.predict(curr_state)
                action = np.argmax(p_a)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_input( next_state )
                curr_state = next_state
                score += 1
            test_scores.append(score)
            
         return np.mean(test_scores)
        
    def run( self):
        scores = deque(maxlen=100)
        states_batch  = deque(maxlen=self.batch_size)
        actions_batch  = deque(maxlen=self.batch_size)
        discounted_rewards_batch  = deque(maxlen=self.batch_size)
        scores.clear()
        self.reset_batch_variables(states_batch,actions_batch, discounted_rewards_batch )
        adv_estimate_data = np.empty(0).reshape(0,1)
        scores_for_plot = deque(maxlen=self.n_episodes)
        solved = False
        for e in range( self.n_episodes ):
            tau_s, tau_discounted_rewards, tau_states, tau_actions = self.experience_trajectory( e )
            scores.append(tau_s)
            scores_for_plot.append(tau_s)
            states_batch += tau_states
            actions_batch += tau_actions
            discounted_rewards_batch += tau_discounted_rewards
            
            if e > 0 and e % self.tau == 0:
                adv_estimate = self.get_advatange_estimate(states_batch, discounted_rewards_batch)
                adv_estimate_data = np.vstack([adv_estimate_data, adv_estimate] )
                self.normalize_batch (adv_estimate,adv_estimate_data )
                
                x = self.prepare_batch(states_batch)
                
                self.CriticNetwork.fit(x,np.array(discounted_rewards_batch) , batch_size = len(states_batch), verbose =0 )
                
                y = self.prepare_batch(actions_batch)
                
                hist = self.ActorNetwork_train.fit([x, adv_estimate], y, batch_size=len(x), verbose=0 )
                #train_loss.append(hist.history['loss'])
                
                self.reset_batch_variables(states_batch,actions_batch, discounted_rewards_batch )


            if  e> 0 and (e+1) % 100  == 0  :
                mean_score = np.mean(scores)
                print('*************************************************************************')
                print (' Mean reward in last 100 episodes during training is {} after {} episodes'.format(mean_score,e+1) )
                print (' Max reward in last 100 episodes during trainign is {}  after {} episodes'.format(np.max(scores),e+1))
                test_score = self.score_model(10)
                print (' Mean test score for 10 episodes is {} after {} episodes'.format(test_score,e+1) )
                if test_score >= self.n_win_ticks:
                    print('Game solved after {} episodes'.format(e+1-100))
                    solved = True
                    break
        
        if not solved:
            print('Could not solve after {} episodes'.format(e+1))
        plt.plot(scores_for_plot)
        return solved, test_score
            
       
if __name__ == "__main__":
    result_solved= []
    result_scores=[]
    for i in range(10):
        solver = PgCartPoleSolver()
        solved, test_score = solver.run()
        result_solved.append(solved)
        result_scores.append(test_score)
    print('Number of Solutions:{}'.format(sum(result_solved)))
    print(result_scores)
    
    del solver
        

