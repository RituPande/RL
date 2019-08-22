## Reinforcement Learning Algorithms to Solve the Cartpole Problem

This repository  implements the classic reinforcement learning  and deep reinforcement learning algorithms to solve cartpole problem.
It uses the OpenAIGym environment.

**cartpolesolver.py** is the base class that implements standard environemnt initializations and utility functions that are required by all reinforcement learning algorithm

**qlearning-cartpole.py** implements the qlearning algorithm by diiscretizing the countinuous state-space for the cartpole problem

**dqlearning-cartpole.py** implements the deep queue learning algorithm to solve the cartpole problem.It also uses a separate target network to predict existing q-values which is updated periodically with weights from the q-values learning network.

**pglearning-cartpole-reinforce.py**, implements the policy gradient REINFORCE algorithm

**pglearning-cartpole.py** implements the policy gradient REINFORCE alogorithm with baseline


