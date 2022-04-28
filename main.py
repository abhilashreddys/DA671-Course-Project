from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from copy import deepcopy
from environment import JumpRiverSwimEnv
from agent import StochasticApproximationAgent, OptimisticDiscountedAgent
from plot import regret_plot
import os
import time

seed = 3
np.random.seed(seed)


class Runner(object):

    def __init__(self, agent, env, nb_runs=10, horizon=10000, filename=None, display_mode=1):
        self.agent = agent
        self.env = env
        self.nb_runs = nb_runs
        self.horizon = horizon
        self.display_mode = display_mode

        if filename:
            self.save_directory = os.path.join('log', filename)
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
        else:
            self.save_directory = None

    def run(self):
        if self.save_directory:
            with open(os.path.join(self.save_directory, 'info.txt'), 'w') as file:
                file.write('*'*10 + ' General Info ' + '*'*10 + '\n')
                file.write('seed = {0}\n'.format(seed))
                file.write('horizon = {0}\n'.format(self.horizon))
                file.write('number of runs = {0}\n'.format(self.nb_runs))
                file.write('*'*10 + ' Agent Info ' + '*'*10 + '\n')
                file.write(self.agent.info())
                file.write('*'*10 + ' Environment Info ' + '*'*10 + '\n')
                file.write(self.env.info())
        for exp in range(self.nb_runs):
            print('======= Experiment {0} ======='.format(exp))
            state = self.env.reset()
            self.agent.reset()
            regrets = [] # regret buffer
            saving_regrets = []
            saving_times = []
            for t in range(self.horizon):
                action = self.agent.act(state)
                next_state, reward = self.env.step(action)

                if agent.t % 1000 == 1: # samples the data at this time
                    saving_times.append(agent.t)
                    saving_regrets.append(np.sum(regrets))
                    regrets = []
                if self.display_mode == 2 and (agent.t-1) % 1000000 == 0:
                    print('experiment = {0}'.format(exp))
                    print('time = {0}'.format(agent.t))
                    print('number of visits to state action pairs =')
                    print(agent.n)
                    print('________________________')
                self.agent.update(next_state, reward)
                state = deepcopy(next_state)
                regrets.append(self.env.optimal_reward() - reward)
            if self.display_mode == 1:
                print('experiment = {0}'.format(exp))
                print('time = {0}'.format(agent.t))
                print('number of visits to state action pairs =')
                print(agent.n)
                print('________________________')

            if self.save_directory:
                np.savetxt(os.path.join(self.save_directory, str(exp)), np.array([saving_times, np.cumsum(saving_regrets)]).T)
        print('Experiments stored in {0}.'.format(self.save_directory))



if __name__ == '__main__':
    T = 50000

    env = JumpRiverSwimEnv()

    # uncomment the desired agent to run
    # agent = StochasticApproximationAgent(env=env, epsilon=0.03)
    agent = OptimisticDiscountedAgent(env=env, gamma=.99, c=1.0)

    storage_counter = 2 # change this to store data in a new folder
    filename = os.path.join(env.__class__.__name__, agent.__class__.__name__ + '_{0}'.format(storage_counter)) # DO NOT change this. Instead change storage_counter.
    start = time.time()
    runner = Runner(agent=agent, env=env, nb_runs=5, horizon=T, filename=filename, display_mode=1)
    runner.run()
    print('*'*50)
    print("Run time: ", time.time() - start)

    # plots
    environments_name = ['JumpRiverSwimEnv']

    agents = ['OptimisticDiscountedAgent']
    alg_storage = {'OptimisticDiscountedAgent': str(storage_counter)}
    legends = {'OptimisticDiscountedAgent': 'Optimistic Q-learning'}

    # agents = ['OptimisticDiscountedAgent', 'StochasticApproximationAgent']
    # alg_storage = {'StochasticApproximationAgent': str(storage_counter)}
    # legends = {'StochasticApproximationAgent': 'Q-learning with $\epsilon$-greedy'}
               
    save_directory = 'plots'

    for env_name in environments_name:
        regret_plot(environment_name=env_name, agents=agents, alg_storage=alg_storage, legends=legends, save_directory=save_directory)