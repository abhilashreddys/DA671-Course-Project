from __future__ import division
import numpy as np
from collections import defaultdict


class StochasticApproximationAgent(object):
    """
    This agent implements the standard Q-learning algorithm for the infinite-horizon
    average-reward setting with epsilon-greedy exploration.
    """
    def __init__(self, env, epsilon):
        self.REF_STATE = 0
        self.alpha = 1.
        self.env = env
        self.epsilon = epsilon

        self.mu = np.zeros([self.env.nb_states, self.env.nb_actions])  # for consistency called self.mu (it is q).
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.n_prime = np.zeros([self.env.nb_states, self.env.nb_actions, self.env.nb_states], dtype=int)

        self.state = None
        self.action = None
        self.t = 1

        self.reset()

    def act(self, state):
        self.state = state

        if np.random.rand() <= self.epsilon:
            self.action = np.random.choice(self.env.actions)
        else:
            self.action = np.argmax(self.mu[self.state])
        return self.action

    def update(self, next_state, reward):
        self.n[self.state, self.action] += 1
        self.alpha = 1.0/self.n[self.state, self.action]
        self.n_prime[self.state, self.action, next_state] += 1
        self.mu[self.state, self.action] = (1 - self.alpha) * self.mu[self.state, self.action] \
                                          + self.alpha * (reward + np.max(self.mu[next_state]) - np.max(self.mu[self.REF_STATE]))
        self.t += 1

    def info(self):
        return 'name = StochasticApproximationAgent\n' + 'epsilon = {0}\n'.format(self.epsilon)

    def reset(self):
        self.state = None
        self.action = None

        self.mu = np.zeros([self.env.nb_states, self.env.nb_actions])
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.n_prime = np.zeros([self.env.nb_states, self.env.nb_actions, self.env.nb_states], dtype=int)

        self.alpha = 1.
        self.t = 1


class OptimisticDiscountedAgent(object):
    """
    Optimistic Q-learning algorithm described in paper
    """

    def __init__(self, env, gamma=0.99, c=1.0):
        # _________ constants __________
        self.gamma = gamma
        self.H = gamma/(1.0-gamma)
        self.c = c
        # ______________________________

        self.env = env
        self.state = None
        self.action = None

        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.mu = self.H * np.ones([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.mu_hat = self.H * np.ones([self.env.nb_states, self.env.nb_actions])  # Q_hat in the algorithm
        self.v_hat = self.H * np.ones(self.env.nb_states)

    def act(self, state):
        self.state = state
        self.action = np.argmax(self.mu_hat[self.state, :])
        return self.action

    def update(self, next_state, reward):
        self.n[self.state, self.action] += 1
        self.t += 1
        bonus = self.c * np.sqrt(self.H/self.n[self.state, self.action])
        alpha = (self.H + 1)/(self.H + self.n[self.state, self.action])
        self.mu[self.state, self.action] = (1-alpha)*self.mu[self.state, self.action] + alpha*(reward + self.gamma*self.v_hat[next_state] + bonus)
        self.mu_hat[self.state, self.action] = min(self.mu_hat[self.state, self.action], self.mu[self.state, self.action])
        self.v_hat[self.state] = np.max(self.mu_hat[self.state, :])

    def info(self):
        return 'name = OptimisticDiscountedAgent\n' + 'gamma = {0}\n'.format(self.gamma) + 'c = {0}\n'.format(self.c)

    def reset(self):
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.mu = self.H * np.ones([self.env.nb_states, self.env.nb_actions])
        self.mu_hat = self.H * np.ones([self.env.nb_states, self.env.nb_actions])
        self.v_hat = self.H * np.ones(self.env.nb_states)
