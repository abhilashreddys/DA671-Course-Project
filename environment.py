import numpy as np
# from utils import policy_iteration, q_value_iteration
from copy import deepcopy

def policy_iteration(c, p, v_star=False):
    """
    find optimal policy for a MDP with costs c and transition kernel p using policy iteration method
    :param c: list or numpy array of shape (nb_states, nb_actions) representing costs
    :param p: list or numpy array of shape (nb_actions, nb_states, nb_states) representing transition
    kernel
    :return: (optimal_cost, optimal_policy) where optimal_cost is a scalar and optimal_policy is a
    numpy array of integers with length nb_states that gives the optimal action at each state.
    """
    eps = 0.000001
    c = np.array(c)
    p = np.array(p)
    n, m = c.shape
    g = np.ones(n, dtype='int')
    b = np.zeros(n)
    a = np.zeros([n, n])
    a[:, 0] = 1
    x = np.zeros(n)
    while True:
        for i in range(n):
            b[i] = c[i, g[i]]
            for j in range(1, n):
                if i == j:
                    a[i, j] = 1 - p[g[i], i, j]
                else:
                    a[i, j] = - p[g[i], i, j]
        z = np.linalg.solve(a, b)
        w = np.block([np.array([0]), z[1:]])
        for i in range(n):
            tmp = c[i, :] + np.dot(p[:, i, :], w)
            g[i] = np.argmin(tmp)
            x[i] = tmp[g[i]]
        m = z[0] + w
        if np.linalg.norm(m - x) < eps:
            break
    opt_cost = z[0]
    if v_star:
        return opt_cost, g, w
    return opt_cost, g


def q_value_iteration(c, p):
    """
    find optimal Q function for a MDP with costs c and transition kernel p using value iteration method
    :param c: list or numpy array of shape (nb_states, nb_actions) representing costs
    :param p: list or numpy array of shape (nb_actions, nb_states, nb_states) representing transition
    kernel
    :return: approximately optimal_Q which is a numpy array of size (nb_states, nb_actions).
    """
    eps = 0.000001
    r = -np.array(c)
    p = np.array(p)
    n, m = c.shape
    q = np.zeros([n, m])
    q_new = np.zeros([n, m])
    s_ref = 0
    while True:
        q_new = r + np.dot(p, np.max(q, axis=1)).T - np.max(q[s_ref])
        if np.max(np.abs(q_new - q)) <= eps:
            break
        q = deepcopy(q_new)
    return q_new

class FiniteMDP(object):
    """
    A superclass for JumpRiverSwimEnv and RandomMDPEnv
    """
    def __init__(self, nb_states, nb_actions, costs, p):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.states = range(self.nb_states)
        self.actions = range(self.nb_actions)
        self.costs = costs
        self.p = p
        self.rewards = -self.costs
        self.span = np.max(np.max(self.costs, axis=1) - np.min(self.costs, axis=1))

        self.state = None
        self.opt_cost = None
        self.opt_policy = None
        self.opt_q = None

    def optimal_cost(self):
        if self.opt_cost is not None:
            return self.opt_cost
        self.opt_cost, self.opt_policy = policy_iteration(self.costs, self.p)
        return self.opt_cost

    def optimal_reward(self):
        return -self.optimal_cost()

    def optimal_policy(self):
        if self.opt_policy is not None:
            return self.opt_policy
        self.opt_cost, self.opt_policy = policy_iteration(self.costs, self.p)
        return self.opt_policy

    def optimal_q(self):
        if self.opt_q is not None:
            return self.opt_q
        self.opt_q = q_value_iteration(self.costs, self.p)
        return self.opt_q

    def info(self):
        s = ''
        s += 'name = {0}\n'.format(self.__class__.__name__)
        s += 'number of states = {0}, number of actions = {1}\n'.format(self.nb_states, self.nb_actions)
        s += 'optimal cost = {0}\n'.format(self.optimal_cost())
        s += 'optimal policy = {0}\n'.format(self.optimal_policy())
        s += 'optimal q =\n{0}\n'.format(self.optimal_q())
        s += 'cost function =\n{0}\n'.format(self.costs)
        s += 'transition kernel=\n{0}\n'.format(self.p)
        return s

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        reward = self.reward(self.state, action)
        self.state = np.random.choice(self.states, p=self.p[action, self.state])
        return self.state, reward

    def reward(self, state, action):
        return -self.costs[state, action]

    def write_info(self, directory=''):
        import os
        np.savetxt(os.path.join(directory, 'p'), self.p.reshape(self.nb_actions*self.nb_states, self.nb_states))
        np.savetxt(os.path.join(directory, 'r'), -self.costs)



class JumpRiverSwimEnv(FiniteMDP):
    def __init__(self):
        e = 1e-2 / 6  # 6*e would be probability of jumping to an arbitrary state.
        costs = np.array([[.8, 1],  # c(state=0, a)
                          [1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 0]])

        p = np.array([[[1-5*e, e, e, e, e, e],
                       [1-5*e, e, e, e, e, e],
                       [e, 1-5*e, e, e, e, e],
                       [e, e, 1-5*e, e, e, e],
                       [e, e, e, 1-5*e, e, e],
                       [e, e, e, e, 1-5*e, e]],  # p(s'|s, 0)

                      [[.7+e, .3-5*e, e, e, e, e],
                       [.1+e, .6+e, .3-5*e, e, e, e],
                       [e, .1+e, .6+e, .3-5*e, e, e],
                       [e, e, .1+e, .6+e, .3-5*e, e],
                       [e, e, e, .1+e, .6+e, .3-5*e],
                       [e, e, e, e, .7+e, .3-5*e]]])  # p(s'|s, 1)

        super(JumpRiverSwimEnv, self).__init__(nb_states=costs.shape[0], nb_actions=costs.shape[1], costs=costs, p=p)