import numpy as np
from .env_dynamic_programming import *

def policy_evaluation(env):

    S = env.states()
    A = env.actions()
    R = env.rewards()

    nb_cells = len(S)

    right_pi = np.zeros((len(S), len(A)))
    right_pi[:, 1] = 1.0

    left_pi = np.zeros((len(S), len(A)))
    left_pi[:, 0] = 1.0

    pi = np.ones((len(S), len(A))) * 0.5
    theta = 0.0000001
    V = []
    for i in range(len(S)):
        if not env.is_state_terminal(i):
            V.append([i, np.random.rand()])
        else:
            V.append([i, 0.0])
    while True:
        delta = 0
        for s in S:
            v = V[s][1]
            V[s][1] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        p = env.transition_probability(s, a, s_p, r)
                        total += p * (R[r] + 0.999 * V[s_p][1])
                total *= pi[s, a]
                V[s][1] += total
            delta = max(delta, np.abs(v - V[s][1]))
        if delta < theta:
            break
    return V


def policy_iteration(env, gamma, theta):

    S = env.states()
    A = env.actions()
    R = env.rewards()

    nb_cells = len(S)

    V = [[i, np.random.rand()] for i in range(len(S))]
    V[0][1] = 0.0
    V[nb_cells - 1][1] = 0.0

    pi = np.random.random((len(S), (len(A))))
    for s in S:
        pi[s] /= np.sum(pi[s])

    pi[0] = 0.0
    pi[nb_cells - 1] = 0.0

    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in S:
                v = V[s][1]
                V[s][1] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):
                            p = env.transition_probability(s, a, s_p, r)
                            total += p * (R[r] + gamma * V[s_p][1])
                    total *= pi[s, a]
                    V[s][1] += total
                delta = max(delta, np.abs(v - V[s][1]))
            if delta < theta:
                break

        # policy improvement
        stable = True
        for s in S:
            old_pi_s = pi[s].copy()
            best_a = -1
            best_a_score = -99999999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        p = env.transition_probability(s, a, s_p, r)
                        total += p * (R[r] + gamma * V[s_p][1])

                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if np.any(pi[s] != old_pi_s):
                stable = False
        if stable:
            print()
            return pi, V

def value_iteration(env, theta, gamma):

    S = env.states()
    A = env.actions()
    R = env.rewards()

    pi = np.ones((len(S), len(A)))
    pi /= len(A)

    V = np.random.random((len(S),))
    while True:
        delta = 0
        for s in S:
            old_v = V[s]
            V[s] = 0.0
            for a in A:
                for s_next in S:
                    for r_idx, r in enumerate(R):
                        p = env.transition_probability(s, a, s_next, r_idx)
                        V[s] += pi[s, a] * p * (r + gamma * V[s_next])
            delta = max(delta, abs(V[s] - old_v))

        if delta < theta:
            break

    Pi = np.zeros((len(S), len(A)))
    for s in S:
        best_action = 0
        best_action_score = -9999999999999
        for a in A:
            tmp_sum = 0
            for s_p in S:
                p1 = env.transition_probability(s, a, s_p, 0)
                p2 = env.transition_probability(s, a, s_p, 1)
                tmp_sum += p1 * (p2 + gamma * V[s_p])
            if tmp_sum > best_action_score:
                best_action = a
                best_action_score = tmp_sum
        Pi[s] = 0.0
        Pi[s, best_action] = 1.0

    return Pi, V
