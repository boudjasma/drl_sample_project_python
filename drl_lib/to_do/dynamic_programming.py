from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from .env_dynamic_programming import *
from .functions_dynamic_programming import *

def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    nb_cells = 7
    env = Env_on_line(nb_cells)
    V = policy_evaluation(env)
    env.display_on_line_world(V, "Policy_evaluation_on_line_world")
    return V

def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    nb_cells = 7
    env = Env_on_line(nb_cells)

    gamma = 0.999
    theta = 0.0000001
    (pi, V) = policy_iteration(env, gamma, theta)
    env.display_on_line_world(V, "Policy_iteration_on_line_world")
    policy = [[V[i][0], [pi[i], V[i][1]]] for i in range(len(V))]

    return policy


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    nb_cells = 7
    env = Env_on_line(nb_cells)

    theta = 0.00001
    gamma = 0.99999

    (pi, V) = policy_iteration(env, gamma, theta)
    env.display_on_line_world(V, "Value_iteration_on_line_world")
    value_iteration = [[V[i][0], [pi[i], V[i][1]]] for i in range(len(V))]
    return value_iteration

def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    height = 5
    width = 5
    env = Env_on_grid(height, width)
    V = [policy_evaluation(env)[i][1] for i in range(len(policy_evaluation(env)))]
    env.display_on_grid_world(height, width, V, "policy_evaluation_on_grid_world")
    return V


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    height = 5
    width = 5
    env = Env_on_grid(height, width)

    gamma = 0.999
    theta = 0.0000001

    (pi, V) = policy_iteration(env, gamma, theta)
    env.display_on_grid_world(height, width, V, "policy_iteration_on_grid_world")

    return policy_iteration(env, gamma, theta)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    height = 5
    width = 5
    env = Env_on_grid(height, width)

    theta = 0.00001
    gamma = 0.99999

    (pi, V) = value_iteration(env, theta, gamma)
    env.display_on_grid_world(height, width, V, "value_iteration_on_grid_world")

    return value_iteration(env, theta, gamma)


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    print("Policy evaluation in secret environment")
    return policy_evaluation(env)


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    gamma = 0.999
    theta = 0.0000001
    print("Policy iteration in secret environment")
    return policy_iteration(env, gamma, theta)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    theta = 0.00001
    gamma = 0.99999
    print("Value iteration in secret environment")
    return value_iteration(env, theta, gamma)


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
