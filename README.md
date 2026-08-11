import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs Value Iteration and returns:
    Value Function,
    Optimal Policy,
    Number of Iterations
    """

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Value Function
    V = np.zeros(n_states)
    iteration = 0
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )
            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))
        iteration += 1

        if delta < theta:
            break
    # Extract Optimal Policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )
        policy[s] = np.argmax(action_values)
    return V, policy, iteration

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------
V, policy, iterations = value_iteration(env)


# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name:  SANIYA G  ")
print("Register Number:  212223240147 ")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()

