import random
from itertools import product

import numpy as np


def calculate_payoff(weights, rewards, strategy):
    """
    Calculate the payoff for each agent given a strategy profile.

    Parameters:
    weights (list): List of weights for each agent.
    rewards (list): List of rewards for each arm.
    strategy (list): List of chosen arms for each agent.

    Returns:
    list: Payoff for each agent.
    """
    N, K = weights.shape

    # Initialize payoff list
    payoffs = [0] * N

    # Calculate the sum of weights for each arm
    sum_weights_per_arm = [0] * K
    for i in range(N):
        arm = strategy[i]
        sum_weights_per_arm[arm] += weights[i][arm]

    # Calculate payoff for each agent
    for i in range(N):
        arm = strategy[i]
        payoffs[i] = (weights[i][arm] / sum_weights_per_arm[arm]) * rewards[arm]

    return payoffs


def find_pne(weights, rewards, isprint=False):
    """
    Find the most efficient Pure Nash Equilibrium.

    Parameters:
    weights (list): List of weights for each agent.
    rewards (list): List of rewards for each arm.

    Returns:
    tuple: Most efficient PNE and its total payoff.
    """
    N, K = weights.shape

    # Generate all possible strategy profiles
    all_strategies = list(product(range(K), repeat=N))

    best_pne = None
    best_total_payoff = 0

    best_delta_1 = 1000

    no_pne = []

    multiple_pne = False
    first_payoff = -1

    # Iterate through all strategy profiles to find PNE
    for strategy in all_strategies:
        is_pne = True
        total_payoff = 0
        current_payoffs = calculate_payoff(weights, rewards, strategy)

        delta_2 = 0
        delta_1 = 1000

        # Check if any agent has an incentive to deviate
        for i in range(N):
            for new_arm in range(K):
                if new_arm == strategy[i]:
                    continue
                # Generate a new strategy profile where agent i deviates to new_arm
                new_strategy = list(strategy)
                new_strategy[i] = new_arm

                # Calculate new payoffs
                new_payoffs = calculate_payoff(weights, rewards, new_strategy)

                # If agent i can get a higher payoff by deviating, then it's not a PNE
                if new_payoffs[i] > current_payoffs[i]:
                    is_pne = False
                    delta_2 = max(delta_2, new_payoffs[i] - current_payoffs[i])

                if new_payoffs[i] <= current_payoffs[i] and is_pne:
                    delta_1 = min(delta_1, current_payoffs[i] - new_payoffs[i])

        # Update the best PNE if necessary
        if is_pne:
            total_payoff = sum(current_payoffs)
            if first_payoff == -1:
                first_payoff = total_payoff
            elif abs(total_payoff - first_payoff) > 1e-5:
                multiple_pne = True
            if isprint:
                print("PNE", strategy, total_payoff, delta_1)
            if delta_1 <= best_delta_1:
                # if total_payoff >= best_total_payoff and delta_1 <= best_delta_1:
                best_total_payoff = total_payoff
                best_pne = strategy
                best_delta_1 = delta_1
            if total_payoff >= best_total_payoff:
                best_total_payoff = total_payoff
        else:
            no_pne.append((sum(current_payoffs), delta_2))
            if isprint:
                print("No PNE", strategy, sum(current_payoffs), delta_2)

    return best_pne, best_total_payoff, best_delta_1, no_pne, multiple_pne


def calculate_delta(weights, rewards, isprint=False):
    best_pne, best_total_payoff, best_delta_pne, no_pne, multiple_pne = find_pne(
        weights, rewards, isprint
    )

    l = [ele[1] for ele in no_pne if ele[0] >= best_total_payoff - 1e-5]
    delta_nopne = min(l) if len(l) > 0 else 100

    if isprint:
        print(best_pne, best_total_payoff, best_delta_pne, delta_nopne)
    return best_delta_pne, delta_nopne, min(best_delta_pne, delta_nopne)


if __name__ == "__main__":
    # Generate random weights and rewards for 3 agents and 3 arms
    # random.seed(42)  # For reproducibility
    N = 3  # Number of agents
    K = 2  # Number of arms

    # PNE > NOPNE
    weights = np.tile([[1.0, 0.8, 0.4]], [K, 1]).reshape(N, K)
    rewards = np.array([1.0, 0.7])
    print(weights, rewards)
    print(calculate_delta(weights, rewards))

    # NOPNE > PNE
    weights = np.tile([[0.9, 0.6, 0.4]], [K, 1]).reshape(N, K)
    rewards = np.array([1, 0.6])
    print(calculate_delta(weights, rewards))

    # Multiple PNE: there exist more insufficient PNE
    # weights = [0.8, 0.3, 0.2]
    # rewards = [0.6, 0.4, 0.2]


    # while True:
    #     # weights = [round(random.uniform(0.1, 1), 1) for _ in range(N)]
    #     # rewards = [round(random.uniform(0.1, 1), 1) for _ in range(K)]

    #     print("Weights:", weights)
    #     print("Rewards:", rewards)

    #     # Find the most efficient PNE
    #     best_pne, best_total_payoff, best_delta_pne, no_pne, multiple_pne = find_pne(
    #         weights, rewards
    #     )

    #     l = [ele[1] for ele in no_pne if ele[0] >= best_total_payoff - 1e-5]
    #     delta_nopne = min(l) if len(l) > 0 else 100

    #     print(best_pne, best_total_payoff, best_delta_pne, delta_nopne)

    #     # if best_delta_pne > delta_nopne + 1e-3:
    #     #     break

    #     if multiple_pne:
    #         break
