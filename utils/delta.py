import random
from itertools import product

import numpy as np
from tqdm import tqdm


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


def find_pne_1(weights, rewards, isprint=False):
    N, K = weights.shape

    # Generate all possible strategy profiles
    all_strategies = list(product(range(K), repeat=N))

    best_pne = None
    best_delta_1 = 1000

    best_nopne = None
    best_delta_2 = 1000

    multiple_pne = False
    first_payoff = -1

    # Iterate through all strategy profiles to find PNE
    for strategy in tqdm(all_strategies):
        is_pne = True
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
                    if delta_2 > best_delta_2:
                        break

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
                best_pne = strategy
                best_delta_1 = delta_1
        else:
            if isprint:
                print("No PNE", strategy, sum(current_payoffs), delta_2)
            if delta_2 <= best_delta_2:
                best_nopne = strategy
                best_delta_2 = delta_2

    return best_pne, best_delta_1, best_nopne, best_delta_2, multiple_pne


def find_pne_2(weights, rewards, isprint=False):
    N, K = weights.shape

    # Generate all possible strategy profiles
    all_strategies = list(product(range(K), repeat=N))

    best_pne = None
    best_delta_1 = 1000

    best_nopne = None
    best_delta_2 = 1000

    multiple_pne = False
    first_payoff = -1
    best_payoff = 0

    for strategy in tqdm(all_strategies):
        strategy = list(strategy)
        weight = np.zeros(K)
        weight_choices = []
        for i, choice in enumerate(strategy):
            weight[choice] += weights[i][choice]
            weight_choices.append(weights[i][choice])

        weight_choices = np.array(weight_choices)

        personal_expected_rewards = (rewards / (weight + 1e-6))[strategy]
        personal_expected_rewards = personal_expected_rewards * weight_choices

        weight = np.tile(weight, N).reshape(N, -1)
        for i, choice in enumerate(strategy):
            weight[i][choice] -= weights[i][choice]

        weight_choices = np.tile(weight_choices.reshape(N, 1), [1, K])
        weight = weight + weight_choices
        reward_deviation = (
            np.tile(rewards.reshape(-1, K), [N, 1])
            / (weight + 1e-6)
            * weight_choices
        )

        for i, choice in enumerate(strategy):
            reward_deviation[i][choice] = -1e6

        reward_best_deviation = np.max(reward_deviation, axis=1)
        delta_rewards = reward_best_deviation - personal_expected_rewards
        if np.max(delta_rewards) <= 1e-6:
            is_pne = True
            delta_1 = -np.max(delta_rewards)
        else:
            is_pne = False
            delta_2 = np.max(delta_rewards)

        # Update the best PNE if necessary
        if is_pne:
            total_payoff = np.sum(personal_expected_rewards)
            if total_payoff > best_payoff:
                best_payoff = total_payoff
            if first_payoff == -1:
                first_payoff = total_payoff
            elif abs(total_payoff - first_payoff) > 1e-5:
                multiple_pne = True
            if isprint:
                print("PNE", strategy, total_payoff, delta_1)
            if delta_1 <= best_delta_1:
                best_pne = strategy
                best_delta_1 = delta_1
        else:
            if isprint:
                print("No PNE", strategy, np.sum(personal_expected_rewards), delta_2)
            if delta_2 <= best_delta_2:
                best_nopne = strategy
                best_delta_2 = delta_2

    return best_pne, best_delta_1, best_nopne, best_delta_2, multiple_pne, best_payoff

def calculate_delta(weights, rewards, isprint=False):
    best_pne, best_delta_1, best_nopne, best_delta_2, multiple_pne, best_payoff = find_pne_2(
        weights, rewards, isprint
    )
    if isprint:
        print(best_pne, best_delta_1, best_nopne, best_delta_2, multiple_pne, best_payoff)

    delta = min(best_delta_1, best_delta_2)
    return best_delta_1, best_delta_2, delta, best_payoff


if __name__ == "__main__":
    N, K = 3, 2

    isprint = True
    # PNE > NOPNE
    weights = np.tile([[1.0], [0.8], [0.4]], [1, K])
    rewards = np.array([1.0, 0.7])
    print(calculate_delta(weights, rewards, isprint=isprint))

    # NOPNE > PNE
    weights = np.tile([[0.9], [0.6], [0.4]], [1, K])
    rewards = np.array([1, 0.6])
    print(calculate_delta(weights, rewards, isprint=isprint))

    # Multiple PNE: there exist insufficient PNE
    N, K = 3, 3
    weights = np.tile([[0.8], [0.3], [0.2]], [1, K])
    rewards = np.array([0.6, 0.4, 0.2])
    print(calculate_delta(weights, rewards, isprint=isprint))

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
