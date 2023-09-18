from itertools import product
import random

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
    N = len(weights)
    K = len(rewards)
    
    # Initialize payoff list
    payoffs = [0] * N
    
    # Calculate the sum of weights for each arm
    sum_weights_per_arm = [0] * K
    for i in range(N):
        arm = strategy[i]
        sum_weights_per_arm[arm] += weights[i]
        
    # Calculate payoff for each agent
    for i in range(N):
        arm = strategy[i]
        payoffs[i] = (weights[i] / sum_weights_per_arm[arm]) * rewards[arm]
        
    return payoffs

def find_pne(weights, rewards):
    """
    Find the most efficient Pure Nash Equilibrium.
    
    Parameters:
    weights (list): List of weights for each agent.
    rewards (list): List of rewards for each arm.
    
    Returns:
    tuple: Most efficient PNE and its total payoff.
    """
    N = len(weights)
    K = len(rewards)
    
    # Generate all possible strategy profiles
    all_strategies = list(product(range(K), repeat=N))
    
    best_pne = None
    best_total_payoff = 0

    best_delta_1 = 1000

    no_pne = []
    
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
            print("PNE", strategy, total_payoff, delta_1)
            if total_payoff >= best_total_payoff and delta_1 <= best_delta_1:
                best_total_payoff = total_payoff
                best_pne = strategy
                best_delta_1 = delta_1
        else:
            no_pne.append((sum(current_payoffs), delta_2))
            print("No PNE", strategy, sum(current_payoffs), delta_2)
                
    return best_pne, best_total_payoff, best_delta_1, no_pne

# Generate random weights and rewards for 3 agents and 3 arms
# random.seed(42)  # For reproducibility

N = 3  # Number of agents
K = 3  # Number of arms

# PNE > NOPNE
weights = [0.8, 0.6, 0.1]
rewards = [0.9, 0.8, 0.6]

# NOPNE > PNE
weights = [0.8, 0.6, 0.1]
rewards = [0.9, 0.8, 0.5]

# weights = [round(random.uniform(0.1, 1), 1) for _ in range(N)]
# rewards = [round(random.uniform(0.1, 1), 1) for _ in range(K)]

print("Weights:", weights)
print("Rewards:", rewards)

# Find the most efficient PNE
best_pne, best_total_payoff, best_delta_pne, no_pne = find_pne(weights, rewards)

l = [ele[1] for ele in no_pne if ele[0] >= best_total_payoff - 1e-5]
delta_nopne = max(l) if len(l) > 0 else 100

print(best_pne, best_total_payoff, best_delta_pne, delta_nopne)

# if best_delta_1 > delta_2:
#     break
