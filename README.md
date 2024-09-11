### Developed by : Ashwin Kumar S
### Ref.No : 212222240013

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi,P,gamma=1.0,theta=1e-10):

    prev_V=np.zeros(len(P))

    while True:
        V=np.zeros(len(P))
        for s in range(len(P)):
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
        if(np.max(np.abs(prev_V-V))<theta):
            break
        prev_V=V.copy()
    return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the first policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing the two policies
if np.max(V1) > np.max(V2):
    print("Policy 1 (pi_1) is better based on the maximum state value.")
else:
    print("Policy 2 (pi_2) is better based on the maximum state value.")
```
## OUTPUT:

### first policy
![image](https://github.com/user-attachments/assets/a10a98e6-95c2-44e5-9d8e-b926e9bd6ae0)
![image](https://github.com/user-attachments/assets/dd936204-0f74-4659-9b17-a6105bcabb65)
![image](https://github.com/user-attachments/assets/a0a16a9d-6aee-46b1-844e-e70d7210698f)

### second policy
![image](https://github.com/user-attachments/assets/7877bce5-e59a-460d-b54c-c72d3919da8f)
![image](https://github.com/user-attachments/assets/05184a24-ddcc-4606-ae0b-c18ea4dc1809)
![image](https://github.com/user-attachments/assets/93a16b71-50c9-4fc7-a088-41e86b0ab269)

### Comparison
![image](https://github.com/user-attachments/assets/a24cf606-1fca-4f45-84f8-37eeeab7b7c1)

### Conclusion
![image](https://github.com/user-attachments/assets/e6ab186b-5aee-4c20-ab18-fee0faa50604)


## RESULT:
Thus the Given Policy have been Evaluated and Optimal Policy has been Computed using Python Programming.


