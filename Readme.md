
# Two-scale Representation-based Evolutionary Reinforcement Learning

Official code for the paper "Two-scale Representation-based Evolutionary Reinforcement Learning". 

**TSR-ERL** is a novel framework to integrate EA and RL.
The cornerstone of TSR-ERL is two-scale representation:
all EA and RL policies share the same nonlinear state representation while maintaining individual linear policy representations.
The state representation conveys expressive common features of the environment learned by all the agents collectively;
the linear policy representation provides a favorable space for efficient policy optimization, where novel behavior-level crossover and mutation operations can be performed.
Moreover, the linear policy representation allows convenient generalization of policy fitness with the help of Policy-extended Value Function Approximator (PeVFA),
further improving the sample efficiency.

# Installation
Known dependencies: MUJOCO 200,
Python (3.6.13), gym (0.10.5), torch (1.1.0), fastrand, wandb, mujoco_py=2.0.2.13

## Hyperparameter
- `-env`: define environment in MUJOCO
- `-OFF_TYPE`: the type of PeVFA, default 1
- `-pr`: The policy representation size
- `-pop_size`: population size
- `-prob_reset_and_sup`: Probability of resetting parameters and super mutations
- `-time_steps`: The step length of H-step return 
- `-theta`: Probability of using MC estimates
- `-frac`: Mutation ratio
- `-gamma`: Gamma for RL
- `-TD3_noise`: Noise for TD3
- `-EA`: Whether to use EA
- `-RL`: Whether to use RL
- `-K`: The number of individuals selected to optimize the shared representation from the population
- `-EA_actor_alpha`: The coefficient used to balance the weights of PeVFA loss
- `-actor_alpha`: The coefficient used to balance the weights of RL loss
- `-tau`: The coefficient for soft updates
- `-seed`: Seed, default from 1 to 5
- `-logdir`: Log Location

## Code structure

- `./parameters.py`: Hyperparameters setting for TSR-ERL

- `./run_tsr.py`: Code to run TSR-ERL

- `./core/agent.py`: Algorithm flow 

- `./core/ddpg.py`: The core code of TSR-ERL (DDPG and TD3 version)

- `./core/mod_utils`: Some Functions for TSR-ERL

- `./core/replay_memory`: replay buffer for TSR-ERL

- `./core/utils`: Some Functions for TSR-ERL

- `./run.sh`: command-line file 

## How to run

We implement and provide TSR-ERL based on TD3 and DDPG. 
Run the `run.sh` file directly, Hyperparameter settings can be found in the paper.



