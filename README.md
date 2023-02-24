
# ICLR 2023: ERL-Re$^2$: Efficient Evolutionary Reinforcement Learning with Shared State Representation and Individual Policy Representation

Official code for the paper "ERL-Re$^2$: Efficient Evolutionary Reinforcement Learning with Shared State Representation and Individual Policy Representation" (<https://arxiv.org/abs/2210.17375>). **ERL-Re$^2$ achieves current SOTA in the ERL field**.

**ERL-Re$^2$** is a novel framework to integrate EA and RL. The cornerstone of ERL-Re$^2$ is two-scale representation: all EA and RL policies share the same nonlinear state representation while maintaining individual linear policy representations. The state representation conveys expressive common features of the environment learned by all the agents collectively; the linear policy representation provides a favorable space for efficient policy optimization, where novel behavior-level crossover and mutation operations can be performed. Moreover, the linear policy representation allows convenient generalization of policy fitness with the help of Policy-extended Value Function Approximator (PeVFA), further improving the sample efficiency. This repository is based on <https://github.com/crisbodnar/pderl>.


# Installation
Known dependencies: MUJOCO 200,
Python (3.6.13), gym (0.10.5), torch (1.1.0), fastrand, wandb, mujoco_py=2.0.2.13

## Hyperparameter
- `-env`: define environment in MUJOCO
- `-OFF_TYPE`: the type of PeVFA, default 1
- `-pr`: The policy representation size, default 64
- `-pop_size`: population size, default 5
- `-prob_reset_and_sup`: Probability of resetting parameters and super mutations, default 0.05
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

- `./parameters.py`: Hyperparameters setting for ERL-Re$^2$

- `./run_re2.py`: Code to run ERL-Re$^2$

- `./core/agent.py`: Algorithm flow 

- `./core/ddpg.py`: The core code of ERL-Re$^2$ (DDPG and TD3 version)

- `./core/mod_utils`: Some Functions for ERL-Re$^2$

- `./core/replay_memory`: replay buffer for ERL-Re$^2$

- `./core/utils`: Some Functions for ERL-Re$^2$

- `./run.sh`: command-line file 

## How to run

We implement and provide ERL-Re$^2$ based on TD3. 
Run the `run.sh` file directly, Hyperparameter settings can be found in the paper.


## Publication

If you find this repository useful, please cite our paper:

    @inproceedings{
    li2023erlre,
    title={{ERL}-Re\${\textasciicircum}2\$: Efficient Evolutionary Reinforcement Learning with Shared State Representation and Individual Policy Representation },
    author={Pengyi Li and Hongyao Tang and Jianye HAO and YAN ZHENG and Xian Fu and Zhaopeng Meng},
    booktitle={International Conference on Learning Representations},
    year={2023}
    }
