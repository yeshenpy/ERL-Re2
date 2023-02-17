import numpy as np
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
from core import replay_memory
from core import ddpg as ddpg
from scipy.spatial import distance
from core import replay_memory
from parameters import Parameters
import torch
from core import utils
import scipy.signal
import torch.nn as nn
import math
import random

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

class Agent:
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Init population
        self.pop = []
        self.buffers = []
        self.all_actors = []
        for _ in range(args.pop_size):
            #self.pop.append(ddpg.GeneticAgent(args))
            genetic = ddpg.GeneticAgent(args)
            self.pop.append(genetic)
            self.all_actors.append(genetic.actor)

        # Init RL Agent


        self.rl_agent = ddpg.TD3(args)
        self.replay_buffer = utils.ReplayBuffer()

        self.all_actors.append(self.rl_agent.actor)

        self.ounoise = ddpg.OUNoise(args.action_dim)
        self.evolver = utils_ne.SSNE(self.args, self.rl_agent.critic, self.evaluate,self.rl_agent.state_embedding, self.args.prob_reset_and_sup, self.args.frac)

        # Population novelty
        self.ns_r = 1.0
        self.ns_delta = 0.1
        self.best_train_reward = 0.0
        self.time_since_improv = 0
        self.step = 1
        self.use_real = 0
        self.total_use = 0
        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None
        self.rl_agent_frames = 0

        self.old_fitness = None
        self.evo_times = 0


    def evaluate(self, agent: ddpg.GeneticAgent or ddpg.TD3, state_embedding_net, is_render=False, is_action_noise=False,
                 store_transition=True, net_index=None, is_random =False, rl_agent_collect_data = False,  use_n_step_return = False,PeVFA=None):
        total_reward = 0.0
        total_error = 0.0
        policy_params = torch.nn.utils.parameters_to_vector(list(agent.actor.parameters())).data.cpu().numpy().reshape([-1])
        state = self.env.reset()
        done = False

        state_list = []
        reward_list = []

        action_list = []
        policy_params_list =[]
        n_step_discount_reward = 0.0
        episode_timesteps = 0
        all_state = []
        all_action = []

        while not done:
            if store_transition:
                self.num_frames += 1; self.gen_frames += 1
                if rl_agent_collect_data:
                    self.rl_agent_frames +=1
            if self.args.render and is_render: self.env.render()
            
            if is_random:
                action = self.env.action_space.sample()
            else :
                action = agent.actor.select_action(np.array(state),state_embedding_net)
                if is_action_noise:
                    
                    action = (action + np.random.normal(0, 0.1, size=self.args.action_dim)).clip(-1.0, 1.0)
            all_state.append(np.array(state))
            all_action.append(np.array(action))
            # Simulate one step in environment
            next_state, reward, done, info = self.env.step(action.flatten())
            done_bool = 0 if episode_timesteps + 1 == 1000 else float(done)
            total_reward += reward
            n_step_discount_reward += math.pow(self.args.gamma,episode_timesteps)*reward
            state_list.append(state)
            reward_list.append(reward)
            policy_params_list.append(policy_params)
            action_list.append(action.flatten())

            transition = (state, action, next_state, reward, done_bool)
            if store_transition:
                next_action = agent.actor.select_action(np.array(next_state), state_embedding_net)
                self.replay_buffer.add((state, next_state, action, reward, done_bool, next_action ,policy_params))
                #self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)
            episode_timesteps += 1
            state = next_state

            if use_n_step_return:
                if self.args.time_steps <= episode_timesteps:
                    next_action = agent.actor.select_action(np.array(next_state), state_embedding_net)
                    param = nn.utils.parameters_to_vector(list(agent.actor.parameters())).data.cpu().numpy()
                    param = torch.FloatTensor(param).to(self.args.device)
                    param = param.repeat(1, 1)

                    # print("1")
                    next_state = torch.FloatTensor(np.array([next_state])).to(self.args.device)
                    next_action = torch.FloatTensor(np.array([next_action])).to(self.args.device)

                    input = torch.cat([next_state, next_action], -1)
                    # print("2")
                    next_Q1, next_Q2 = PeVFA.forward(input, param)
                    # print("3")
                    next_state_Q = torch.min(next_Q1, next_Q2).cpu().data.numpy().flatten()
                    n_step_discount_reward += math.pow(self.args.gamma,episode_timesteps) *next_state_Q[0]
                    break
        if store_transition: self.num_games += 1

        return {'n_step_discount_reward':n_step_discount_reward,'reward': total_reward,  'td_error': total_error, "state_list": state_list, "reward_list":reward_list, "policy_prams_list":policy_params_list, "action_list":action_list}


    def rl_to_evo(self, rl_agent: ddpg.TD3, evo_net: ddpg.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self):
        epochs = self.args.ns_epochs
        novelties = np.zeros(len(self.pop))
        for _ in range(epochs):
            transitions = self.replay_buffer.sample(self.args.batch_size)
            batch = replay_memory.Transition(*zip(*transitions))

            for i, net in enumerate(self.pop):
                novelties[i] += (net.get_novelty(batch))
        return novelties / epochs

    def train_ddpg(self, evo_times,all_fitness, state_list_list,reward_list_list, policy_params_list_list,action_list_list):
        bcs_loss, pgs_loss,c_q,t_q = [], [],[],[]


        if len(self.replay_buffer.storage) >= 5000:#self.args.batch_size * 5:


            before_rewards = np.zeros(len(self.pop))

            ddpg.hard_update(self.rl_agent.old_state_embedding, self.rl_agent.state_embedding)
            for gen in self.pop:
                ddpg.hard_update(gen.old_actor, gen.actor)

            discount_reward_list_list =[]
            for reward_list in reward_list_list:
                discount_reward_list = discount(reward_list,0.99)
                discount_reward_list_list.append(discount_reward_list)
            state_list_list = np.concatenate(np.array(state_list_list))
            #print("state_list_list ",state_list_list.shape)
            discount_reward_list_list = np.concatenate(np.array(discount_reward_list_list))
            #print("discount_reward_list_list ",discount_reward_list_list.shape)
            policy_params_list_list = np.concatenate(np.array(policy_params_list_list))
            #print("policy_params_list_list ", policy_params_list_list.shape)
            action_list_list = np.concatenate(np.array(action_list_list))
            pgl, delta,pre_loss,pv_loss,keep_c_loss= self.rl_agent.train(evo_times,all_fitness, self.pop , state_list_list, policy_params_list_list, discount_reward_list_list,action_list_list, self.replay_buffer ,int(self.gen_frames * self.args.frac_frames_train), self.args.batch_size, discount=self.args.gamma, tau=self.args.tau,policy_noise=self.args.TD3_noise,train_OFN_use_multi_actor=self.args.random_choose,all_actor=self.all_actors)
            after_rewards = np.zeros(len(self.pop))
        else:
            before_rewards = np.zeros(len(self.pop))
            after_rewards = np.zeros(len(self.pop))
            delta = 0.0
            pgl = 0.0
            pre_loss = 0.0
            keep_c_loss = [0.0]
            pv_loss = 0.0
        add_rewards = np.mean(after_rewards - before_rewards)
        return {'pv_loss':pv_loss,'bcs_loss': delta, 'pgs_loss': pgl,"current_q":0.0, "target_q":0.0, "pre_loss":pre_loss}, keep_c_loss, add_rewards

    def train(self):
        self.gen_frames = 0
        self.iterations += 1
 
        # ========================== EVOLUTION  ==========================
        # Evaluate genomes/individuals
        real_rewards = np.zeros(len(self.pop))
        fake_rewards = np.zeros(len(self.pop))
        MC_n_steps_rewards = np.zeros(len(self.pop))
        state_list_list = []

        reward_list_list = []
        policy_parms_list_list =[]
        action_list_list =[]


        if self.args.EA and self.rl_agent_frames>=self.args.init_steps:
            self.evo_times +=1
            random_num_num = random.random()
            if random_num_num< self.args.theta:
                for i, net in enumerate(self.pop):
                    for _ in range(self.args.num_evals):
                        episode = self.evaluate(net, self.rl_agent.state_embedding, is_render=False, is_action_noise=False,net_index=i)
                        real_rewards[i] += episode['reward']
                real_rewards /= self.args.num_evals
                all_fitness = real_rewards
            else :
                for i, net in enumerate(self.pop):
                    episode = self.evaluate(net, self.rl_agent.state_embedding, is_render=False, is_action_noise=False,net_index=i,use_n_step_return = True,PeVFA=self.rl_agent.PVN)
                    fake_rewards[i] += episode['n_step_discount_reward']
                    MC_n_steps_rewards[i]  +=episode['reward']
                all_fitness = fake_rewards

        else :
            all_fitness = np.zeros(len(self.pop))

        self.total_use +=1.0
        # all_fitness = 0.8 * rankdata(rewards) + 0.2 * rankdata(errors)

        keep_c_loss = [0.0 / 1000]
        min_fintess = 0.0
        best_old_fitness = 0.0
        temp_reward =0.0

        # Validation test for NeuroEvolution champion
        best_train_fitness = np.max(all_fitness)
        champion = self.pop[np.argmax(all_fitness)]

        test_score = 0
        
        if self.args.EA and self.rl_agent_frames>=self.args.init_steps:
            for eval in range(10):
                episode = self.evaluate(champion, self.rl_agent.state_embedding, is_render=True, is_action_noise=False, store_transition=False)
                test_score += episode['reward']
        test_score /= 10.0

        # NeuroEvolution's probabilistic selection and recombination step
        if self.args.EA:
            elite_index = self.evolver.epoch(self.pop, all_fitness)
        else :
            elite_index = 0
        # ========================== DDPG or TD3 ===========================
        # Collect experience for training

        if self.args.RL:
            is_random = (self.rl_agent_frames < self.args.init_steps)
            episode = self.evaluate(self.rl_agent, self.rl_agent.state_embedding, is_action_noise=True, is_random=is_random,rl_agent_collect_data=True)

            state_list_list.append(episode['state_list'])
            reward_list_list.append(episode['reward_list'])
            policy_parms_list_list.append(episode['policy_prams_list'])
            action_list_list.append(episode['action_list'])

            if self.rl_agent_frames>=self.args.init_steps:
                losses, _, add_rewards = self.train_ddpg(self.evo_times,all_fitness, state_list_list,reward_list_list,policy_parms_list_list,action_list_list)
            else :
                losses = {'bcs_loss': 0.0, 'pgs_loss': 0.0 ,"current_q":0.0, "target_q":0.0, "pv_loss":0.0, "pre_loss":0.0}
                add_rewards = np.zeros(len(self.pop)) 
        else :
            losses = {'bcs_loss': 0.0, 'pgs_loss': 0.0 ,"current_q":0.0, "target_q":0.0,"pv_loss":0.0, "pre_loss":0.0}

            add_rewards = np.zeros(len(self.pop))

        L1_before_after = np.zeros(len(self.pop))

        # Validation test for RL agent
        testr = 0
        
        if self.args.RL:
            for eval in range(10):
                ddpg_stats = self.evaluate(self.rl_agent, self.rl_agent.state_embedding,store_transition=False, is_action_noise=False)
                testr += ddpg_stats['reward']
            testr /= 10.0
  
        #Sync RL Agent to NE every few steps
        if self.args.EA and self.args.RL and  self.rl_agent_frames>=self.args.init_steps:
           if self.iterations % self.args.rl_to_ea_synch_period == 0:
               # Replace any index different from the new elite
               replace_index = np.argmin(all_fitness)
               if replace_index == elite_index:
                   replace_index = (replace_index + 1) % len(self.pop)

               self.rl_to_evo(self.rl_agent, self.pop[replace_index])
               self.evolver.rl_policy = replace_index
               print('Sync from RL --> Nevo')


        self.old_fitness = all_fitness
        # -------------------------- Collect statistics --------------------------


        return {
            'min_fintess':min_fintess,
            'best_old_fitness':best_old_fitness,
            'new_fitness':temp_reward,
            'best_train_fitness': best_train_fitness,
            'test_score': test_score,
            'elite_index': elite_index,
            'ddpg_reward': testr,
            'pvn_loss':losses['pv_loss'],
            'pg_loss': np.mean(losses['pgs_loss']),
            'bc_loss': np.mean(losses['bcs_loss']),
            'current_q': np.mean(losses['current_q']),
            'target_q':np.mean(losses['target_q']),
            'pre_loss': np.mean(losses['pre_loss']),
            'pop_novelty': np.mean(0),
            'before_rewards':all_fitness,
            'add_rewards':add_rewards,
            'l1_before_after':L1_before_after,
            'keep_c_loss':np.mean(keep_c_loss)
        }


class Archive:
    """A record of past behaviour characterisations (BC) in the population"""

    def __init__(self, args):
        self.args = args
        # Past behaviours
        self.bcs = []

    def add_bc(self, bc):
        if len(self.bcs) + 1 > self.args.archive_size:
            self.bcs = self.bcs[1:]
        self.bcs.append(bc)

    def get_novelty(self, this_bc):
        if self.size() == 0:
            return np.array(this_bc).T @ np.array(this_bc)
        distances = np.ravel(distance.cdist(np.expand_dims(this_bc, axis=0), np.array(self.bcs), metric='sqeuclidean'))
        distances = np.sort(distances)
        return distances[:self.args.ns_k].mean()

    def size(self):
        return len(self.bcs)