import random
import numpy as np
from core.ddpg import GeneticAgent, hard_update
from typing import List
from core import replay_memory
import fastrand, math
import torch
import torch.distributions as dist
from core.mod_utils import is_lnorm_key
from parameters import Parameters
import os


class SSNE:
    def __init__(self, args: Parameters, critic, evaluate, state_embedding, prob_reset_and_sup, frac):
        self.state_embedding = state_embedding
        self.current_gen = 0
        self.args = args;
        self.critic = critic
        self.prob_reset_and_sup = prob_reset_and_sup
        
        self.frac = frac
        self.population_size = self.args.pop_size
        self.num_elitists = int(self.args.elite_fraction * args.pop_size)
        self.evaluate = evaluate
        self.stats = PopulationStats(self.args)
        if self.num_elitists < 1: self.num_elitists = 1
 
        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1: GeneticAgent, gene2: GeneticAgent):
        # Evaluate the parents
        trials = 5
        if self.args.opstat and self.stats.should_log():
            test_score_p1 = 0
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p1 += episode['reward']
            test_score_p1 /= trials

            test_score_p2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene2, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p2 += episode['reward']
            test_score_p2 /= trials

        b_1 = None 
        b_2 = None 
        for param1, param2 in zip(gene1.actor.parameters(), gene2.actor.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data 
            if len(W1.shape) == 1:
                b_1 = W1
                b_2 = W2
            
                
       
        for param1, param2 in zip(gene1.actor.parameters(), gene2.actor.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2: #Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr,:] = W2[ind_cr,:]
                        b_1[ind_cr] = b_2[ind_cr]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr,:] = W1[ind_cr,:]
                        b_2[ind_cr] = b_1[ind_cr]

        # Evaluate the children
        if self.args.opstat and self.stats.should_log():
            test_score_c1 = 0
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c1 += episode['reward']
            test_score_c1 /= trials

            test_score_c2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c2 += episode['reward']
            test_score_c2 /= trials

            if self.args.verbose_crossover:
                print("==================== Classic Crossover ======================")
                print("Parent 1", test_score_p1)
                print("Parent 2", test_score_p2)
                print("Child 1", test_score_c1)
                print("Child 2", test_score_c2)

            self.stats.add({
                'cros_parent1_fit': test_score_p1,
                'cros_parent2_fit': test_score_p2,
                'cros_child_fit': np.mean([test_score_c1, test_score_c2]),
                'cros_child1_fit': test_score_c1,
                'cros_child2_fit': test_score_c2,
            })
    
    def distilation_crossover(self, gene1: GeneticAgent, gene2: GeneticAgent):
        new_agent = GeneticAgent(self.args)
        new_agent.buffer.add_latest_from(gene1.buffer, self.args.individual_bs // 2)
        new_agent.buffer.add_latest_from(gene2.buffer, self.args.individual_bs // 2)
        new_agent.buffer.shuffle()

        hard_update(new_agent.actor, gene2.actor)
        batch_size = min(128, len(new_agent.buffer))
        iters = len(new_agent.buffer) // batch_size
        losses = []
        for epoch in range(12):
            for i in range(iters):
                batch = new_agent.buffer.sample(batch_size)
                losses.append(new_agent.update_parameters(batch, gene1.actor, gene2.actor, self.critic))

        if self.args.opstat and self.stats.should_log():

            test_score_p1 = 0
            trials = 5
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p1 += episode['reward']
            test_score_p1 /= trials

            test_score_p2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene2, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p2 += episode['reward']
            test_score_p2 /= trials

            test_score_c = 0
            for eval in range(trials):
                episode = self.evaluate(new_agent, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c += episode['reward']
            test_score_c /= trials

            if self.args.verbose_crossover:
                print("==================== Distillation Crossover ======================")
                print("MSE Loss:", np.mean(losses[-40:]))
                print("Parent 1", test_score_p1)
                print("Parent 2", test_score_p2)
                print("Crossover performance: ", test_score_c)

            self.stats.add({
                'cros_parent1_fit': test_score_p1,
                'cros_parent2_fit': test_score_p2,
                'cros_child_fit': test_score_c,
            })

        return new_agent

    def mutate_inplace(self, gene: GeneticAgent):
        trials = 5
        if self.stats.should_log():
            test_score_p = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p += episode['reward']
            test_score_p /= trials

        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = self.prob_reset_and_sup
        reset_prob = super_mut_prob + self.prob_reset_and_sup

        num_params = len(list(gene.actor.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.actor.state_dict()

        for i, key in enumerate(model_params): #Mutate each param

            if is_lnorm_key(key):
                continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2: #Weights, no bias

                ssne_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_variables = W.shape[0]
                    # Crossover opertation [Indexed by row]
                    for index in range(num_variables):
                        #ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                        #ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                         
                         
                        random_num_num = random.random()
                        if random_num_num < 1.0 :
                            #print(W)
                            index_list = random.sample(range(W.shape[1]), int(W.shape[1]*self.frac) )
                            random_num = random.random()
                            if random_num < super_mut_prob:  # Super Mutation probability
                                for ind in index_list:      
                                    W[index, ind] += random.gauss(0, super_mut_strength * W[index, ind])
                            elif random_num < reset_prob:  # Reset probability        
                                for ind in index_list:    
                                    W[index, ind] =  random.gauss(0, 1)
                            else:  # mutation even normal
                                for ind in index_list:    
                                    W[index, ind] += random.gauss(0, mut_strength *W[index, ind])
                        
                            # Regularization hard limit
                            W[index, :] = np.clip(W[index, :], a_min=-1000000, a_max=1000000)

        if self.stats.should_log():
            test_score_c = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c += episode['reward']
            test_score_c /= trials

            self.stats.add({
                'mut_parent_fit': test_score_p,
                'mut_child_fit': test_score_c,
            })

            if self.args.verbose_crossover:
                print("==================== Mutation ======================")
                print("Fitness before: ", test_score_p)
                print("Fitness after: ", test_score_c)

    def proximal_mutate(self, gene: GeneticAgent, mag):
        # Based on code from https://github.com/uber-research/safemutations 
        trials = 5
        if self.stats.should_log():
            test_score_p = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p += episode['reward']
            test_score_p /= trials

        model = gene.actor

        batch = gene.buffer.sample(min(self.args.mutation_batch_size, len(gene.buffer)))
        state, _, _, _, _ = batch
        output = model(state, self.state_embedding)

        params = model.extract_parameters()
        tot_size = model.count_parameters()
        num_outputs = output.size()[1]

        if self.args.mutation_noise:
            mag_dist = dist.Normal(self.args.mutation_mag, 0.02)
            mag = mag_dist.sample()

        # initial perturbation
        normal = dist.Normal(torch.zeros_like(params), torch.ones_like(params) * mag)
        delta = normal.sample()
        # uniform = delta.clone().detach().data.uniform_(0, 1)
        # delta[uniform > 0.1] = 0.0

        # we want to calculate a jacobian of derivatives of each output's sensitivity to each parameter
        jacobian = torch.zeros(num_outputs, tot_size).to(self.args.device)
        grad_output = torch.zeros(output.size()).to(self.args.device)

        # do a backward pass for each output
        for i in range(num_outputs):
            model.zero_grad()
            grad_output.zero_()
            grad_output[:, i] = 1.0

            output.backward(grad_output, retain_graph=True)
            jacobian[i] = model.extract_grad()

        # summed gradients sensitivity
        scaling = torch.sqrt((jacobian**2).sum(0))
        scaling[scaling == 0] = 1.0
        scaling[scaling < 0.01] = 0.01
        delta /= scaling
        new_params = params + delta

        model.inject_parameters(new_params)

        if self.stats.should_log():
            test_score_c = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c += episode['reward']
            test_score_c /= trials

            self.stats.add({
                'mut_parent_fit': test_score_p,
                'mut_child_fit': test_score_c,
            })

            if self.args.verbose_crossover:
                print("==================== Mutation ======================")
                print("Fitness before: ", test_score_p)
                print("Fitness after: ", test_score_c)
                print("Mean mutation change:", torch.mean(torch.abs(new_params - params)).item())

    def clone(self, master: GeneticAgent, replacee: GeneticAgent):  # Replace the replacee individual with master
        for target_param, source_param in zip(replacee.actor.parameters(), master.actor.parameters()):
            target_param.data.copy_(source_param.data)
        replacee.buffer.reset()
        replacee.buffer.add_content_of(master.buffer)

    def reset_genome(self, gene: GeneticAgent):
        for param in (gene.actor.parameters()):
            param.data.copy_(param.data)

    @staticmethod
    def sort_groups_by_fitness(genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)
    
    @staticmethod
    def get_distance(gene1: GeneticAgent, gene2: GeneticAgent):
        batch_size = min(256, min(len(gene1.buffer), len(gene2.buffer)))
        batch_gene1 = gene1.buffer.sample_from_latest(batch_size, 1000)
        batch_gene2 = gene2.buffer.sample_from_latest(batch_size, 1000)

        return gene1.actor.get_novelty(batch_gene2) + gene2.actor.get_novelty(batch_gene1)
    
    @staticmethod
    def sort_groups_by_distance(genomes, pop):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                groups.append((second, first, SSNE.get_distance(pop[first], pop[second])))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    def epoch(self, pop: List[GeneticAgent], fitness_evals):
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = np.argsort(fitness_evals)[::-1]
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # COMPUTE RL_SELECTION RATE
        if self.rl_policy is not None: # RL Transfer happened
            self.selection_stats['total'] += 1.0

            if self.rl_policy in elitist_index: self.selection_stats['elite'] += 1.0
            elif self.rl_policy in offsprings: self.selection_stats['selected'] += 1.0
            elif self.rl_policy in unselects: self.selection_stats['discarded'] += 1.0
            self.rl_policy = None

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            try: replacee = unselects.pop(0)
            except: replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee])

        # Crossover between elite and offsprings for the unselected genes with 100 percent probability
        if self.args.distil:
            if self.args.distil_type == 'fitness':
                sorted_groups = SSNE.sort_groups_by_fitness(new_elitists + offsprings, fitness_evals)
            elif self.args.distil_type == 'dist':
                sorted_groups = SSNE.sort_groups_by_distance(new_elitists + offsprings, pop)
            else:
                raise NotImplementedError('Unknown distilation type')
            for i, unselected in enumerate(unselects):
                first, second, _ = sorted_groups[i % len(sorted_groups)]
                if fitness_evals[first] < fitness_evals[second]:
                    first, second = second, first
                self.clone(self.distilation_crossover(pop[first], pop[second]), pop[unselected])
        else:
            if len(unselects) % 2 != 0:  # Number of unselects left should be even
                unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
            for i, j in zip(unselects[0::2], unselects[1::2]):
                off_i = random.choice(new_elitists)
                off_j = random.choice(offsprings)
                self.clone(master=pop[off_i], replacee=pop[i])
                self.clone(master=pop[off_j], replacee=pop[j])
                self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i in offsprings:
            if random.random() < self.args.crossover_prob:
                others = offsprings.copy()
                others.remove(i)
                off_j = random.choice(others)
                self.clone(self.distilation_crossover(pop[i], pop[off_j]), pop[i])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    if self.args.proximal_mut:
                        self.proximal_mutate(pop[i], mag=self.args.mutation_mag)
                    else:
                        self.mutate_inplace(pop[i])

        if self.stats.should_log():
            self.stats.log()
        self.stats.reset()
        return new_elitists[0]


def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))


class PopulationStats:
    def __init__(self, args: Parameters, file='population.csv'):
        self.data = {}
        self.args = args
        self.save_path = os.path.join(args.save_foldername, file)
        self.generation = 0

        if not os.path.exists(args.save_foldername):
            os.makedirs(args.save_foldername)

    def add(self, res):
        for k, v in res.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log(self):
        with open(self.save_path, 'a+') as f:
            if self.generation == 0:
                f.write('generation,')
                for i, k in enumerate(self.data):
                    if i > 0:
                        f.write(',')
                    f.write(k)
                f.write('\n')

            f.write(str(self.generation))
            f.write(',')
            for i, k in enumerate(self.data):
                if i > 0:
                    f.write(',')
                f.write(str(np.mean(self.data[k])))
            f.write('\n')

    def should_log(self):
        return self.generation % self.args.opstat_freq == 0 and self.args.opstat

    def reset(self):
        for k in self.data:
            self.data[k] = []
        self.generation += 1


