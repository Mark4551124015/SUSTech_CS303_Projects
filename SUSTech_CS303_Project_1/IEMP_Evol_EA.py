import numpy as np

from networkx import DiGraph
import random as rand
from queue import Queue
import os
import time
import matplotlib.pyplot as plt
import argparse
from itertools import accumulate
from bisect import bisect_right
import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', type=str, help='test case number')
parser = parser.parse_args()
casenum=parser.c
NET_WORK='cases/Heuristic/map{}/dataset{}'.format(casenum,casenum)
SEED='cases/Heuristic/map{}/seed'.format(casenum)
SEED_B='cases/Heuristic/map{}/seed_balanced'.format(casenum)
OUTPUT='cases/Heuristic/map{}/score'.format(casenum)


def prob(p:float) -> bool:
    if rand.random() < p:
        return True
    return False

class IEM:
    @classmethod
    def add_edge(cls, G:DiGraph, OG: DiGraph, u, v, c1, c2, neib: dict, cut_off):
        neib.setdefault(u, None)
        if not neib[u]: neib[u]=set({})
        neib[u].add(v)
        OG.add_edge(u, v, c1=c1,c2=c2)
        if (float(c1) >= cut_off or float(c2) >= cut_off):
            G.add_edge(u, v, c1=c1,c2=c2)
    @classmethod
    def create(cls, net_path, seed_path, cut_off, sample_N, K):
        print("N",sample_N,"\tcutoff_edge",cut_off)
        g,og = DiGraph(), DiGraph()
        neib = {}
        with open(net_path, 'r') as f:
            n, m = f.readline().strip().split(' ')
            for line in f:
                u, v, c1, c2 = line.strip().split(' ')
                cls.add_edge(g, og, u,v,c1,c2, neib, cut_off)
        i1,i2=set({}),set({})
        with open(seed_path, 'r') as f:
            k1, k2 = f.readline().strip().split(' ')
            k1, k2 = int(k1), int(k2)
            for index, line in enumerate(f):
                node = line.strip()
                g.add_node(node)
                og.add_node(node)
                if index < k1:
                    i1.add(node)
                else:
                    i2.add(node)
        iem = cls(g, og, i1, i2, neib, sample_N, K)
        return iem
    
    def prop(self, g:DiGraph, ci, sol:tuple[set,set]) -> tuple[set,set]:
        s1,s2 = sol
        queue = Queue()
        new_nodes=set({})
        if ci == 'c1':
            new_nodes = s1.union(self.i1)
        else:
            new_nodes = s2.union(self.i2)
        exposed, activated = set(), set()
        exposed.union(new_nodes)
        activated.union(new_nodes)
        for i in new_nodes:
            queue.put(i)
        while not queue.empty():
            node = queue.get()
            self.neib.setdefault(node, set({}))
            exposed = exposed.union(self.neib[node])
            for v in g[node].keys():
                c = g[node][v][ci]
                if prob(float(c)) and (v not in activated):
                    queue.put(v)
                    activated.add(v)
        return exposed, activated
    
    def __init__(self, g: DiGraph, og: DiGraph, i1:set, i2:set,neib:dict, sample_N, K:int) -> None:
        self.g:DiGraph = g
        self.og:DiGraph = og
        self.i1,self.i2 = set(),set()
        self.i1 = self.i1.union(i1)
        self.i2 = self.i2.union(i2)
        self.neib = neib
        self.N = sample_N
        self.pool_1 = set(g.nodes).difference(i1)
        self.pool_2 = set(g.nodes).difference(i2)
        self.K = K

    def get_pool(self,s1:set,s2:set):
        return self.pool_1.difference(s1), self.pool_2.difference(s2)

    def init_solution(self):
        pool_1, pool_2 = self.get_pool(set(),set())
        for node in pool_1:
            self.neib.setdefault(node, None)
            if not self.neib[node]: self.neib[node]=set({})
        for node in pool_2:
            self.neib.setdefault(node, None)
            if not self.neib[node]: self.neib[node]=set({})
        pool_1_w = [(node, len(self.neib[node])**3 )for node in pool_1]
        pool_2_w = [(node, len(self.neib[node])**3 )for node in pool_2]

        sum_1 = sum([node[-1] for node in pool_1_w])
        sum_2 = sum([node[-1] for node in pool_2_w])

        wheel_1 = list(accumulate([i[-1]/sum_1 for i in pool_1_w]))
        wheel_2 = list(accumulate([i[-1]/sum_2 for i in pool_2_w]))
        s1,s2 = set({}), set({})
        # node = rand.choice(list(pool_1))
        node = pool_1_w[bisect_right(wheel_1, rand.random())][0]
        pool_1.remove(node)
        s1.add(node)
        # node = rand.choice(list(pool_2))
        node = pool_2_w[bisect_right(wheel_2, rand.random())][0]
        pool_2.remove(node)
        s2.add(node)
        for i in range(self.K-2):
            if prob(0.5):
                # node = rand.choice(list(pool_1))
                node = pool_1_w[bisect_right(wheel_1, rand.random())][0]
                while node not in pool_1:
                    node = pool_1_w[bisect_right(wheel_1, rand.random())][0]
                pool_1.remove(node)
                s1.add(node)
            else:
                # node = rand.choice(list(pool_2))
                node = pool_2_w[bisect_right(wheel_2, rand.random())][0]
                while node not in pool_2:
                    node = pool_2_w[bisect_right(wheel_2, rand.random())][0]
                pool_2.remove(node)
                s2.add(node)
        sol = (s1, s2)
        return (sol, self.value(sol))

    
    def value(self, sol:tuple[set,set]) -> int:
        tmp = []
        for i in range(self.N):
            r1,ac1 = self.prop(self.g,'c1', sol)
            r2,ac2 = self.prop(self.g,'c2', sol)
            delta = r1.difference(r2).union(r2.difference(r1)) 
            tmp.append(len(self.g.nodes) - len(delta))
        return np.average(tmp)
    

    def save(self, seed_b_path, sol):
        (s1, s2), score = sol
        with open(seed_b_path, 'w') as f:
            f.write('%d %d\r\n' % (len(s1), len(s2)))
            for data in s1:
                f.write('%s\r\n' % data)
            for data in s2:
                f.write('%s\r\n' % data)

class EA:
    def __init__(self, population_size:int, generations:int, mutation_rate:float, crossover_rate:float, special_rate:float, patient:int, problem_instance:IEM):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.problem_instance = problem_instance
        self.special_rate = special_rate
        self.patient = patient

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.problem_instance.init_solution()
            population.append(individual)
        return population

    def select_parents(self, population):
        # 这里可以使用适应度函数选择父代个体
        info = [individual for individual in population]
        info.sort(key=lambda x: x[-1])
        fit = [i[-1] for i in info]
        min_fit = min(fit)
        fit = [(i - min_fit)**3 for i in fit]
        # Create roulette wheel.
        sum_fit = sum(fit)
        wheel = list(accumulate([i/sum_fit for i in fit]))
        father = info[bisect_right(wheel, rand.random())]
        mother = info[bisect_right(wheel, rand.random())]
        return father, mother
    
    def crossover(self, parent1, parent2):
        (s1_A, s2_A), score1 = copy.deepcopy(parent1)
        (s1_B, s2_B), score2 = copy.deepcopy(parent2)
        r = min ([self.problem_instance.K // 2 // 2, len(s1_A)//2, len(s1_B)//2, len(s2_A)//2, len(s2_B)//2])
        for i in range(r):
            nodeA = rand.choice(list(s1_A))
            nodeB = rand.choice(list(s1_B))
            if prob(self.crossover_rate):
                s1_A.remove(nodeA)
                s1_A.add(nodeB)
                s1_B.remove(nodeB)
                s1_B.add(nodeA)
            nodeA = rand.choice(list(s2_A))
            nodeB = rand.choice(list(s2_B))
            if prob(self.crossover_rate):
                s2_A.remove(nodeA)
                s2_A.add(nodeB)
                s2_B.remove(nodeB)
                s2_B.add(nodeA)
        son1 = (s1_A, s2_A)
        son2 = (s1_B, s2_B)
        return (son1, 0), (son2, 0)

    def mutate(self, individual):
        (s1,s2), score = copy.deepcopy(individual)
        s1_, s2_ = set(), set()
        s1_ = s1_.union(s1)
        s2_ = s2_.union(s2)
        pool_1, pool_2 = self.problem_instance.get_pool(s1,s2)

        for i in range(self.problem_instance.K // 2):
            nodeA = rand.choice(list(s1_))
            nodeB = rand.choice(list(pool_1))
            if prob(self.mutation_rate):
                if prob(self.special_rate):
                    if prob(0.5):
                        if len(s1_)+len(s2_) < self.problem_instance.K:
                            s1_.add(nodeB)
                            pool_1.remove(nodeB)
                    else:
                        if len(s1_)>1:
                            s1_.remove(nodeA)
                            pool_1.add(nodeA)
                else:
                    s1_.add(nodeB)
                    pool_1.remove(nodeB)
                    s1_.remove(nodeA)
                    pool_1.add(nodeA)

            nodeA = rand.choice(list(s2_))
            nodeB = rand.choice(list(pool_2))
            if prob(self.mutation_rate):
                if prob(self.special_rate):
                    if prob(0.5):
                        if len(s1_)+len(s2_) < self.problem_instance.K:
                            s2_.add(nodeB)
                            pool_2.remove(nodeB)
                    else:
                        if len(s2_)>1:
                            s2_.remove(nodeA)
                            pool_2.add(nodeA)
                else:
                    s2_.add(nodeB)
                    pool_2.remove(nodeB)
                    s2_.remove(nodeA)
                    pool_2.add(nodeA)
            new = (s1_, s2_)
        return new, self.problem_instance.value(new)

    def evolve(self):
        record = []
        cnt = 0
        population = self.initialize_population()
        best_fitness = float('-inf')
        best_solution = set({}), best_fitness
        start = time.time()
        for generation in range(self.generations):
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                population.extend([child1, child2])
            population.sort(key=lambda x: -x[-1])
            population = population[:self.population_size]
            # 更新最佳解
            old_fitness = best_fitness
            for individual in population:
                sol, fitness = individual
                record.append(fitness)
                if fitness >= best_fitness and (sol != best_solution[0]):
                    best_solution = individual
                    best_fitness = fitness
            if best_fitness == old_fitness:
                cnt += 1
                if cnt > self.patient:
                    break
            else:
                cnt = 0
            
            print("{}    score {}   solution:{}".format(generation, best_fitness.__round__(2), best_solution))
        end = time.time()
        print(f"time cost {round(end - start, 2)}s")
        return best_solution, record


if __name__=='__main__':
    iem = IEM.create(NET_WORK, SEED, cut_off=0.05, sample_N=5, K=5)
    node_length = len(set(iem.g.nodes))
    edge_length = len(set(iem.g.edges))
    print("Nodes: ",node_length, "\tEdges: ", edge_length)
    if node_length > 600:
        gen = 50
        p_size = 20
        patient = 30
    else:
        gen = 60
        p_size = 30
        patient = 30
    ea = EA(
        population_size=p_size,
        generations=gen,
        mutation_rate=0.5,
        crossover_rate=0.5,
        special_rate=0.4,
        patient=patient,
        problem_instance=iem
    )
    sol, record = ea.evolve()
    print(sol)
    
    iem.save(SEED_B,sol)
    plt.plot(record)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.savefig('result_EA.jpg')