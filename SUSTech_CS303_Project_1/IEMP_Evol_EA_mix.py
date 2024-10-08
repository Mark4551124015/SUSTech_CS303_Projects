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
K=5
neib={}
cutoff_edge = 0.05
N=15
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
        if ci == 'ci':
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
        s1,s2 = set({}), set({})
        node = rand.choice(list(pool_1))
        pool_1.remove(node)
        s1.add(node)
        node = rand.choice(list(pool_2))
        pool_2.remove(node)
        s2.add(node)
        for i in range(self.K-2):
            if prob(0.5):
                node = rand.choice(list(pool_1))
                pool_1.remove(node)
                s1.add(node)
            else:
                node = rand.choice(list(pool_2))
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
    def __init__(self, population_size:int, generations:int, mutation_rate:float, crossover_rate:float, 
                 special_rate:float, patient:int, problem_instance:IEM, init_pop:list):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.problem_instance = problem_instance
        self.special_rate = special_rate
        self.patient = patient
        self.init_pop = init_pop

    def initialize_population(self):
        population = self.init_pop
        for _ in range(self.population_size - len(population)):
            individual = self.problem_instance.init_solution()
            population.append(individual)
        return population

    def select_parents(self, population):
        # 这里可以使用适应度函数选择父代个体
        info = [individual for individual in population]
        info.sort(key=lambda x: x[-1])
        fit = [i[-1] for i in info]
        min_fit = min(fit)
        fit = [(i - min_fit)*5 for i in fit]
        # Create roulette wheel.
        sum_fit = sum(fit)
        wheel = list(accumulate([i/sum_fit for i in fit]))
        father_idx = bisect_right(wheel, rand.random())
        father = info[father_idx]
        mother_idx = (father_idx + 1) % len(wheel)
        mother = info[mother_idx]
        return father, mother
    
    def crossover(self, parent1, parent2):
        # print(parent1)
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
        return (son1, self.value(son1)), (son2, self.value(son1))
    def value(self, indiv):
        return self.problem_instance.value(indiv)
    
    def mutate(self, individual):
        (s1,s2), score = copy.deepcopy(individual)
        s1_, s2_ = set(), set()
        s1_ = s1_.union(s1)
        s2_ = s2_.union(s2)
        pool_1, pool_2 = self.problem_instance.get_pool(s1,s2)

        for i in range(self.problem_instance.K // 2):
            if len(s1_) < 1:
                nodeA = ''
            else:
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
                    if nodeA != '' and nodeB != '':
                        s1_.add(nodeB)
                        pool_1.remove(nodeB)
                        s1_.remove(nodeA)
                        pool_1.add(nodeA)
            if len(s2_) < 1:
                nodeA = ''
            else:
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
                    if nodeA != '' and nodeB != '':
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

def init(net_path,seed_path) -> tuple[DiGraph, DiGraph, set, set]:
    g = DiGraph()
    og = DiGraph()

    with open(net_path, 'r') as f:
        n, m = f.readline().strip().split(' ')
        for line in f:
            u, v, c1, c2 = line.strip().split(' ')
            add_edge(g, og, u,v,c1,c2)
    i1=set()
    i2=set()
    with open(seed_path, 'r') as f:
        k1, k2 = f.readline().strip().split(' ')
        k1, k2 = int(k1), int(k2)
        for index, line in enumerate(f):
            node = line.strip()
            g.add_node(node)
            if index < k1:
                i1.add(node)
            else:
                i2.add(node)
    return g, og, i1, i2
def add_edge(G:DiGraph, OG: DiGraph, u, v, c1, c2):
    neib.setdefault(u, None)
    if neib[u] is None:
        neib[u]={v}
    else:
        neib[u].add(v)

    OG.add_node(u)
    OG.add_node(v)
    OG.add_edge(u, v, c1=c1,c2=c2)
    if float(c1) < cutoff_edge and float(c2) < cutoff_edge:
        return
    G.add_edge(u, v, c1=c1,c2=c2)

def prop(g:DiGraph,si:set,ci, exposed:set, activated:set) -> tuple[set,set]:
    queue = Queue()
    exposed = exposed.union(si)
    activated = activated.union(si)
    for i in si:
        queue.put(i)
    while not queue.empty():
        node = queue.get()
        neib.setdefault(node, set())
        exposed = exposed.union(neib[node])
        for v in g[node].keys():
            c = g[node][v][ci]

            if prob(float(c)) and (v not in activated):
                queue.put(v)
                activated.add(v)
    return exposed, activated
def sample(g:DiGraph,s1:set(),s2:set(), preset:tuple[set,set,set,set,int]):
    ex1,ac1,ex2,ac2,score = preset
    r1,ac1 = prop(g,s1,'c1',ex1,ac1)
    r2,ac2 = prop(g,s2,'c2',ex2,ac2)
    delta = r1.difference(r2).union(r2.difference(r1)) 
    return len(g.nodes) - (len(delta))
def getOrigin(g:DiGraph,s1:set,s2:set) -> list[tuple[set,set,set,set,int]]:
    ret = []
    for i in range(N):
        ac = set()
        ex = set()
        ex1,ac1 = prop(g,s1,'c1',ac,ex)
        ac = set()
        ex = set()
        ex2,ac2 = prop(g,s2,'c2',ac,ex)
        delta = ex1.difference(ex2).union(ex2.difference(ex1)) 
        score = len(g.nodes) - (len(delta))
        ret.append((ex1,ac1,ex2,ac2,score))
    return ret
def greedy(net_path,seed_path,seed_b_path,k):
    global N
    tmpN = N
    N=5
    g,og,i1,i2 = init(net_path,seed_path)
    _,__, o1, o2 = init(net_path,seed_path)
    pool_1 = set(g.nodes).difference(i1)
    pool_2 = set(g.nodes).difference(i2)
    while k > 0:
        k-=1
        maxN = 1,1,0
        origin = getOrigin(g, i1, i2)
        for index,node in enumerate(pool_1):
            tmp = [(sample(g,{node},{},preset) - preset[-1]) for preset in origin]
            score1 = np.average(tmp)
            if score1 > maxN[-1]: maxN = (node, 'c1', score1)
        for index,node in enumerate(pool_2):
            tmp = [sample(g,{},{node},preset) - preset[-1] for preset in origin]
            score2 = np.average(tmp)
            if score2 > maxN[-1]: maxN = (node, 'c2', score2)
        node, ci, score = maxN
        
        if ci == 'c1':
            i1.add(node)
            pool_1.remove(node)
        else:
            i2.add(node)
            pool_2.remove(node)
        print(maxN)
    s1 = i1.difference(o1)
    s2 = i2.difference(o2)
    N=tmpN
    return s1, s2

if __name__=='__main__':
    iem = IEM.create(NET_WORK, SEED, cut_off=0.05, sample_N=12, K=K)
    node_length = len(set(iem.g.nodes))
    edge_length = len(set(iem.g.edges))
    print("Nodes: ",node_length, "\tEdges: ", edge_length)
    if node_length > 600:
        gen = 100
        p_size = 5
        patient = 30
        N = 5
    else:
        gen = 100
        p_size = 10
        patient = 20
    init_pop = greedy(NET_WORK,SEED,SEED_B,K)
    ea = EA(
        population_size=p_size,
        generations=gen,
        mutation_rate=0.5,
        crossover_rate=0.5,
        special_rate=0.0,
        patient=patient,
        problem_instance=iem,
        init_pop=[(init_pop, iem.value(init_pop))]*1
    )
    sol, record = ea.evolve()
    print(sol)
    
    iem.save(SEED_B,sol)
    plt.plot(record)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.savefig('result_EA.jpg')