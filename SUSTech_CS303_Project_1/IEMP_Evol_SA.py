import numpy as np

from networkx import DiGraph
import random as rand
from queue import Queue
import os
import time
import matplotlib.pyplot as plt
import argparse

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
        iem = cls(g, og, i1, i2, set({}), set({}), neib, sample_N, K)
        return iem
    
    def prop(self, g:DiGraph, ci) -> tuple[set,set]:
        queue = Queue()
        new_nodes=set({})
        match ci:
            case 'c1':
                new_nodes = self.s1.union(self.i1)
            case 'c2':
                new_nodes = self.s2.union(self.i2)
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
    
    def __init__(self, g: DiGraph, og: DiGraph, i1:set, i2:set, s1:set, s2:set, neib:dict, sample_N, K:int) -> None:
        self.g:DiGraph = g
        self.og:DiGraph = og
        self.i1,self.i2 = set(),set()
        self.s1,self.s2 = set(),set()
        self.i1 = self.i1.union(i1)
        self.i2 = self.i2.union(i2)
        self.s1 = self.s1.union(s1)
        self.s2 = self.s2.union(s2)
        self.neib = neib
        self.N = sample_N
        self.pool_1 = set(g.nodes).difference(i1).difference(s1)
        self.pool_2 = set(g.nodes).difference(i2).difference(s2)
        self.K = K


    def init_solution(self):
        node = rand.choice(list(self.pool_1))
        self.pool_1.remove(node)
        self.s1.add(node)
        node = rand.choice(list(self.pool_2))
        self.pool_2.remove(node)
        self.s2.add(node)
        for i in range(self.K-2):
            if prob(0.5):
                node = rand.choice(list(self.pool_1))
                self.pool_1.remove(node)
                self.s1.add(node)
            else:
                node = rand.choice(list(self.pool_2))
                self.pool_2.remove(node)
                self.s2.add(node)
        iem=  IEM(self.g, self.og, self.i1, self.i2, 
                    self.s1,self.s2,self.neib, self.N,self.K)
        print(iem.s1, iem.s2)
        return iem

    def value(self) -> int:
        tmp = []
        for i in range(self.N):
            r1,ac1 = self.prop(self.g,'c1')
            r2,ac2 = self.prop(self.g,'c2')
            delta = r1.difference(r2).union(r2.difference(r1)) 
            tmp.append(len(self.g.nodes) - len(delta))
        return np.average(tmp)
    
    def get_len(self):
        return len(list(self.s1)) + len(list(self.s2))
     
    def local_search(self, special):
        newS1, newS2 = set(), set()
        newS1 = newS1.union(self.s1)
        newS2 = newS2.union(self.s2)
        iem = IEM(self.g, self.og, self.i1, self.i2, 
                    self.s1,self.s2,self.neib, self.N,self.K)
        print(iem.s1, iem.s2)
        
        for i in range(rand.randint(1,int(iem.K))):
            
            new_node1 = rand.choice(list(iem.pool_1))
            old_node1 = rand.choice(list(iem.s1))
            if prob(0.5):
                if prob(1-special):
                    newS1 = newS1.difference({old_node1})
                    newS1 = newS1.union({new_node1})
                    iem.pool_1 = iem.pool_1.difference({new_node1})
                    iem.pool_2 = iem.pool_2.union({old_node1})
                else:
                    if iem.get_len() < iem.K:
                        newS1 = newS1.union({new_node1})
                        iem.pool_1 = iem.pool_1.difference(new_node1)
                        new_node1 = rand.choice(list(iem.pool_1))
                        old_node1 = rand.choice(list(iem.s1))
                        newS1 = newS1.difference({old_node1})
                        newS1 = newS1.union({new_node1})
                        iem.pool_1 = iem.pool_1.difference({new_node1})
                        iem.pool_2 = iem.pool_2.union({old_node1})
            else:
                if prob(1-special):
                    newS1 = newS1.difference({old_node1})
                    newS1 = newS1.union({new_node1})
                    iem.pool_1 = iem.pool_1.difference(new_node1)
                    iem.pool_2 = iem.pool_2.union({old_node1})
                else:
                    if len(list(iem.s1)) > 1:
                        newS1 = newS1.difference({old_node1})
                        iem.pool_1 = iem.pool_1.union({new_node1})
                        new_node1 = rand.choice(list(iem.pool_1))
                        old_node1 = rand.choice(list(iem.s1))
                        newS1 = newS1.difference({old_node1})
                        newS1 = newS1.union({new_node1})
                        iem.pool_1 = iem.pool_1.difference({new_node1})
                        iem.pool_2 = iem.pool_2.union({old_node1})
            iem = IEM(iem.g, iem.og, iem.i1, iem.i2, 
                    newS1,iem.s2,iem.neib, iem.N,iem.K)
            
        for i in range(rand.randint(1,int(iem.K))):
            new_node2 = rand.choice(list(iem.pool_2))
            old_node2 = rand.choice(list(iem.s2))
            if prob(0.5):
                if prob(1-special):
                    newS2 = newS2.difference({old_node2})
                    newS2 = newS2.union({new_node2})
                    iem.pool_2 = iem.pool_2.difference({new_node2})
                    iem.pool_2 = iem.pool_2.union({old_node2})
                else:
                    if iem.get_len() < iem.K:
                        newS2 = newS2.union({new_node2})
                        iem.pool_2 = iem.pool_2.difference(new_node2)
                        new_node2 = rand.choice(list(iem.pool_2))
                        old_node2 = rand.choice(list(iem.s2))
                        newS2 = newS2.difference({old_node2})
                        newS2 = newS2.union({new_node2})
                        iem.pool_2 = iem.pool_2.difference({new_node2})
                        iem.pool_2 = iem.pool_2.union({old_node2})
            else:
                if prob(1-special):
                    newS2 = newS2.difference({old_node2})
                    newS2 = newS2.union({new_node2})
                    iem.pool_2 = iem.pool_2.difference(new_node2)
                    iem.pool_2 = iem.pool_2.union({old_node2})
                else:
                    if len(list(iem.s2)) > 1:
                        newS2 = newS2.difference({old_node2})
                        iem.pool_2 = iem.pool_2.union({new_node2})
                        new_node2 = rand.choice(list(iem.pool_2))
                        old_node2 = rand.choice(list(iem.s2))
                        newS2 = newS2.difference({old_node2})
                        newS2 = newS2.union({new_node2})
                        iem.pool_2 = iem.pool_2.difference({new_node2})
                        iem.pool_2 = iem.pool_2.union({old_node2})
            iem = IEM(iem.g, iem.og, iem.i1, iem.i2, 
                    iem.s1,newS2,iem.neib, iem.N,iem.K)
            
        if prob(0.5):
            node1 = rand.choice(list(iem.s1))
            node2 = rand.choice(list(iem.s2))
            iem.pool_2 = iem.pool_2.difference({node2})
            iem.pool_2 = iem.pool_2.union({node1})
            iem.pool_1 = iem.pool_2.difference({node1})
            iem.pool_1 = iem.pool_2.union({node2})
            iem.s1 = iem.s1.union({node2}).difference({node1})
            iem.s2 = iem.s2.union({node1}).difference({node2})
        return iem
    

    def save(self, seed_b_path):
        with open(seed_b_path, 'w') as f:
            f.write('%d %d\r\n' % (len(self.s1), len(self.s2)))

            for data in self.s1:
                f.write('%s\r\n' % data)
            for data in self.s2:
                f.write('%s\r\n' % data)

def SA(initial:IEM, schedule, halt, log_interval=1, patient=20, spacial=0.1, ratio=0.1):
    state = initial.init_solution()
    t = 1
    T = schedule(t)
    f = []
    old_value = state.value()
    cnt = 0
    while not halt(T):
        now_value = old_value
        new_state = state.local_search(spacial)
        new_value = new_state.value()
        interest = np.exp(-(old_value - new_value)/T*ratio)
        interest = min(interest, 0.5)
        good = new_value >= old_value
        bad_but = (not good) and prob(interest)
        if good or bad_but:
            state = new_state
            old_value = new_value
            now_value = new_value
            cnt = 0
        else:
            cnt+=1
        f.append(now_value)
        if t % log_interval == 0:
            if bad_but:
                print(f"step {t}: T={T}, current_value={now_value}")
            else:
                print(f"step {t}: T={T}, current_value={now_value}")
        t += 1
        T = schedule(T)
        if cnt >= patient:
            break
    return state, f


if __name__=='__main__':
    iem = IEM.create(NET_WORK, SEED, cut_off=0.05, sample_N=15, K=5)
    solution, record = SA(
        initial= iem,
        schedule=lambda t: 0.995*t, 
        halt=lambda T: T<1e-4,
        ratio = 0.1,
        patient=500,
        spacial=0.2
    )
    solution.save(SEED_B)
    plt.plot(record)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.savefig('result_SA.jpg')