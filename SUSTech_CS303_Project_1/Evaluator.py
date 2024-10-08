import numpy as np

from networkx import DiGraph
import random as rand
from queue import Queue
import os
import time
import argparse

# casenum=3
# NET_WORK='cases/Heuristic/map{}/dataset{}'.format(casenum,casenum)
# SEED='cases/Heuristic/map{}/seed'.format(casenum)
# SEED_B='cases/Heuristic/map{}/seed_balanced'.format(casenum)
# K=5
N=15

parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', type=str, help='<social network>')
parser.add_argument('-i', type=str, help='<initial seed set>')
parser.add_argument('-b', type=str, help='<balanced seed set>')
parser.add_argument('-k', type=str, help='<budget>')
parser.add_argument('-o', type=str, help='<object value output path>')
parser = parser.parse_args()

NET_WORK = parser.n
SEED = parser.i
SEED_B = parser.b
K=parser.k
OUTPUT=parser.o

# neib = {}
def prob_full(p:float) -> bool:
    if rand.random() < p:
        return True
    return False

def prop_full(g:DiGraph,si:set,ci) -> set:
    queue = Queue()
    exposed = set()
    exposed = exposed.union(si)
    activated = set()
    activated = activated.union(si)
    for i in si:
        queue.put(i)
    while not queue.empty():
        node = queue.get()
        if node not in set(g.nodes):
            exposed.add(node)
            continue
        for v in g[node].keys():
            c = g[node][v][ci]
            exposed.add(v)
            if prob_full(float(c)) and (v not in activated):
                queue.put(v)
                activated.add(v)
    return exposed

def sample_full(g:DiGraph,s1:set,s2:set):
    tmp = []
    for i in range(N):
        r1 = prop_full(g,s1,'c1')
        r2 = prop_full(g,s2,'c2')
        delta = r1.difference(r2).union(r2.difference(r1)) 
        tmp.append(len(delta))
    avg = np.average(tmp)
    return len(g.nodes) - avg

def init(net_path,seed_path,seed_b_path) -> tuple[DiGraph, DiGraph, set, set]:
    og = DiGraph()
    with open(net_path, 'r') as f:
        n, m = f.readline().strip().split(' ')
        for line in f:
            u, v, c1, c2 = line.strip().split(' ')
            add_edge(og, u,v,c1,c2)
    i1=set()
    i2=set()

    with open(seed_path, 'r') as f:
        k1, k2 = f.readline().strip().split(' ')
        k1, k2 = int(k1), int(k2)
        for index, line in enumerate(f):
            node = line.strip()
            if index < k1:
                i1.add(node)
            else:
                i2.add(node)
    with open(seed_b_path, 'r') as f:
        k1, k2 = f.readline().strip().split(' ')
        k1, k2 = int(k1), int(k2)
        for index, line in enumerate(f):
            node = line.strip()
            if index < k1:
                i1.add(node)
            else:
                i2.add(node)
    return og, i1, i2
def add_edge(OG: DiGraph, u, v, c1, c2):
    OG.add_node(u)
    OG.add_node(v)
    OG.add_edge(u, v, c1=c1,c2=c2)

def save(PATH, obj):
    with open(PATH, 'w') as f:
        f.write(obj)

def solution(net_path,seed_path,seed_b_path,k):
    og,i1,i2 = init(net_path,seed_path, seed_b_path)
    start=time.time()
    Score = sample_full(og, i1, i2)
    print("before balance", Score)
    end = time.time()
    print("Time Costed {:.2f} s".format(end-start))
    save(OUTPUT, str(Score))
    
if __name__=='__main__':
    solution(NET_WORK,SEED,SEED_B,K)

