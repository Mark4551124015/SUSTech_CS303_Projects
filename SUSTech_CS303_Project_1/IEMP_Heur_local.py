import numpy as np

from networkx import DiGraph
import random as rand
from queue import Queue
import os
import time
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', type=str, help='test case number')
parser = parser.parse_args()
casenum=parser.c
NET_WORK='cases/Heuristic/map{}/dataset{}'.format(casenum,casenum)
SEED='cases/Heuristic/map{}/seed'.format(casenum)
SEED_B='cases/Heuristic/map{}/seed_balanced'.format(casenum)
OUTPUT='cases/Heuristic/map{}/score'.format(casenum)

K=5
N=5
cutoff_edge = 0.05

neib = {}

print('K',K," N",N," cutoff_edge",cutoff_edge)

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
def prob(p:float) -> bool:
    if rand.random() < p:
        return True
    return False

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
    return len(g.nodes) - (len(delta)), r1,r2,ac1,ac2

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
def solution(net_path,seed_path,seed_b_path,k):
    g,og,i1,i2 = init(net_path,seed_path)
    _,__, o1, o2 = init(net_path,seed_path)
    pool_1 = set(g.nodes).difference(i1)
    pool_2 = set(g.nodes).difference(i2)
    print("Nodes {}, Edges {}".format(len(set(g.nodes)), len(set(g.edges))))
    start=time.time()
    while k > 0:
        k-=1
        maxN = 1,1,0
        maxSET = None
        origin = getOrigin(g, i1, i2)
        for index,node in enumerate(pool_1):
            # tmp = [(sample(g,{node},{},preset) - preset[-1]) for preset in origin]
            score = []
            ac1 = []
            ac2 = []
            r1 = []
            r2 = []
            for preset in origin:
                ev, a1, a2, re1, re2 = sample(g,{node},{},preset)
                score.append(ev)
                ac1.append(a1)
                ac2.append(a2)
                r1.append(re1)
                r2.append(re2)
            score1 = np.average(score)
            if score1 > maxN[-1]: 
                maxN = (node, 'c1', score1)
                maxSET = (ac1, ac2, r1, r2)
        for index,node in enumerate(pool_2):
            # tmp = [sample(g,{},{node},preset) - preset[-1] for preset in origin]
            score = []
            ac1 = []
            ac2 = []
            r1 = []
            r2 = []
            for preset in origin:
                ev, a1, a2, re1, re2 = sample(g,{},{node},preset)
                score.append(ev)
                ac1.append(a1)
                ac2.append(a2)
                r1.append(re1)
                r2.append(re2)
            score2 = np.average(score)
            if score2 > maxN[-1]: 
                maxN = (node, 'c2', score2)
                maxSET = (ac1, ac2, r1, r2)
        node, ci, score = maxN
        if ci =='c1':
            i1.add(node)
            pool_1.remove(node)
        else:
            i2.add(node)
            pool_2.remove(node)

        (ac11, ac22, ex11, ex22) = maxSET
        for i in range(N):
            ex1,ac1,ex2,ac2,origin_score = origin[i]
            
            ex1 = ex1.union(ex11[i])
            ex2 = ex2.union(ex22[i])
            ac1 = ac1.union(ac11[i])
            ac2 = ac1.union(ac22[i])
            origin[i] = (ex1,ac1,ex2,ac2,score)

        print(maxN)
    end = time.time()
    print("Time Costed {:.2f} s".format(end-start))
    s1 = i1.difference(o1)
    s2 = i2.difference(o2)
    with open(seed_b_path, 'w') as f:
        f.write('%d %d\r\n' % (len(s1), len(s2)))
        for data in s1:
            f.write('%s\r\n' % data)
        for data in s2:
            f.write('%s\r\n' % data)


if __name__=='__main__':
    solution(NET_WORK,SEED,SEED_B,K)
    # g,og,i1,i2 = init(NET_WORK,SEED)
    # origin = getOrigin(g, i1, i2)
    # tmp = [ i[-1] for i in origin]
    # print('\n{}'.format(np.average(tmp)))

