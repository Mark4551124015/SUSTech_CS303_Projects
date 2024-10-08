import numpy as np
from networkx import DiGraph
# from queue import Queue
import random as rand
import argparse

class Queue:
    def __init__(self) -> None:
        self.g = []

    def get(self):
        return self.g.pop(0)

    def empty(self):
        return self.g.__len__() == 0
    
    def put(self,item):
        self.g.append(item)
        
parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', type=str, help='<social network>', default="")
parser.add_argument('-i', type=str, help='<initial seed set>', default="")
parser.add_argument('-b', type=str, help='<balanced seed set>', default="")
parser.add_argument('-k', type=str, help='<budget>', default=1)

parser = parser.parse_args()
NET_WORK = parser.n
SEED = parser.i
SEED_B = parser.b
K=int(parser.k)

# OUTPUT=parser.o

N=5
N_full=50
cutoff_edge = 0.05
cutoff_p = 0.05
neib = {}

def init(net_path,seed_path) -> tuple[DiGraph, DiGraph, set, set]:
    g = DiGraph()
    og = DiGraph()

    with open(net_path, 'r') as f:
        n, m = f.readline().strip().split(' ')
        for index, line in enumerate(f):
            if index >= int(n)+int(m):  break
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
            elif index < k1+k2:
                i2.add(node)
            else:
                break

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



def solution(net_path,seed_path,seed_b_path,k):
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
        if  ci == 'c1':
            i1.add(node)
            pool_1.remove(node)
        else:
            i2.add(node)
            pool_2.remove(node)
    s1 = i1.difference(o1)
    s2 = i2.difference(o2)
    with open(seed_b_path, 'w') as f:
        f.write('%d %d' % (len(s1), len(s2)))
        for data in s1:
            f.write('\n%s' % data)

        for data in s2:
            f.write('\n%s' % data)
        f.close()

if __name__=='__main__':
    # with open(SEED_B, 'w') as f:
    #         f.write(str(K-2) + ' ' + '2\n')
    #         for i in range(K):
    #             f.write(str(i+1) + '\n')
    #         f.close()

    solution(NET_WORK,SEED,SEED_B,K)
   