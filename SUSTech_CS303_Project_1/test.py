import numpy as np
from networkx import DiGraph
import random as rand
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, help="The absolute path of the social network file")
    parser.add_argument("-i", type=str, help="The absolute path of the two campaigns' initial seed set")
    parser.add_argument("-b", type=str, help="The absolute path of the two campaigns' balanced seed set")
    parser.add_argument("-k", type=int, help="The positive integer budget")
    args = parser.parse_args()
    n = args.n
    i = args.i
    b = args.b
    k = args.k
    
    with open(b, 'w') as f:
        f.write(str(k-2) + ' ' + str(2)+'\n')
        for i in range(k):
            f.write(str(i+1) + '\n')
