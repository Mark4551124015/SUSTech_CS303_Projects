import numpy as np
import networkx as nx
from networkx import DiGraph
import random as rand
from queue import Queue
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


casenum=2
NET_WORK='cases/Heuristic/map{}/dataset{}'.format(casenum,casenum)


def cut_range(step:float):
    cut = []
    label = []
    prob = 0.0
    while round(prob,4) <= 1.0:
        cut.append(round(prob,4))
        label.append(u"({:.3f},{:0.3f}])".format(round(prob,4), min(round(prob+step,4), 1.0)))
        prob+=step
    if label[-1] == '(1.000,1.000])':
        label = label[:-1]
    return cut, label

dt = []
g = DiGraph()
with open(NET_WORK, 'r') as f:
    n, m = f.readline().strip().split(' ')
    for line in f:
        u, v, c1, c2 = line.strip().split(' ')
        dt.append(float(c1))
        dt.append(float(c2))
df = pd.DataFrame(dt,columns=['p']).iloc[:,0]

cut, label = cut_range(0.05)

a = pd.cut(df, cut, labels=label)
b=a.value_counts().sort_index()
c={'section':b.index,'frequency':b.values}
e=pd.DataFrame(c)
print(b)
print(sum(b))


ax = plt.figure(figsize=(40, 20)).add_subplot(111)
sns.barplot(x="section",y="frequency",data=e,palette="Set3") #palette设置颜色
ax.set_xlabel('range', fontsize=25)
ax.set_ylabel('Freq', fontsize=25)
ax.set_title('Distribution', size=35)
plt.xticks(fontsize=25, rotation=30)
plt.yticks(fontsize=25)
for x, y in zip(range(len(label)), e.frequency):
    ax.text(x, y, '%d'%y, ha='center', va='bottom', fontsize=25, color='grey')

plt.savefig('Dist.jpg',dpi=500) 