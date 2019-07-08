#coding:utf-8
#基于矩阵的方法,直接求矩阵的逆，最后的r0表示逆矩阵的行，例如，给我用户U推荐物品，从U对应的节点开始游走，概率结果为该节点对应的列
import numpy as np
from numpy.linalg import solve
import time
from scipy.sparse.linalg import gmres,lgmres
from scipy.sparse import csr_matrix
 
alpha=0.8
vertex=['A','B','C','a','b','c','d']
# M = np.array([[0,        0,        0,        0.5,      0,        0.333,    0],
#                    [0,        0,        0,        0.5,      1.0,      0.333,    0.5],
#                    [0,        0,        0,        0,        0,        0.333,    0.5],
#                    [0.5,      0.25,     0,        0,        0,        0,        0],
#                    [0,        0.25,     0,        0,        0,        0,        0],
#                    [0.5,      0.25,     0.5,      0,        0,        0,        0],
#                    [0,        0.25,     0.5,      0,        0,        0,        0]])

M=np.array([[0,        0,        0,        0.5,      0,        0.5,      0],
                   [0,        0,        0,        0.25,     0.25,     0.25,     0.25],
                   [0,        0,        0,        0,        0,        0.5,      0.5],
                   [0.5,      0.5,      0,        0,        0,        0,        0],
                   [0,        1.0,      0,        0,        0,        0,        0],
                   [0.333,    0.333,    0.333,    0,        0,        0,        0],
                   [0,        0.5,      0.5,      0,        0,        0,        0]])

r0=np.array([[0],[0],[0],[0],[0],[0],[0]])#从'A'开始游走

    #直接解线性方程法
A=np.eye(n)-alpha*M.T
b=np.eye(n)*(1-alpha)
begin=time.time()
r=solve(A,b)
end=time.time()
print('user time',end-begin)
rank={}
# for j in np.arange(n):
#     rank[vertex[j]]=r[j]
# for ele in vertex:
#     print(ele, rank[ele])
for i in range(r.shape[1]):
    print(vertex[i], r[:,i])

"""
user time 0.000179290771484375
A [0.3137876  0.16565576 0.07569236 0.15864619 0.03313115 0.18892314
 0.0634081 ]
B [0.08282788 0.3895794  0.08282788 0.11104703 0.07791588 0.14417818
 0.11104703]
C [0.07569236 0.16565576 0.3137876  0.0634081  0.03313115 0.18892314
 0.15864619]
a [0.15864619 0.22209406 0.0634081  0.30787729 0.04441881 0.13324053
 0.06978205]
b [0.0662623  0.31166352 0.0662623  0.08883763 0.2623327  0.11534255
 0.08883763]
c [0.12582281 0.19204534 0.12582281 0.08873819 0.03840907 0.33906732
 0.08873819]
d [0.0634081  0.22209406 0.15864619 0.06978205 0.04441881 0.13324053
 0.30787729]
 """
 
 
 """
 一般循环迭代的方法
 """
 def PersonalRank(G, alpha, root, max_step):
    rank = {index:0 for index in G.keys()}
    for i in range(max_step):
        for j in rank.keys():
            temp =0
            for k in G[j]:
                temp += alpha*rank[k]/len(G[k])
            rank[j] = temp
            if j==root:
                rank[j] += 1-alpha
    return rank
    

if __name__ == "__main__":
    G = {'A' : ['a','c'],
         'B' : ['a', 'b', 'c', 'd'],
         'C' : ['c', 'd'],
         'a' : ['A', 'B'],
         'b' : ['B'],
         'c' : ['A', 'B','C'],
         'd' : ['B', 'C']}
    
    rank = PersonalRank(G, 0.8, 'b', 1000)
    vertex=['A','B','C','a','b','c','d']
    result = []

    for index in vertex:
        result.append([index, rank[index]])
    result.sort(key=lambda x:x[1], reverse=True)
    print(result)
    """
    [['B', 0.3117744610281924], ['b', 0.26235489220563846], ['c', 0.11542288557213932], ['a', 0.0888888888888889], ['d', 0.0888888888888889], ['A', 0.06633499170812605], ['C', 0.06633499170812605]]
    """
