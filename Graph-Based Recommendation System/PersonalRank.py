import numpy as np

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
