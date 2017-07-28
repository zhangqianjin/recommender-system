#!/usr/bin/python
# Created by zhangqianjin (2017)
# 这里的特征个数k是自己给出的，不能太小
# 参数变量(比如lamda,k等)可以使用网格搜索的方法筛选出最合适的参数值
# An implementation of matrix factorization adding bias

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
import numpy
def matrix_factorization(R, P, Q, K,b_user, b_item,all_mean, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j]) - b_user[i][0] - b_item[0][j] -all_mean
                    b_user[i][0] =  b_user[i][0] + alpha * (2 * eij  - beta * b_user[i][0])
                    b_item[0][j] =  b_item[0][j] + alpha * (2 * eij  - beta *b_item[0][j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j] - b_user[i][0] - b_item[0][j] -all_mean), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) + pow(b_item[0][j],2) + pow(b_user[i][0],2))
        if e < 0.001:
            break
    return P, Q.T


if __name__ == "__main__":
    R = [
         [5,4,4,3,5,0],
         [0,4,5,0,3,1],
         [5,4,0,1,3,0],
         [0,4,5,3,1,5],
         [1,0,3,5,0,5],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 3

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)
    b_user = numpy.random.rand(N,1)
    b_item = numpy.random.rand(1,M)
    all_mean = sum(sum(R))*1.0/23

    nP, nQ = matrix_factorization(R, P, Q, K, b_user, b_item, all_mean)
    print R
    T = numpy.dot(nP,nQ.T) + b_user + b_item + all_mean
    print T

