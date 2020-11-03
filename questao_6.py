import numpy as np
from numpy.linalg import matrix_power
from operator import add

def page_rank_1(x0,A,alpha,n_iter):
    shape = len(x0)
    beta_v = shape*[1/shape]
    inc = np.array(x0).dot(matrix_power(A,n_iter))
    inc_l = inc.tolist()
    xt = [i*(alpha**(n_iter)) for i in inc_l]
    for i in range(n_iter-1,0,-1):
        inc = np.array(beta_v).dot(matrix_power(A,i))
        inc_l = inc.tolist()
        xt = list( map(add, xt, [j*(alpha**(i)) for j in inc_l]))
    xt = list( map(add, xt, [i*alpha for i in beta_v]))
    return xt

A = np.array([[0, 0, 0.5, 0.5, 0] \
        , [0.5, 0, 0, 0, 0.5] \
        , [0, 0, 0, 1, 0] \
        , [0, 0.5, 0, 0, 0.5] \
        , [0, 0, 0, 1, 0]])

alpha = [0.1, 0.9]
n_iter = [5, 10, 30]
x0 = 5*[0.2]

print("Transition matrix: ")
print(A)
print('\n')
print("Initial vector: ")
print(x0)

for i in range(len(alpha)):
    for j in range(len(n_iter)):
        xt = page_rank_1(x0,A,alpha[i],n_iter[j])
        print("Page rank solution for alpha = {} and {} iterations: {}".format(alpha[i],n_iter[j],xt))
