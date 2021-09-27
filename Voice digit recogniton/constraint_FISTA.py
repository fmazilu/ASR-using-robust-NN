import numpy as np
import time


def Constraint(w, Y, A, B, nit, rho):
    gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
    for i in range(nit):
        w_new = w - (np.transpose(A) @ Y @ np.transpose(B))
        w_new *= np.greater_equal(w_new, 0) # ensure non-negative weights
        T = A @ w_new @ B
        [u,s,v] = np.linalg.svd(T)
        criterion = np.linalg.norm(w_new - w, ord='fro')
        constraint = np.linalg.norm(s[s > rho] - rho, ord=2)
        print( 'iteration:', i+1, 'criterion: ', criterion, 'constraint: ', constraint)
        Yt = Y + gam * T
        [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
        s1 = np.clip(s1, 0, rho)
        Y = Yt - gam * np.dot(u1 * s1, v1)
        if (criterion < 30 and constraint < 0.1):
            print(i)
            return w_new
    return w_new

def Constraint_Fista(w, Y0, A, B, nit, rho):
    Y = Y0
    Yold = Y0
    gam = 1 / ((np.linalg.norm(A, ord=2) * np.linalg.norm(B, ord=2) + np.spacing(1)) ** 2)
    alpha = 2.1

    for i in range(nit):
        # eta = (i - 1) / (i + alpha)
        eta = i / (i + 1 + alpha)
        Z = Y + eta * (Y - Yold)
        Yold = Y
        w_new = w - A.T  @ Z @ B.T
        w_new *= np.greater_equal(w_new, 0)
        T = A @ w_new @ B
        [u,s,v] = np.linalg.svd(T)
        criterion = np.linalg.norm(w_new - w, ord='fro')
        constraint = np.linalg.norm(s[s > rho] - rho, ord=2)
        print( 'iteration:', i+1, 'criterion: ', criterion, 'constraint: ', constraint)
        Yt = Z + gam * T
        [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
        s1 = np.clip(s1, 0, rho)
        Y = Yt - gam * np.dot(u1 * s1, v1)
        if (criterion < 30 and constraint < 0.01):
            print(i)
            return w_new
    return w_new



p = 7
n = 128
m = 64
q = 64

gen_new_set = True 
if (gen_new_set):
    w = np.random.randn(m, n)
    A = np.eye(p, n) # np.random.randn
    B = np.eye(m, q)
    Y0 = np.zeros([p,q])
    nit = 100
    rho = 1  # L_cns to be ensured

    
tic = time.time()
w_new = Constraint(np.transpose(w), Y0, A, B, nit, rho)
tac = time.time()
print(tac - tic)
print()

tic = time.time()
w_new_Fiesta = Constraint_Fista(np.transpose(w), Y0, A, B, nit, rho)
tac = time.time()
print(tac - tic)
print()
print(np.allclose(w_new, w_new_Fiesta))



