import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import Callback
import numpy as np


# Constraint callback class
class norm_constraint(Callback):
    def __init__(self, rho):
        self.rho = rho
        self.m = 0

    def on_train_begin(self, logs=None):
        m = 0
        for l in self.model.layers:
            if 'dense' in l.name:
                m = m + 1
        self.m = m

    def get_projection(self, w, rho):
        w = w * np.greater_equal(w, 0)
        norm = np.linalg.norm(w, ord=2)
        return w * np.power(rho, 1/self.m) /(norm + np.spacing(1))

    def on_batch_end(self, batch, logs=None):
        for l in self.model.layers:
            if 'dense' in l.name:
                w = l.get_weights()[0]
                b = l.get_weights()[1]
                w_new = self.get_projection(w, self.rho)
                l.set_weights([w_new, b])


# Constraint class
class customConstraint(Constraint):  #### De vazut API
    def __init__(self, rho):
        super(Constraint, self).__init__()
        self.rho = rho

    def __call__(self, w):
        w = w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
        norm = tf.norm(w, ord=2)
        return w * self.rho / (norm + np.spacing(1))

    def get_config(self):
        return {'rho': self.rho}


# Constraint callback class
class norm_constraint_FISTA(Callback):
    def __init__(self, rho, nit):
        super(Callback, self).__init__()
        self.rho = rho
        self.m = 0
        self.nit = nit


    def get_w_list(self):
        w_list = []
        for l in self.model.layers:
            if 'dense' in l.name:
                w = l.get_weights()[0]
                w_list.append(w)
        return w_list


    def Constraint_Fista(self, w, Y0, A, B, nit, rho):
        Y = Y0
        Yold = Y0
        gam = 1 / ((np.linalg.norm(A, ord=2) * np.linalg.norm(B, ord=2) + np.spacing(1)) ** 2)
        alpha = 2.1

        for i in range(nit):
            # eta = (i - 1) / (i + alpha)
            eta = i / (i + 1 + alpha)
            Z = Y + eta * (Y - Yold)
            Yold = Y
            w_new = w - A.T @ Z @ B.T
            w_new *= np.greater_equal(w_new, 0)
            T = A @ w_new @ B
            [u, s, v] = np.linalg.svd(T)
            criterion = np.linalg.norm(w_new - w, ord='fro')
            constraint = np.linalg.norm(s[s > rho] - rho, ord=2)
            # print('iteration:', i + 1, 'criterion: ', criterion, 'constraint: ', constraint)
            Yt = Z + gam * T
            [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
            s1 = np.clip(s1, 0, rho)
            Y = Yt - gam * np.dot(u1 * s1, v1)
            if (criterion < 30 and constraint < 0.01):
                # print(i)
                return w_new
        return w_new

    def get_projection(self, w):
        A = []
        B = []
        w_list = self.get_w_list()
        for index in reversed(range(len(w_list))):
            if np.array_equal(w, w_list[index]):
                w_index = index
        for index in reversed(range(len(w_list))):
            if index > w_index:
                if A == []:
                    A = np.array(w_list[index]).transpose()
                else:
                    A = np.matmul(A, np.array(w_list[index]).transpose())
            elif index < w_index:
                if B == []:
                    B = np.array(w_list[index]).transpose()
                else:
                    B = np.matmul(B, np.array(w_list[index]).transpose())
            if w_index == 0:
                B = np.eye(np.array(w).shape[0], np.array(w).shape[0])
            if w_index == (len(w_list)-1):
                A = np.eye(np.array(w).shape[1], np.array(w).shape[1])
        A = np.array(A)
        B = np.array(B)
        Y0 = np.zeros([A.shape[0], B.shape[1]])
        w_new = self.Constraint_Fista(w.T, Y0, A, B, self.nit, self.rho)
        return w_new

    def on_batch_end(self, batch, logs=None):
        for l in self.model.layers:
            if 'dense' in l.name:
                w = l.get_weights()[0]
                b = l.get_weights()[1]
                w_new = self.get_projection(w)
                l.set_weights([w_new.T, b])


class simple_norm_constraint(Callback):
    def __init__(self, rho):
        super(Callback, self).__init__()
        self.rho = rho
        self.m = 0

    def get_w_list(self):
        w_list = []
        for l in self.model.layers:
            if 'dense' in l.name:
                w = l.get_weights()[0]
                # print(f"w inainte de modificare {w}")
                w_list.append(w)
        return w_list

    def get_projection(self, w):
        w_list = self.get_w_list()
        cst = []
        for index in reversed(range(len(w_list))):
            if cst == []:
                cst = np.array(w_list[index]).transpose()
                # print(f"cst: {cst}")
            else:
                cst = np.matmul(cst, np.array(w_list[index]).transpose())
        # print(f"cst {np.linalg.norm(cst, ord=2)}")
        w_new = w * np.greater_equal(w, 0)
        w_new = w_new * np.power((self.rho / (np.linalg.norm(cst, ord=2) + np.spacing(1))), 1/len(w_list))
        # print(f"no of layers {len(w_list)}")

        return w_new

    def on_batch_end(self, batch, logs=None):
        for l in self.model.layers:
            if 'dense' in l.name:
                w = l.get_weights()[0]
                b = l.get_weights()[1]
                w_new = self.get_projection(w)
                l.set_weights([w_new, b])
