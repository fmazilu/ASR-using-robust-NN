import numpy as np
import matplotlib.pyplot as plt

def mixtgauss(N, p, sigma0, sigma1):
    """
    gives a mixtuare of gaussian noise
    arguments:
    N: data length
    p: probability of peaks
    sigma0: standard deviation of backgrond noise
    sigma1: standard deviation of impulse noise

    output: x: output noise
    """
    q = np.random.randn(N,1)
    # print(q)
    # print(q.shape)
    u = q < p
    # print(u)
    # print(sigma1 * u)
    # print(1-u)
    x = (sigma0 * (1 - u) + sigma1 * u) * np.random.randn(N, 1)
   
    return x


def add_noise(x, p, alpha):
    '''
    returns the signal with noise averaged by k

    arguments:
    x: input clean signal
    outputs:
    x_noisy: noisy signal

    '''
    N = x.shape[0]
    sigma0 = alpha
    sigma1 = 10 * alpha

    noise = mixtgauss(N, p, sigma0, sigma1)

    x_noisy = x + noise

    return x_noisy


# # plot the noise
# N = 100
# p = 0.01
# sigma0 = 1
# sigma1 = 10
#
# x = mixtgauss(N, p, sigma0, sigma1)
# print(x.shape)
#
# x2 = np.random.randn(N, 1)
# # print(x2)
# plt.plot(x2, 'r', x, 'g')
# plt.show()



