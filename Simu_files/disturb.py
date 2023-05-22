import numpy as np


def disturbance(T, h, dim, flag):
    t = np.arange(0, T, h)     # discretising time
    t = np.reshape(t, (1, t.size))
    L = t.size   # Total time instance
    if(flag == 1):
        if (dim == 3):
            dis = (2 + \
                np.concatenate((2*np.sin(t), np.cos(t), 0.5 *
                               (np.cos(t)+np.sin(t))), axis=0))
    elif(flag == 2):
        dis = np.random.normal(3, 1, size=(dim, L))
    else:
        dis = np.zeros((dim, L))

    return dis

# def AWGN(snr,sp, dim):
#     std = np.power(np.divide(sp,snr),0.5)
#     dis = np.random.normal(2, 1, size=(dim, 1))