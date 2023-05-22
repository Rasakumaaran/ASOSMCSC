import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    f = lambda x : 0.5*np.cos(x)
    A = []
    for i in range(40):
        a = np.rad2deg(f(i))
        A.append(a)
        print("i",i)
    plt.plot(A)
    plt.show()