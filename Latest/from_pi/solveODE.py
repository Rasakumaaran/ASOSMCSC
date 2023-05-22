def RK4(func, t, y_0, h, u):
    # k1 = func(t, y_0, u)
    # k2 = func(t+h/2, y_0+(h/2)*k1, u)
    # k3 = func(t+h/2, y_0+(h/2)*k2, u)
    # k4 = func(t+h, y_0+h*k3, u)
    # y = y_0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    k1 = h*func(t, y_0, u)
    k2 = h*func(t+h/2, y_0+(h/2)*k1, u)
    k3 = h*func(t+h/2, y_0+(h/2)*k2, u)
    k4 = h*func(t+h, y_0+h*k3, u)
    y = y_0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

    # print(f'[RK4] out: {y}')
    return y
