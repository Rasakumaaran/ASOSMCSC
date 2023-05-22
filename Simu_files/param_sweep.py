
def Kbar_fn(_, x, Kbar_param):
    eta = Kbar_param.get("eta")
    sigNom = Kbar_param.get("sNom")
    fNom = Kbar_param.get("fNom")
    rho = Kbar_param.get("rho")
    Kbar = eta*sigNom*fNom - rho*eta*x
    return Kbar


def Theta_fn(_, x, Kai_param):
    alpha = Kai_param.get("alp")
    U = Kai_param.get("U")
    Kai = np.zeros((2, 3))
    #print('\n H:\n', H, '\n U:\n', U, '\n tdd:\n',tau0_ddot, '\n VecProd:\n', np.dot(g, alpha))
    Kai[0, :] = x[1, :]
    temp =  alpha + U 
    Kai[1, :] = temp.T
    return Kai

def ctrlr(gains):
    h = 1/400     # sampling rate(in Sec)
    T = 30    # Total time
    # J = np.diag([0.008, 0.009, 0.017])  # Polar MOI in body frame -----------Tsalla
    J = np.diag([0.029125,0.029125,0.055225])  # Polar MOI in body frame ------- Iris
    dim = 3       # system dimension
    dis = disturbance(T, h, dim, flag=1)

    t = np.arange(0, T, h)     # discretising time
    t = np.reshape(t, (1, t.size))
    L = t.size   # Total time instance
    
    ang = np.zeros((dim, L+1))  # Euler angles (x1)
    ang_dot = np.zeros((dim, L+1))  # ang. Vel. in world frame (x2)
    omega = np.zeros((dim, L))  # ang. Vel. in body frame
    err1 = np.zeros((dim, L))  # error angles
    mse = np.zeros((dim, L))  # mean Sq. error of angles 
    err2 = np.zeros((dim, L))  # error ang_vel
    err_intg = np.zeros((dim, L))  # integration of kai in slide variable
    sigma = np.zeros((dim, L))  # sliding variable
    Kbar = np.zeros((L+1, 1))  # adap gain
    U = np.zeros((dim, L))
    torq = np.zeros((dim, L))

    # temp = np.array([0.35, -0.25, -0.1]).T  # comment while using ROS
    temp = np.array([0.0 , 0.0, 0.0]).T 
    ang[:, 0] = temp

    radii = 0.2
    ang_ref = np.concatenate(
        (radii*np.sin(t), radii*np.cos(t), radii*np.ones((1, L))), axis=0)
    ang_dot_ref = np.concatenate(
        (radii*np.cos(t), -radii*np.sin(t), np.zeros((1, L))), axis=0)
    ang_ddot_ref = np.concatenate(
        (-radii*np.sin(t), -radii*np.cos(t), np.zeros((1, L))), axis=0)
    
    om1 = gains["om1"]*np.eye(dim)/10
    om2 = gains["om2"]*om1/10
    eps = gains["eps"]/10
    eta = gains["eta"]/10

    rho = 0.001
    Kbar[0] = 0.1
    Gama = 1.0*om1
    om3 = 1.0*np.eye(dim)
    
    for i in np.arange(0, L, 1):
        err1[:, i] = ang[:, i] - ang_ref[:, i]
        err2[:, i] = ang_dot[:, i] - ang_dot_ref[:, i]
        
        # publish to /setpoint_attitude

        R = [[1, np.sin(ang[0, i])*np.tan(ang[1, i]), np.cos(ang[0, i])*np.tan(ang[1, i])],
            [0, np.cos(ang[0, i]), -np.sin(ang[0, i])],
            [0, np.sin(ang[0, i])/np.cos(ang[1, i]), np.cos(ang[0, i])/np.cos(ang[1, i])]]
        R = np.array(R)

        if(i == 0):
            Rold = R
        R_dot = (R-Rold)/h
        Rold = R

        beta = np.dot(R, np.linalg.inv(J))

        omega[:, i] = np.dot(np.linalg.inv(R), ang_dot[:, i])

        # computing Sliding variable
        # trapezoidal integration
        if (i == 0):
            err_intg[:, i] = h*err1[:, i]/2
        elif (i == 1):
            err_intg[:, i] = h*(err1[:, i] + err1[:, i-1])/2
        else:
            err_intg[:, i] = (err1[:, i] + 2/h*err_intg[:, i-1] + err1[:, i-1])*h/2

        sigma[:, i] = np.dot(om1, err1[:, i]) + \
            np.dot(om2, err_intg[:, i]) + np.dot(om3, err2[:, i])
        sigNom = np.linalg.norm(sigma[:, i])

        H = np.zeros((dim, ))
        # H = np.multiply(np.tanh(tau1[:, i]/2), np.power((tau2[:, i]), 2)) 
        
        temp = (np.dot(-om1, err2[:, i]) - np.dot(om2, err1[:, i]) - H)
        # print(f" \nerr1[] :  {err1[:, i]}  \nerr2[] :  {err2[:, i]}")
        temp = np.reshape(temp, (temp.size, 1))
        # print(f"\n Temp :  {temp} \n Shape : {temp.shape}")   
        f = np.concatenate((np.eye(3), temp),
                        axis=1)  # try adding seperately
        fNom = np.linalg.norm(f)
        # adaptive gain
        Kbar_param = {"eta": eta, "sNom": sigNom, "fNom": fNom, "rho": rho}
        #######
        Kbar[i+1] = RK4(Kbar_fn, 0, Kbar[i], h, Kbar_param)
        ########

        if (sigNom < eps):
            sat = sigma[:, i]/eps
        else:
            sat = sigma[:, i]/sigNom

        ##########  actually this U is g*inverse(lambda)*v
        U[:, i] = -(np.dot(Gama, sigma[:, i]) + Kbar[i]*fNom*sat +
                    np.dot(om1, err2[:, i]) + np.dot(om2, err1[:, i]) + H)
        
        # print(f"\n omg : {omega}, \n U : {U}")

        alpha = np.dot(beta, (np.cross(-omega[:, i], np.dot(J, omega[:, i])) +
                    dis[:, i])) + np.dot(R_dot, omega[:, i]) 

        err_param = {"alp": alpha,  "U": U[:, i]}
        ###########
        Y = RK4(Theta_fn, 0, np.array([ang[:, i], ang_dot[:, i]]), h, err_param)
        ##########
        ang[:, i+1] = Y[0, :].T
        ang_dot[:, i+1] = Y[1, :].T

        # if(np.linalg.norm(err1)>10**2):
        #     return 0
        # elif(i<L):
        #     continue
        # else:
        #     return 1
    mse = np.power(err1,2)
    return (np.mean(mse[0,:])+ np.mean(mse[1,:])+ np.mean(mse[2,:]))

        
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from solveODE import RK4
    from disturb import disturbance

    import psweep as ps
    '''om1=4.0, om2=5.0, om3=1, eps=0.5, eta=2, gama=1 Iris without bounds'''
    i1 = ps.plist("om1",np.arange(39,42,1, dtype=np.int32).tolist())  
    i2 = ps.plist("om2",np.arange(49,52,1, dtype=np.int32).tolist()) 
    i3 = ps.plist("eps",np.arange( 2, 7,1, dtype=np.int32).tolist())   
    i4 = ps.plist("eta",np.arange(19,22,1, dtype=np.int32).tolist()) 

    gains = ps.pgrid(i1,i2,i3,i4)
    df = ps.run_local(ctrlr, gains)

    print(df)
       
