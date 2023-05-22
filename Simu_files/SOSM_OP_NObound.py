 
import numpy as np
import matplotlib.pyplot as plt
from solveODE import RK4
from disturb import disturbance
import time


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


# pre- allocating spaces
h = 1/300     # sampling rate(in Sec)
T = 20    # Total time
t = np.arange(0, T, h)     # discretising time
t = np.reshape(t, (1, t.size))
L = t.size   # Total time instance
dim = 3       # system dimension

flagOpt = 1

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

# radii = 0.2
# ang_ref = np.concatenate(
#     (radii*np.sin(t), radii*np.cos(t), radii*np.ones((1, L))), axis=0)
# ang_dot_ref = np.concatenate(
#     (radii*np.cos(t), -radii*np.sin(t), np.zeros((1, L))), axis=0)
# ang_ddot_ref = np.concatenate(
#     (-radii*np.sin(t), -radii*np.cos(t), np.zeros((1, L))), axis=0)
# ang_ref = np.zeros((dim, 1))
# for ti in np.arange(0, L, 1):
#     if(ti>5/h):
#         ang_temp= np.array([[0.32], [0],[0]])
#     else:
#         ang_temp = np.zeros((3,1))
#     ang_ref = np.concatenate((ang_ref, ang_temp), axis=1)
# ang_ref = np.array(ang_ref, dtype=np.float64)
ang_ref = np.concatenate(
    (np.zeros((1, L)), np.zeros((1, L)), np.zeros((1, L))), axis=0)
ang_dot_ref = np.concatenate(
    (np.zeros((1, L)), np.zeros((1, L)), np.zeros((1, L))), axis=0)
ang_ddot_ref = np.concatenate(
    (np.zeros((1, L)), np.zeros((1, L)), np.zeros((1, L))), axis=0)   

# constants in ctrl law
'''om1=1.94, om2=13, om3=1, eps=9.2, eta=0.15, gama=0.5 good gen'''
'''om1=2.5, om2=12.0, om3=1, eps=6.8, eta=0.3, gama=0.5 good iris'''
'''om1=1.96, om2=9.4, om3=1, eps=4.7, eta=0.46, gama=1 used with bounds'''
'''om1=4.0, om2=5.0, om3=1, eps=0.5, eta=2, gama=1 Iris without bounds @200hz'''
'''om1=2.5, om2=12, om3=1, eps=1.8, eta=0.7, gama=1 Tsalla without bounds @300hz'''
om1 = 4*np.eye(dim)      ##old estimate: 1.8, 6.8, 4.7, 2; for tsalla  
om2 = 5*om1
eps = 0.5
eta = 2

rho = 0.001
Kbar[0] = 0.1
Gama = 1.0*om1
om3 = 1.0*np.eye(dim)
J = np.diag([0.009, 0.009, 0.018])  # Polar MOI in body frame -----------Tsalla
# J = np.diag([0.029125,0.029125,0.055225])  # Polar MOI in body frame ------- Iris


#dis = np.zeros((dim, L))
dis = disturbance(T, h, dim, flag=1)

time_taken = []
ti = 0
for i in np.arange(0, L, 1):
    # #subscribe to MoCap (or) IMU for measurement of orientation and append it column wise
    # np.append(ang,__)
    # np.append(ang_dot,__)
    loop_start_time = time.perf_counter()
    
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
    time_taken.append(time.perf_counter() - loop_start_time)
    # print(f"\nerr_param  :\n {err_param} \nKbar_param  :\n {Kbar_param}")
    # print(f" \n Y  :  {Y}")
    # print(f"\n ang : {ang[:,i]} \nangDot : {ang_dot[:,i]}")

avg_time_taken = np.mean(time_taken)
# print(f'Average time taken per iteration: {avg_time_taken}')
# print(f'Hertz: {1/avg_time_taken}')
# print(f'err last: {err1[:,-1]} \n')
# print(err1)
mse = np.power(err1,2)
print(f"#####\n MSE values: {np.mean(mse[0,:])}, {np.mean(mse[1,:])}, {np.mean(mse[2,:])}")
if(flagOpt == 1):
    t = np.reshape(t, (t.size,))

    plt.subplot(2, 2, 1)
    plt.plot(t, ang[0, :-1], label='Actual', linewidth='1.2')
    plt.plot(t, ang_ref[0, :-1], label='Ref', linewidth='0.8')
    # plt.plot(t, dis[0, :], 'k.', markersize=0.1)
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("X-angle(rad)")

    plt.subplot(2, 2, 2)
    plt.plot(t, ang[1, :-1], label='Actual', linewidth='1.2')
    plt.plot(t, ang_ref[1, :-1], label='Ref', linewidth='0.8')
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("Y-angle(rad)")

    plt.subplot(2, 2, 3)
    plt.plot(t, ang[2, :-1], label='Actual', linewidth='1.2')
    plt.plot(t, ang_ref[2, :-1], label='Ref', linewidth='0.8')
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("Z-angle(rad)")

    plt.subplot(2, 2, 4)
    plt.plot(t, err1[0, :], label='x err', linewidth='1.4')
    plt.plot(t, err1[1, :], label='y err', linewidth='1.2')
    plt.plot(t, err1[2, :], label='z err', linewidth='1.2')
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("KAI error angle(-)")

    # plt.suptitle('Attitude Tracking results \nParametrs: om1 = 1.96*np.eye(3), om2 = 9.4*om1, eps = 4.7, eta = 0.46, Gama = om1, rho = 0.001, Kbar[0] = 0.1', fontsize=10)
    plt.show()

    # plt.cla()
    # plt.close()

    # plt.subplot(1, 3, 1)
    # plt.plot(t, torq[0, :], label='x ctrl', linewidth='1')
    # plt.legend()
    # plt.xlabel("Time(sec)")
    # plt.ylabel("input Torque(Nm)")
    # plt.subplot(1, 3, 2)
    # plt.plot(t, torq[1, :], label='y ctrl', linewidth='1')
    # plt.legend()
    # plt.subplot(1, 3, 3)
    # plt.plot(t, torq[2, :], label='z ctrl', linewidth='1')
    # plt.legend()
    # plt.suptitle('Control Signal(in Inertial frame)')

    # plt.show()



