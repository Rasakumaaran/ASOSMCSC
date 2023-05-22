 
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


def Kai_fn(_, x, Kai_param):
    alpha = Kai_param.get("alp")
    g = Kai_param.get("g")
    H = Kai_param.get("H")
    tau0_ddot = Kai_param.get("tdd")
    U = Kai_param.get("U")
    Kai = np.zeros((2, 3))
    #print('\n H:\n', H, '\n U:\n', U, '\n tdd:\n',tau0_ddot, '\n VecProd:\n', np.dot(g, alpha))
    Kai[0, :] = x[1, :]
    temp = np.dot(g, alpha) + H + U - tau0_ddot
    Kai[1, :] = temp.T
    return Kai


# pre- allocating spaces
h = 1/800     # sampling rate(in Sec)
T = 50    # Total time
t = np.arange(0, T, h)     # discretising time
t = np.reshape(t, (1, t.size))
L = t.size   # Total time instance
dim = 3       # system dimension

flagOpt = 1

ang = np.zeros((dim, L))  # Euler angles (x1)
ang_dot = np.zeros((dim, L))  # ang. Vel. in world frame (x2)
omega = np.zeros((dim, L))  # ang. Vel. in body frame
err1 = np.zeros((dim, L))  # error angles
mse = np.zeros((dim, L))  # mean Sq. error of angles 
err2 = np.zeros((dim, L))  # error ang_vel
tau1 = np.zeros((dim, L))
tau2 = np.zeros((dim, L))
kai2 = np.zeros((dim, L+1))
kai1 = np.zeros((dim, L+1))  # unconstarained variable error(tau-tau0)
kai_intg = np.zeros((dim, L))  # integration of kai in slide variable
sigma = np.zeros((dim, L))  # sliding variable
Kbar = np.zeros((L+1, 1))  # adap gain
U = np.zeros((dim, L))
torq = np.zeros((dim, L))

# temp = np.array([0.35, -0.25, -0.1]).T  # comment while using ROS
temp = np.array([0.0 , 0.0, 0.0]).T 
ang[:, 0] = temp

radii = 0.1
ang_ref = np.concatenate(
    (radii*np.sin(t), radii*np.cos(t), radii*np.ones((1, L))), axis=0)
ang_dot_ref = np.concatenate(
    (radii*np.cos(t), -radii*np.sin(t), np.zeros((1, L))), axis=0)
ang_ddot_ref = np.concatenate(
    (-radii*np.sin(t), -radii*np.cos(t), np.zeros((1, L))), axis=0)
UL = np.ones((3, ))*(0.4)
LL = -UL  # Upper & Lower limits

# constants in ctrl law
'''om1=1.94, om2=13, om3=1, eps=9.2, eta=0.15, gama=0.5 good gen'''
'''om1=1.915, om2=12.8, om3=1, eps=9.2, eta=0.2, gama=0.5 old'''
'''om1=2.5, om2=12.0, om3=1, eps=6.8, eta=0.3, gama=0.5 good iris'''
'''om1=1.96, om2=9.4, om3=1, eps=4.7, eta=0.46, gama=0.5 PrevPerf'''
# om1c = 1.845 ; om2c = 10.45 ; gamac = 50
# eta = 4.7 ; eps = 183

om1c = 2.5 ; om2c = 12 ; gamac = 0.50
eta = 0.3 ; eps = 6.8
om1 = om1c*np.eye(dim)
om2 = om2c*om1
rho = 0.001
Kbar[0] = 0.1
Gama = gamac*np.eye(dim)
om3 = 1.0*np.eye(dim)
# J = np.diag([0.009, 0.009, 0.018])  # Polar MOI in body frame -----------Tsalla
# J = np.diag([0.01, 0.01, 0.02]) 
J = np.diag([0.029125,0.029125,0.055225])  # Polar MOI in body frame ------- Iris


#dis = np.zeros((dim, L))
dis = disturbance(T, h, dim, flag=1)

time_taken = []

for i in np.arange(0, L, 1):
    # #subscribe to MoCap (or) IMU for measurement of orientation and append it column wise
    # np.append(ang,__)
    # np.append(ang_dot,__)
    loop_start_time = time.perf_counter()

    Ulim = UL-ang_ref[:, i]
    Llim = LL-ang_ref[:, i]
    Lamda = np.diag(Ulim-Llim)
    tau0 = np.log(np.divide(-Llim, Ulim))
    tau0_dot = (np.divide(-ang_dot_ref[:, i], Llim)) - \
        (np.divide(-ang_dot_ref[:, i], Ulim))
    tau0_ddot = np.divide((np.multiply(-ang_ddot_ref[:, i], Llim) - np.power(ang_dot_ref[:, i], 2)), np.power(Llim, 2)) \
        - np.divide((np.multiply(-ang_ddot_ref[:, i], Ulim) -
                    np.power(ang_dot_ref[:, i], 2)), np.power(Ulim, 2))

    tau1[:, i] = kai1[:, i] + tau0
    tau2[:, i] = kai2[:, i] + tau0_dot

    if (i == 0):
        err1[:, i] = ang[:, i] - ang_ref[:, i]
        err2[:, i] = ang_dot[:, i] - ang_dot_ref[:, i]
        tau1[:, i] = np.log(
            np.divide((err1[:, i] - Llim), (Ulim - err1[:, i])))
        kai1[:, i] = tau1[:, i] - tau0

    psi = np.divide(np.exp(tau1[:, i]), (np.exp(tau1[:, i])+1))  # sigmoid fun
    psid = np.divide(np.exp(tau1[:, i]), np.power(
        (np.exp(tau1[:, i])+1), 2))  # derivative of sigmoid fun
    psi_dot = np.diag(psid)
    g = np.linalg.inv(np.dot(Lamda, psi_dot))

    if (i == 0):
        tau2[:, i] = np.dot(g, (err2[:, i] + ang_dot_ref[:, i]))
        kai2[:, i] = tau2[:, i] - tau0_dot

    if (i > 0):
        err2[:, i] = np.dot(np.linalg.inv(g), tau2[:, i]) + \
            (-ang_dot_ref[:, i])
        err1[:, i] = Llim + np.dot(Lamda, psi)
        ang_dot[:, i] = err2[:, i] + ang_dot_ref[:, i]
        ang[:, i] = err1[:, i] + ang_ref[:, i]

    # publish to /setpoint_attitude

    R = [[1, np.sin(ang[0, i])*np.tan(ang[1, i]), np.cos(ang[0, i])*np.tan(ang[1, i])],
         [0, np.cos(ang[0, i]), -np.sin(ang[0, i])],
         [0, np.sin(ang[0, i])/np.cos(ang[1, i]), np.cos(ang[0, i])/np.cos(ang[1, i])]]
    R = np.array(R)

    if(i == 0):
        Rold = R
    R_dot = (R - Rold)/h
    Rold = R

    beta = np.dot(R, np.linalg.inv(J))

    omega[:, i] = np.dot(np.linalg.inv(R), ang_dot[:, i])

    # computing Sliding variable
    # trapezoidal integration
    if (i == 0):
        kai_intg[:, i] = h*kai1[:, i]/2
    elif (i == 1):
        kai_intg[:, i] = h*(kai1[:, i] + kai1[:, i-1])/2
    else:
        kai_intg[:, i] = (kai1[:, i] + 2/h*kai_intg[:, i-1] + kai1[:, i-1])*h/2

    kai_diff = kai2[:, i]
    # print(f"\nkai_int :  {kai_intg},  \n shape : {kai_intg[:, i].shape}")
    sigma[:, i] = np.dot(om1, kai1[:, i]) + \
        np.dot(om2, kai_intg[:, i]) + np.dot(om3, kai_diff)
    sigNom = np.linalg.norm(sigma[:, i])

    H = np.zeros((dim, 1))
    H = np.multiply(np.tanh(tau1[:, i]/2), np.power((tau2[:, i]), 2)) 
    # H = np.multiply(np.tanh(
    #     tau1[:, i]/2), np.power((tau2[:, i]), 2)) + np.dot(g, ang_ddot_ref[:, i])

    temp = (np.dot(-om1, kai_diff) - np.dot(om2, kai1[:, i]) - H)
    # print(f" \nkai1[] :  {kai1[:, i]}  \nkai2[] :  {kai2[:, i]}")
    temp = np.reshape(temp, (temp.size, 1))
    # print(f"\n Temp :  {temp} \n Shape : {temp.shape}")   
    f = np.concatenate((np.linalg.inv(psi_dot), temp),
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
                np.dot(om1, kai_diff) + np.dot(om2, kai1[:, i]) + H)
    torq[:,i] = np.dot(np.linalg.inv(beta), np.dot(np.linalg.inv(g) ,U[:,i]))
    # print(f"\n omg : {omega}, \n U : {U}")

    '''
    U[:,i] = np.linalg.inv(np.dot(g, beta)) @ U[:,i]  #actual torque cmd
    how to include thrust??
    # publish to /actuator_control 
    '''
    alpha = np.dot(beta, (np.cross(-omega[:, i], np.dot(J, omega[:, i])) +
                   dis[:, i])) + np.dot(R_dot, omega[:, i]) - ang_ddot_ref[:, i]

    Kai_param = {"alp": alpha, "g": g, "H": H, "tdd": tau0_ddot, "U": U[:, i]}
    ###########
    Y = RK4(Kai_fn, 0, np.array([kai1[:, i], kai2[:, i]]), h, Kai_param)
    ##########
    kai1[:, i+1] = Y[0, :].T
    kai2[:, i+1] = Y[1, :].T
    time_taken.append(time.perf_counter() - loop_start_time)
    # print(f"\nKai_param  :\n {Kai_param} \nKbar_param  :\n {Kbar_param}")
    # print(f" \n Y  :  {Y}")
    # print(f"\n ang : {ang[:,i]} \nangDot : {ang_dot[:,i]}")

avg_time_taken = np.mean(time_taken)
# print(f'Average time taken per iteration: {avg_time_taken}')
# print(f'Hertz: {1/avg_time_taken}')
# print(f'Kai last: {kai1[:,-1]} \n')
# print(kai1)
mse = np.power(err1,2)
print(f"#####\n MSE values: {np.mean(mse[0,:])}, {np.mean(mse[1,:])}, {np.mean(mse[2,:])}")
if(flagOpt == 1):
    t = np.reshape(t, (t.size,))

    plt.subplot(2, 2, 1)
    plt.plot(t, ang[0, :], label='Actual', linewidth='1.2')
    plt.plot(t, ang_ref[0, :], label='Ref', linewidth='0.8')
    # plt.plot(t, dis[0, :], 'k.', markersize=0.1)
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("X-angle(rad)")

    plt.subplot(2, 2, 2)
    plt.plot(t, ang[1, :], label='Actual', linewidth='1.2')
    plt.plot(t, ang_ref[1, :], label='Ref', linewidth='0.8')
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("Y-angle(rad)")

    plt.subplot(2, 2, 3)
    plt.plot(t, ang[2, :], label='Actual', linewidth='1.2')
    plt.plot(t, ang_ref[2, :], label='Ref', linewidth='0.8')
    plt.legend()
    plt.xlabel("Time(sec)")
    plt.ylabel("Z-angle(rad)")

    plt.subplot(2, 2, 4)
    # plt.plot(t, kai1[0, :-1], label='x err', linewidth='1.4')
    # plt.plot(t, kai1[1, :-1], label='y err', linewidth='1.2')
    # plt.plot(t, kai1[2, :-1], label='z err', linewidth='1.2')
    plt.plot(t, torq[0, :], label='x torq', linewidth='1.4')
    plt.plot(t, torq[1, :], label='y torq', linewidth='1.2')
    plt.plot(t, torq[2, :], label='z torq', linewidth='1.2')
    plt.legend()
    plt.xlabel("Time(sec)")
    # plt.ylabel("KAI error angle(-)")
    plt.ylabel("Torq")

    plt.suptitle(f'Attitude Tracking (Numerical) \nParametrs: om1 = {om1c}*eye(3), om2 = {om2c}*om1, eps = {eps}, eta = {eta}, Gama = {gamac}eye(3), rho = 0.001, Kbar[0] = 0.1', fontsize=10)
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



