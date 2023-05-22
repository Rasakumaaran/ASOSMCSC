import numpy as np
import matplotlib.pyplot as plt
import time

h = 1/200     # sampling rate(in Sec)
T = 50    # Total time
t = np.arange(0, T, h)     # discretising time
t = np.reshape(t, (1, t.size))
L = t.size   # Total time instance
dim = 3       # system dimension
radii = 1

omega_list, tau_list, Ialp_list = [], [], []
R_nom = []

ang_ref = np.concatenate(
    (radii*np.sin(t), radii*np.cos(t), radii*np.ones((1, L))), axis=0)
ang_dot_ref = np.concatenate(
    (radii*np.cos(t), -radii*np.sin(t), np.zeros((1, L))), axis=0)
ang_ddot_ref = np.concatenate(
    (-radii*np.sin(t), -radii*np.cos(t), np.zeros((1, L))), axis=0)

t1 = 0
J = np.diag([0.019, 0.019, 0.035])
J_inv = np.linalg.pinv(J)
tau_cap = 2.21
t0 = time.perf_counter()

for i in np.arange(0, L, 1):
    # ang_ref = [radii*np.sin(t1), radii*np.cos(t1), 0.2]
    # ang_ref_d = np.rad2deg(ang_ref)
    # ang_dot_ref = np.array([radii*np.cos(t1), -radii*np.sin(t1), 0.0])
    # ang_ddot_ref = np.array([-radii*np.sin(t1), -radii*np.cos(t1), 0.0])
    
    R = [[1, np.sin(ang_ref[0,i])*np.tan(ang_ref[1,i]), np.cos(ang_ref[0,i])*np.tan(ang_ref[1,i])],
        [0, np.cos(ang_ref[0,i]), -np.sin(ang_ref[0,i])],
        [0, np.sin(ang_ref[0,i])/np.cos(ang_ref[1,i]), np.cos(ang_ref[0,i])/np.cos(ang_ref[1,i])]]
    R = np.array(R, dtype=np.float64)
    R_inv = np.linalg.pinv(R)
    R_nom.append(np.linalg.norm(R))
    
    # Rd12 = np.cos(ang_ref[0, i])*np.tan(ang_ref[1, i])*ang_dot_ref[0,i] + np.sin(ang_ref[0, i])*ang_dot_ref[1,i]/(np.cos(ang_ref[1, i]))**2
    # Rd13 = -np.sin(ang_ref[0, i])*np.tan(ang_ref[1, i])*ang_dot_ref[0,i] + np.cos(ang_ref[0, i])*ang_dot_ref[1,i]/(np.cos(ang_ref[1, i]))**2
    # Rd32 = (np.cos(ang_ref[0, i])*np.cos(ang_ref[1, i])*ang_dot_ref[0,i] + np.sin(ang_ref[0, i])*np.sin(ang_ref[1, i])*ang_dot_ref[1,i])/(np.cos(ang_ref[1, i]))**2
    # Rd33 = (-np.sin(ang_ref[0, i])*np.cos(ang_ref[1, i])*ang_dot_ref[0,i] + np.cos(ang_ref[0, i])*np.sin(ang_ref[1, i])*ang_dot_ref[1,i])/(np.cos(ang_ref[1, i]))**2
    # R_dot = [[0, Rd12, Rd13],
    #      [0, -np.sin(ang_ref[0, i])*ang_dot_ref[0,i], -np.cos(ang_ref[0, i])*ang_dot_ref[0,i]],
    #      [0, Rd32, Rd33]]
    # R_dot = np.array(R_dot)
    if(i == 0):
        Rold = R
    R_dot = (R - Rold)/h
    Rold = R
    
    omega_ref = R_inv @ ang_dot_ref[:,i]

    tau = J @ R_inv @ (ang_ddot_ref[:,i] - R_dot @ omega_ref) + np.cross(omega_ref, np.dot(J, omega_ref))           
    tau = J_inv @ tau
    Ialp = J @ ang_ddot_ref[:,i]
    # tau_norm = np.where(abs(tau)>tau_cap, tau_cap, tau)
    # tau_norm = (2/7.8)*(tau_norm+3.9) - 1
    # act.controls = [tau_norm[0], tau_norm[1], tau_norm[2], 0.4, 0,0,0,0]
    
    omega_list.append(omega_ref)
    tau_list.append(tau)
    Ialp_list.append(Ialp)
    
omega_list = np.array(omega_list)
tau_list = np.array(tau_list)
Ialp_list = np.array(Ialp_list)
R_nom = np.array(R_nom)

print(ang_ref.shape, tau_list.shape, omega_list.shape, omega_list.shape)
print(np.mean(R_nom))
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t[0,:], tau_list[:,1], label='Body')
plt.plot(t[0,:], ang_ddot_ref[1,:], label='World')
plt.legend()
plt.ylabel("Torq(Nm)")

plt.subplot(2,1,2)
plt.plot(t[0,:], omega_list[:,0], label='Body')
plt.plot(t[0,:], ang_dot_ref[0,:], label='World')
plt.legend()
plt.ylabel("AngVel(rad/s)")
plt.show()


