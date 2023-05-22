import numpy as np
import rospy
from solveODE import RK4
from collections import deque

##############
# Modified gains, H, fNom
##############  

class controller:
    def __init__(self):
        self.h = 1/400     # sampling rate(in Sec)
        self.dim = 3       # system dimension
        self.t = 0.0       # initial time
        self.radii = 0.15    #radius to track

        self.ang_qu_len = 7
        self.ang_qu = deque(maxlen= self.ang_qu_len)
        self.kai1_qu = deque(maxlen= 3)
        self.kai_intg_qu = deque(maxlen= 2)
        self.Kbar_qu = deque(maxlen= 2)
        self.kai1_qu.append(np.zeros((self.dim,)))      # unconstrained variable error(tau-tau0)
        self.kai_intg_qu.append(np.zeros((self.dim,)))   # integration of kai in slide variable
        self.Kai = np.zeros((2, 3))
        
        self.beta, self.R_inv = np.eye(3), np.eye(3)
        
        ###########
        # Upper & Lower limits
        self.UL = np.ones((3,))*(0.4)
        self.LL = -self.UL  
        self.Lamda = np.diag((self.UL-self.LL))
        self.Lamda_inv = np.linalg.pinv(self.Lamda)

        # constants in ctrl law
        '''om1=1.94, om2=13, om3=1, eps=9.2, eta=0.15, gama=0.5 good gen'''
        '''om1=1.915, om2=12.8, om3=1, eps=9.2, eta=0.2, gama=0.5 old(i)'''
        '''om1=2.5, om2=12.0, om3=1, eps=6.8, eta=0.3, gama=0.5 good iris (ii)'''
        
        '''4.5,11,20,0.1,5,0.1  - Iris (200Hz) (iii)'''
        '''{4,15,15,0.2,12,0.1} (iv) {3,10,12,0.1,8} (v) {2.5,15,15,0.1,7} (vi) -  Iris(90Hz sine, 120Hz Gauss)'''
        
        '''5,20,35,10,20,0.1 good with sinusoidal disturbance, Sigma btw (-3,8)     Tsalla CAD 
        5,20,35,10,72,0.5 both disturb, Adap gain along with disturb, Sigma btw (-3,8) (1,15)'''
        '''{3,9,42,10,20,0.5}, {4,9,80,16,30,0.5} //{2,4.5,1.1,2.5,80}, {2.8,9,25,10,20}'''
        
        '''With Calculated MOI {5,20,15,8.5,60,0.1,0.01}, {5,20,5,8.5,55,0.1,0.01} with reduced sine disturbances
        With Usual sine disturbances {5,20,15,8.5,55,0.1,1}     \\{5.5,28,12,7,70,0.1,0.01}
        With slight higher disturbances {6,28,15,8.5,65,0.1,0.01}'''
        
        '''{2.5,8,2,0.1,5} {2,4,2,3,4} {0.5,0.5,1,0.05,1} SITL didn't fail and didn't track'''
        '''{0.1, 0.2, 1, 0.05, 1, 0.1, 0.1}  S&HITL didn't fail and didn't track (slowly unsteady)'''
        # self.om1 = np.diag([1, 1, 1.2])
        # self.om2 = np.diag([2, 2, 2.1])
        # self.Gama = np.diag([1, 1, 1])
        self.om1 = 0.5*np.eye(3)
        self.om2 = 0.8*np.eye(3)
        self.Gama = 1*np.eye(3)
        self.eta = 0.4                 
        self.rho = 1.2
        
        self.eps = 0.1
        self.Kbar_qu.append(0.1)
        
        self.J = np.diag([0.029125,0.029125,0.055225])  # Polar MOI in body frame ------- Iris
        # self.J = np.diag([0.019, 0.019, 0.035])      ## MOI for Tsalla Calculated
        # self.J = np.diag([0.01, 0.01, 0.019])
        self.m = 1.5                             # mass
        self.Jinv = np.linalg.pinv(self.J)
        
        self.cons1 = min(np.min(np.diag(self.Gama)), self.rho/2) / max(0.5, 0.5/self.eta)   
        self.cons2 = 1e9   
        self.tau_max = np.array([3.8, 3.8, 0.18])

    def Kbar_fn(self,_, x, Kbar_param):
        eta, sigNom, fNom, rho = Kbar_param[0], Kbar_param[1], Kbar_param[2], Kbar_param[3]
        self.Kbar = eta*sigNom*fNom - rho*eta*x
        return self.Kbar
  
    def Teta_fn(self,_, x, Kai_param):
        # alpha , U = Kai_param[0], Kai_param[1]
        alpha = Kai_param.get("alp")
        U = Kai_param.get("U")
        
        self.Kai[0, :] = x[1, :]
        # temp = alpha + U
        omega = self.R_inv @ x[1, :]
        temp = np.dot(self.beta, (np.cross(-omega, np.dot(self.J,omega)) )) + np.dot(self.R_dot,omega) + U
        self.Kai[1, :] = temp.T
        # print(f'[Teta_fn] X: {x} ; Kai: {self.Kai} ')
        return self.Kai

    def controller(self, states, freq):  
        self.h = 1/freq
        # print(f'[self.controller] states: {states}')
        ang = states[0, :]
        self.R = [[1, np.sin(ang[0])*np.tan(ang[1]), np.cos(ang[0])*np.tan(ang[1])],\
            [0, np.cos(ang[0]), -np.sin(ang[0])],\
            [0, np.sin(ang[0])/np.cos(ang[1]), np.cos(ang[0])/np.cos(ang[1])]]
        self.R = np.array(self.R, dtype=np.float64)
        self.R_inv = np.linalg.inv(self.R)
        
        ang_dot = states[1, :]
        omega = ang_dot
        ang_dot = np.dot(self.R , ang_dot)    ## converted to ang. vel. to world frame coords

        # self.ang_qu.append(ang)
        # if(self.t==0.0):
        #     ang_dot = self.ang_qu[0]
        # elif((self.t/self.h)<=self.ang_qu_len):
        #     ang_dot = (self.ang_qu[-1] - self.ang_qu[-2])/self.h
        # else:
        #     ang_dot = (self.ang_qu[-1] + 2*self.ang_qu[-2] - 2*self.ang_qu[-4] - self.ang_qu[-5])/(8*self.h)
        #     # from http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/ 
        # omega = self.R_inv @ ang_dot

        ang_ref = np.array([self.radii*np.sin(self.t), self.radii*np.cos(self.t), 0.0])
        ang_dot_ref = np.array([self.radii*np.cos(self.t), -self.radii*np.sin(self.t), 0.0])
        ang_ddot_ref = np.array([-self.radii*np.sin(self.t), -self.radii*np.cos(self.t), 0.0])
        # # self.ang_ref, self.ang_dot_ref, self.ang_ddot_ref = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))

        # ang_ref = np.array([0.0, self.radii*np.cos(self.t), 0.0])
        # ang_dot_ref = np.array([0.0, -self.radii*np.sin(self.t), 0.0])
        # ang_ddot_ref = np.array([0.0, -self.radii*np.cos(self.t), 0.0])
                
        Ulim = self.UL - ang_ref
        Llim = self.LL - ang_ref

        # x = self.Ulim - self.Llim
        # Lamda = np.diag(x)
        tau0 = np.log(np.divide(-Llim, Ulim))
        tau0_dot = (np.divide(-ang_dot_ref, Llim)) - \
            (np.divide(-ang_dot_ref, Ulim))
        tau0_ddot = np.divide((np.multiply(-ang_ddot_ref, Llim) - np.power(ang_dot_ref, 2)), np.power(Llim, 2)) \
            - np.divide((np.multiply(-ang_ddot_ref, Ulim) - np.power(ang_dot_ref, 2)), np.power(Ulim, 2))
        
        err1 = ang - ang_ref
        # self.err2 = self.ang_dot - self.ang_dot_ref
        
        tau1 = np.log(np.divide((err1 - Llim), (Ulim - err1)))
        psid = np.divide(np.exp(tau1), np.power((np.exp(tau1)+1), 2))
        psi_dot = np.diag(psid)
        # psi_dot_inv = np.diag(1/psid)
        # psi_dot_inv = np.linalg.pinv(psi_dot)
        LamPsi = np.dot(self.Lamda , psi_dot)
        g = np.linalg.pinv(LamPsi)
        
        # tau2 = g @ (self.err2 + self.ang_dot_ref)
        tau2 = np.dot(g , ang_dot)
        kai1_inst = (tau1 - tau0)
        self.kai1_qu.append(kai1_inst)
        
        if(self.t == 0.0):
            self.Rold = self.R
        self.R_dot = (self.R - self.Rold)/self.h
        self.Rold = self.R

        self.beta = np.dot(self.R , self.Jinv)

        # computing Sliding variable
        # trapezoidal integration
        if (self.t == 0.0):
            temp = self.h*self.kai1_qu[-1]/2
        elif (self.t == self.h):
            temp = self.h*(self.kai1_qu[-1] + self.kai1_qu[-2])/2
        else:
            temp = (self.kai1_qu[-1] + self.kai1_qu[-2])*self.h/2 + self.kai_intg_qu[-1]
        self.kai_intg_qu.append(temp)

        kai_diff = tau2 - tau0_dot
    
        sigma = np.dot(self.om1 , self.kai1_qu[-1]) + np.dot(self.om2 , self.kai_intg_qu[-1]) + kai_diff
        sigNom = np.linalg.norm(sigma)

        # H = np.multiply(np.tanh(tau1/2), np.power((tau2), 2)) + np.dot(g , ang_ddot_ref)
        H = np.multiply(np.tanh(tau1/2), np.power((tau2), 2)) - tau0_ddot

        # temp = (np.dot(-self.om1,kai_diff) - np.dot(self.om2,self.kai1_qu[-1]) - H)
        # temp = np.reshape(temp, (temp.shape[0],1))    
        # f = np.concatenate((psi_dot_inv, temp),axis=1)
        # fNom = np.linalg.norm(f)
        fNom = np.linalg.norm(g)
        # adaptive gain
        # Kbar_param = {"eta": self.eta, "sNom": sigNom, "fNom": fNom, "rho": self.rho}
        Kbar_param = [self.eta, sigNom, fNom, self.rho]
        #######
        out = RK4(self.Kbar_fn, 0.0, self.Kbar_qu[-1], self.h, Kbar_param)
        self.Kbar_qu.append(out)
        ########

        if (sigNom < self.eps):
            sat = sigma/self.eps
        else:
            sat = sigma/sigNom

        mu2 = self.Kbar_qu[-1]*fNom*sat
        Mu = -(np.dot(self.Gama,sigma) + mu2 + np.dot(self.om1,kai_diff) + \
            np.dot(self.om2,self.kai1_qu[-1]) + H)
        U = LamPsi@Mu
        Tau = np.linalg.pinv(self.beta) @ U
        Tau_norm = np.divide(Tau, self.tau_max)

        alpha = np.dot(self.beta, (np.cross(-omega, np.dot(self.J,omega)) )) + \
            np.dot(self.R_dot,omega) 
        
        alpha_max = 450     ## without delta, it's 60. Assuming delta=4 Nm in each axis, it's 450
        Lyap = 0.5*np.dot(sigma.T, sigma) + (0.5/self.eta)*(self.Kbar_qu[-1] - alpha_max)**2
        Lyap_dot = sigma.T @ (g@alpha - self.Gama@sigma - self.Kbar_qu[-1]*fNom*sat) + (self.Kbar_qu[-1] - alpha_max)*(sigNom*fNom - self.rho*self.Kbar_qu[-1])
        temp = -Lyap_dot/Lyap
        if self.cons2>temp :
            self.cons2 = temp
            
        # Teta_param = [alpha, U] #{"alp": alpha, "U": U}
        Teta_param = {"alp": alpha, "U": U}
        initC = np.array([ang, ang_dot])
        print(f'[self.controller] alpha: {alpha}')

        ###########
        Y = RK4(self.Teta_fn, 0.0, initC, self.h, Teta_param)
        ##########
        ang_pub = Y[0,:]
        ang_dot_pub = Y[1,:]
        ########### Thrust Force(in Kgf) calculation                  
        thrust = self.m/(np.cos(ang_pub[1])*np.cos(ang_pub[0]))       ## considering Z-world vel& Accl.=0
        thrust_norm = (thrust-1.5)*(0.32/1.35) + 0.72               ## how to ensure the mapping??
        # thrust_norm = thrust_norm if thrust_norm<0.85 else 0.85
        # thrust_norm = 0.74

        self.t = self.t + self.h 
        ### since for now we are passing only attitude setpoint 
        # R = [[1, np.sin(ang_pub[0])*np.tan(ang_pub[1]), np.cos(ang_pub[0])*np.tan(ang_pub[1])],\
        #     [0, np.cos(ang_pub[0]), -np.sin(ang_pub[0])],\
        #     [0, np.sin(ang_pub[0])/np.cos(ang_pub[1]), np.cos(ang_pub[0])/np.cos(ang_pub[1])]]
        # R = np.array(R, dtype=np.float64)
        # R_inv = np.linalg.pinv(R)
        ang_dot_pub  = np.dot(self.R_inv,ang_dot_pub)          ## angular velocity in body reference frame
        final = {'time':self.t, 'ang_vel':ang_dot_pub.tolist(), 'ang':ang_pub.tolist(), 'thrust':thrust_norm, 'torq':Tau.tolist(), 'torq_norm':Tau_norm.tolist(), 'kbar':out, 'sigma':sigma.tolist(), \
                 'a_ref':ang_ref.tolist(), 'adot_ref':ang_dot_ref.tolist(), 'err1':kai1_inst.tolist(), 'Const1':self.cons1, 'Const2':self.cons2 }
        print(f'stable : {((self.cons1 > self.cons2) & (self.cons2 > 0))}')
        return final

        