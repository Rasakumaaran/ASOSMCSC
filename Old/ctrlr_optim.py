import numpy as np
# import matplotlib.pyplot as plt
# from pyrsistent import s
from solveODE import RK4

from tf.transformations import euler_from_quaternion

class controller:
    def __init__(self):
        self.h = 1/400     # sampling rate(in Sec)
        self.dim = 3       # system dimension
        self.t = 0         # initial time
        self.radii = 0.15    #radius to track

        self.ang = np.zeros((self.dim, 1))      # Euler angles (x1)
        self.ang_dot = np.zeros((self.dim, 1))  # ang. Vel. in world frame (x2)
        self.omega = np.zeros((self.dim, 1))    # ang. Vel. in body frame
        self.err1 = np.zeros((self.dim, 1))     # error angles
        self.err2 = np.zeros((self.dim, 1))     # error ang_vel
        self.tau1 = np.zeros((self.dim, 1))
        self.tau2 = np.zeros((self.dim, 1))
        self.kai1 = np.zeros((self.dim, 1))     # unconstarained variable error(tau-tau0)
        self.kai_intg = np.zeros((self.dim, 1))  # integration of kai in slide variable
        self.sigma = np.zeros((self.dim, 1))    # sliding variable
        self.Kbar = []                          # adap gain
        self.U = np.zeros((self.dim, 1))
        
        ###########
        # Upper & Lower limits
        self.UL = np.ones((3, 1))*(0.9)
        self.LL = -self.UL  

        # constants in ctrl law
        '''om1=1.94, om2=13, om3=1, eps=9.2, eta=0.15, gama=0.5 good gen'''
        '''om1=1.915, om2=12.8, om3=1, eps=9.2, eta=0.2, gama=0.5 old'''
        '''om1=2.5, om2=12.0, om3=1, eps=6.8, eta=0.3, gama=0.5 good iris'''
        self.om1 = 0.1*np.eye(self.dim)
        self.om2 = 0.2*self.om1
        # self.om3 = 1.0*np.eye(self.dim)
        self.eps = 0.1
        self.eta = 0.05
        self.Gama = 1*self.om1
        self.rho = 1
        
        self.Kbar = np.append(self.Kbar, 0.1)
        
        # self.J = np.diag([0.01, 0.01, 0.02])  # Polar MOI in body frame ----- Tsalla
        # self.m = 1.8                             # mass
        self.J = np.diag([0.029125,0.029125,0.055225])  # Polar MOI in body frame ------- Iris
        self.m = 1.5
        

    def Kbar_fn(self,_, x, Kbar_param):
        eta = Kbar_param.get("eta")
        sigNom = Kbar_param.get("sNom")
        fNom = Kbar_param.get("fNom")
        rho = Kbar_param.get("rho")

        self.Kbar = eta*sigNom*fNom - rho*eta*x
        return self.Kbar
  
    def Teta_fn(self,_, x, Kai_param):
        alpha = Kai_param.get("alp")
        U = Kai_param.get("U")

        self.Kai = np.zeros((2, 3))
        #print('\n H:\n', H, '\n U:\n', U, '\n tdd:\n',tau0_ddot, '\n VecProd:\n', np.dot(g, alpha))
        self.Kai[0, :] = x[1, :]
        temp = alpha + U
        self.Kai[1, :] = temp.T
        return self.Kai

    def transformation(self,Lamda, tau1):
        psi = np.divide(np.exp(tau1), (np.exp(tau1)+1))  # sigmoid fun
        # derivative of sigmoid fun
        psid = np.divide(np.exp(tau1), np.power((np.exp(tau1)+1), 2))
        psid = np.reshape(psid,(psid.shape[0],))
        psi_dot = np.diag(psid)
        # print(f"\npsidot:{psi_dot}")
        LamPsi = np.dot(Lamda, psi_dot)
        g = np.linalg.inv(LamPsi)

        output = {"g": g, "LamPsi":LamPsi, "psi": psi, "psi_dot": psi_dot}
        return output

    def rotMat(self,ang,inverse):
        R = [[1, np.sin(ang[0])*np.tan(ang[1]), np.cos(ang[0])*np.tan(ang[1])],
            [0, np.cos(ang[0]), -np.sin(ang[0])],
            [0, np.sin(ang[0])/np.cos(ang[1]), np.cos(ang[0])/np.cos(ang[1])]]
        R = np.array(R, dtype=np.float64)
        if (inverse==1):
            R = np.linalg.inv(R)    
        return R

    def controller(self, states, freq):  

        self.ang = states[0, :]  
        self.ang = self.ang.T                   
        # self.ang = self.ang.T - np.ones((3,1))*np.pi        # convert the angles in range (-pi,pi)
        self.ang_dot = states[1, :]
        self.ang_dot = self.ang_dot.T
        
        self.h = 1/freq

        omega = self.ang_dot
        self.R = self.rotMat(self.ang, 0)
        self.ang_dot = np.dot(self.R, self.ang_dot)    ## converted to ang. vel. to world frame coords

        self.ang_ref = np.array([[self.radii*np.sin(self.t)], [self.radii*np.cos(self.t)], [0.2]])
        self.ang_dot_ref = np.array([[self.radii*np.cos(self.t)], [-self.radii*np.sin(self.t)], [0]])
        self.ang_ddot_ref = np.array([[-self.radii*np.sin(self.t)], [-self.radii*np.cos(self.t)], [0]])

        dis = np.array([[0], [0],[0]])
        
        self.Ulim = self.UL - self.ang_ref
        self.Llim = self.LL - self.ang_ref
        
        x = self.Ulim - self.Llim
        x = np.reshape(x,(x.shape[0],))
        Lamda = np.diag(x)
        self.tau0 = np.log(np.divide(-self.Llim, self.Ulim))
        # print(f"\ntau0:{self.tau0}, \n angle: {self.ang}, \n angle dot: {self.ang_dot}")
        self.tau0_dot = (np.divide(-self.ang_dot_ref, self.Llim)) - \
            (np.divide(-self.ang_dot_ref, self.Ulim))
        self.tau0_ddot = np.divide((np.multiply(-self.ang_ddot_ref, self.Llim) - np.power(self.ang_dot_ref, 2)), np.power(self.Llim, 2)) \
            - np.divide((np.multiply(-self.ang_ddot_ref, self.Ulim) -
                        np.power(self.ang_dot_ref, 2)), np.power(self.Ulim, 2))

        self.err1 = self.ang - self.ang_ref
        self.err2 = self.ang_dot - self.ang_dot_ref
        
        self.tau1 = np.log(np.divide((self.err1 - self.Llim), (self.Ulim - self.err1)))
        # print(f"\n error:{self.err1}, \n R:{R} \nRef_angles:{self.ang_ref}  \nLambda:{Lamda}  \nTau1: {self.tau1}")
        trans = self.transformation(Lamda, self.tau1)
        g = trans.get("g")
        psi_dot = trans.get("psi_dot")
        LamPsi = trans.get("LamPsi")

        self.tau2 = np.dot(g, (self.err2 + self.ang_dot_ref))
        kai1_inst = (self.tau1 - self.tau0)
        self.kai1 = np.concatenate((self.kai1, kai1_inst), axis=1)
        # print(f"\ntu1Shap: {np.shape(self.tau1)},  \ntu2Shap: {np.shape(self.tau2)} \nki1Shap: {np.shape(self.kai1)}  \nki2Shap: {np.shape(self.kai2)}")
        
        if(self.t == 0):
            self.Rold = self.R
        self.R_dot = (self.R - self.Rold)/self.h
        self.Rold = self.R
        R_inv = np.linalg.inv(self.R)

        beta = np.dot(self.R, np.linalg.inv(self.J))

        # computing Sliding variable
        # trapezoidal integration
        if (self.t == 0):
            temp = self.h*self.kai1[:, -1]/2
        elif (self.t == self.h):
            temp = self.h*(self.kai1[:, -1] + self.kai1[:, -2])/2
        else:
            temp = (self.kai1[:, -1] + self.kai1[:, -2])*self.h/2 + self.kai_intg[:, -1]
        temp = np.reshape(temp,(temp.shape[0],1))
        self.kai_intg = np.concatenate((self.kai_intg, temp), axis=1)

        kai_diff = self.tau2 - self.tau0_dot
        kai_diff = np.reshape(kai_diff, (kai_diff.size, ))
    
        sigma = np.dot(self.om1, self.kai1[:, -1]) + \
            np.dot(self.om2, self.kai_intg[:, -1]) + kai_diff
        sigNom = np.linalg.norm(sigma)

        H = np.zeros((self.dim, 1))
        H = np.multiply(np.tanh(self.tau1/2), np.power((self.tau2), 2)) + np.dot(g, self.ang_ddot_ref)
        H = np.reshape(H,(H.shape[0], ))

        temp = (np.dot(-self.om1, kai_diff) - np.dot(self.om2, self.kai1[:, -1]) - H)
        # print(f"\n H :  {H} \n I Temp :  {temp} \nkai_diff :  {kai_diff} \nkai1[] :  {self.kai1[:, -1]}")
        temp = np.reshape(temp, (temp.size, 1))
        # print(f"\n Temp :  {temp} \n Shape : {temp.shape}")    
        f = np.concatenate((np.linalg.inv(psi_dot), temp),axis=1)
        fNom = np.linalg.norm(f)
        # adaptive gain
        Kbar_param = {"eta": self.eta, "sNom": sigNom, "fNom": fNom, "rho": self.rho}
        #######
        out = RK4(self.Kbar_fn, 0, self.Kbar[-1], self.h, Kbar_param)
        self.Kbar = np.append(self.Kbar, out)
        ########

        if (sigNom < self.eps):
            sat = sigma/self.eps
        else:
            sat = sigma/sigNom
        sat = np.reshape(sat,(sat.shape[0], ))

        Mu = -(np.dot(self.Gama, sigma) + self.Kbar[-1]*fNom*sat +
            np.dot(self.om1, kai_diff) + np.dot(self.om2, self.kai1[:, -1]) + H)
        U = np.dot(LamPsi,Mu)
        Tau = np.linalg.inv(beta) @ U
        # print(f"\n Ushap : {U.shape} \n OM : {omega} \n PROd : {np.dot(self.J, omega)}")
        omega = np.reshape(omega,(omega.shape[0], ))
        dis = np.reshape(dis,(dis.shape[0], ))
        x = np.reshape(self.ang_ddot_ref, (self.ang_ddot_ref.shape[0], ))
        
        alpha = np.dot(beta, (np.cross(-omega, np.dot(self.J, omega)) + dis)) + np.dot(self.R_dot, omega) 
        # alpha = np.dot(beta, (np.cross(-omega, np.dot(self.J, omega)) + dis)) + np.dot(self.R_dot, omega) - x

        Teta_param = {"alp": alpha, "U": U}
        initC = np.array([self.ang, self.ang_dot])
        # initC = np.array([self.err1, self.err2])
        initC = np.reshape(initC, (initC.shape[0], initC.shape[1]))
        # print(f"\nKai_param  :\n {Kai_param} \nKbar_param  :\n {Kbar_param}")
        ###########
        Y = RK4(self.Teta_fn, 0, initC, self.h, Teta_param)
        ##########
        ang_dot_pub = np.reshape(Y[1, :].T, ((Y[1, :].T).shape[0], 1))
        ang_pub = np.reshape(Y[0, :].T, ((Y[0, :].T).shape[0], 1))
        # ang_dot_pub = np.reshape(Y[1, :].T, ((Y[1, :].T).shape[0], 1)) + self.ang_dot_ref
        # ang_pub = np.reshape(Y[0, :].T, ((Y[0, :].T).shape[0], 1))     + self.ang_ref

        ########### Thrust Force(in Kgf) calculation                  
        thrust = self.m/(np.cos(ang_pub[1])*np.cos(ang_pub[0]))       ## considering Z-world vel& Accl.=0
        thrust_norm = (thrust-1.5)*(0.32/1.35) + 0.72               ## how to ensure the mapping??
        # thrust_norm = thrust_norm if thrust_norm<0.85 else 0.85

        self.t = self.t + self.h  
        R_inv = self.rotMat(ang_pub, 1)
        ang_dot_pub  =  np.dot(R_inv, ang_dot_pub)          ## angular velocity in body reference frame
        final = {'time':self.t, 'ang_vel':ang_dot_pub.tolist(), 'ang':ang_pub.tolist(), 'thrust':thrust_norm, 'torq':Tau.tolist(), \
                 'a_ref':self.ang_ref.tolist(), 'adot_ref':self.ang_dot_ref.tolist(), 'TwistStat':states.tolist() , 'kai1':kai1_inst.tolist(), 'disturb':dis.tolist()}
        # print(f"final_output : {final}")
        # print("******** end **********")
        return final

        