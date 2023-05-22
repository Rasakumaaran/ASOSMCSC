import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import s
from solveODE import RK4

from tf.transformations import euler_from_quaternion

class controller:
    def __init__(self):
        self.h = 1/500     # sampling rate(in Sec)
        self.dim = 3       # system dimension
        self.t = 0         # initial time
        self.radii = 0.1  #radius to track

        self.ang = np.zeros((self.dim, 1))  # Euler angles (x1)
        self.ang_dot = np.zeros((self.dim, 1))  # ang. Vel. in world frame (x2)
        self.omega = np.zeros((self.dim, 1))  # ang. Vel. in body frame
        self.err1 = np.zeros((self.dim, 1))  # error angles
        self.err2 = np.zeros((self.dim, 1))  # error ang_vel
        self.tau1 = np.zeros((self.dim, 1))
        self.tau2 = np.zeros((self.dim, 1))
        self.kai2 = np.zeros((self.dim, 1))
        self.kai1 = np.zeros((self.dim, 1))  # unconstarained variable error(tau-tau0)
        self.kai_intg = np.zeros((self.dim, 1))  # integration of kai in slide variable
        self.sigma = np.zeros((self.dim, 1))  # sliding variable
        self.Kbar = []  # adap gain
        self.U = np.zeros((self.dim, 1))
        self.errors = np.zeros((self.dim, 2, 1))
        self.statespace = np.zeros((self.dim, 2, 1))
        
        ###########
        # Upper & Lower limits
        self.UL = np.ones((3, 1))*(np.pi*4/9)
        self.LL = -self.UL  

        # constants in ctrl law
        self.om1 = 1.915*np.eye(self.dim)
        self.om2 = 12.8*self.om1
        self.eps = 9.2
        self.eta = 0.2
        self.Gama = 0.5*self.om1
        self.rho = 0.001
        self.Kbar = np.append(self.Kbar, 0.01)
        # self.J = np.diag([0.009, 0.009, 0.016])  # Polar MOI in body frame ----- Tsalla
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

    def Kai_fn(self,_, x, Kai_param):
        alpha = Kai_param.get("alp")
        g = Kai_param.get("g")
        H = Kai_param.get("H")
        tau0_ddot = Kai_param.get("tdd")
        U = Kai_param.get("U")

        self.Kai = np.zeros((2, 3))
        #print('\n H:\n', H, '\n U:\n', U, '\n tdd:\n',tau0_ddot, '\n VecProd:\n', np.dot(g, alpha))
        self.Kai[0, :] = x[1, :]
        temp = np.dot(g, alpha) + H + U - tau0_ddot
        self.Kai[1, :] = temp.T
        return self.Kai

    def transformation(self,Lamda, tau1):
        psi = np.divide(np.exp(tau1), (np.exp(tau1)+1))  # sigmoid fun
        # derivative of sigmoid fun
        psid = np.divide(np.exp(tau1), np.power((np.exp(tau1)+1), 2))
        psid = np.reshape(psid,(psid.shape[0],))
        psi_dot = np.diag(psid)
        # print(f"\npsidot:{psi_dot}")
        g = np.linalg.inv(np.dot(Lamda, psi_dot))

        output = {"g": g, "psi": psi, "psi_dot": psi_dot}
        return output

    def rotMat(self,ang,inverse):
        R = [[1, np.sin(ang[0])*np.tan(ang[1]), np.cos(ang[0])*np.tan(ang[1])],
            [0, np.cos(ang[0]), -np.sin(ang[0])],
            [0, np.sin(ang[0])/np.cos(ang[1]), np.cos(ang[0])/np.cos(ang[1])]]
        R = np.array(R, dtype=np.float)
        if (inverse==1):
            R = np.linalg.inv(R)
            
        return R


    def controller(self, states):  # state's dim (2,3) [angle,ang_vel];; store err,ang

        self.ang = states[0, :]                     ## ensure angles are RPY w.r.t world frame
        self.ang = self.ang.T
        self.ang_dot = states[1, :]
        self.ang_dot = self.ang_dot.T

        self.R = self.rotMat(self.ang, 0)
        self.ang_dot = np.dot(self.R, self.ang_dot)    ## converted to world frame coords

        self.ang_ref = np.array([[self.radii*np.sin(self.t)], [self.radii*np.cos(self.t)], [0.2]])
        self.ang_dot_ref = np.array([[self.radii*np.cos(self.t)], [-self.radii*np.sin(self.t)], [0]])
        self.ang_ddot_ref = np.array([[-self.radii*np.sin(self.t)], [-self.radii*np.cos(self.t)], [0]])

        # dis = np.array([[2*np.sin(self.t)], [np.cos(self.t)],[0.5*(np.cos(self.t)+np.sin(self.t))]])  # disturbance
        # dis = np.array([[2*np.sin(self.t)], [np.cos(self.t)],[0.5*(np.cos(self.t)+np.sin(self.t))]])*0  # zero disturbance
        dis = np.array([[0], [0],[0]])

        self.Ulim = self.UL - self.ang_ref
        self.Llim = self.LL - self.ang_ref
        
        x = (self.Ulim - self.Llim)
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

        self.tau2 = np.dot(g, (self.err2 + self.ang_dot_ref))
        self.kai1 = np.concatenate((self.kai1, (self.tau1 - self.tau0)), axis=1)
        self.kai2 = np.concatenate((self.kai2, (self.tau2 - self.tau0_dot)), axis=1)
        # print(f"\ntu1Shap: {np.shape(self.tau1)},  \ntu2Shap: {np.shape(self.tau2)} \nki1Shap: {np.shape(self.kai1)}  \nki2Shap: {np.shape(self.kai2)}")
        # print(f"t = {self.t}")
        if(self.t == 0):
            self.Rold = self.R
        self.R_dot = (self.R - self.Rold)/self.h
        self.Rold = self.R
        R_inv = self.rotMat(self.ang, 1)

        beta = np.dot(self.R, np.linalg.inv(self.J))

        omega = np.dot(R_inv, self.ang_dot)

        # computing Sliding variable
        # trapezoidal integration
        if (self.t == 0):
            temp = self.h*self.kai1[:, -1]/2
        elif (self.t == self.h):
            temp = self.h*(self.kai1[:, -1] + self.kai1[:, -2])/2
        else:
            temp = (self.kai1[:, -1] + 2/self.h*self.kai_intg[:, -1] + self.kai1[:, -2])*self.h/2
        temp = np.reshape(temp,(temp.shape[0],1))
        self.kai_intg = np.concatenate((self.kai_intg, temp), axis=1)

        kai_diff = self.kai2[:, -1]

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

        U = -(np.dot(self.Gama, sigma) + self.Kbar[-1]*fNom*sat +
            np.dot(self.om1, kai_diff) + np.dot(self.om2, self.kai1[:, -1]) + H)
        # print(f"\n Ushap : {U.shape} \n OM : {omega} \n PROd : {np.dot(self.J, omega)}")
        omega = np.reshape(omega,(omega.shape[0], ))
        dis = np.reshape(dis,(dis.shape[0], ))
        x = np.reshape(self.ang_ddot_ref, (self.ang_ddot_ref.shape[0], ))
        
        alpha = np.dot(beta, (np.cross(-omega, np.dot(self.J, omega)) + dis)
                    ) + np.dot(self.R_dot, omega) - x

        x = np.reshape( self.tau0_ddot,(self.tau0_ddot.shape[0], ))
        Kai_param = {"alp": alpha, "g": g, "H": H, "tdd": x, "U": U}
        # print(f"\nKai_param  :\n {Kai_param} \nKbar_param  :\n {Kbar_param}")
        ###########
        Y = RK4(self.Kai_fn, 0, np.array([self.kai1[:, -1], self.kai2[:, -1]]), self.h, Kai_param)
        ##########
        
        self.tau1 = self.tau0 + np.reshape(Y[0, :].T, ((Y[0, :].T).shape[0], 1))
        self.tau2 = self.tau0_dot + np.reshape(Y[1, :].T, ((Y[1, :].T).shape[0], 1))
        # print(f" \n Y  :  {Y} \n tu1 :  {self.tau1}  \n tu2 :  {self.tau2}")
        
        trans = self.transformation(Lamda, self.tau1)
        g = trans.get("g")
        psi = trans.get("psi")

        temp2 = np.dot(np.linalg.inv(g), self.tau2) + (-self.ang_dot_ref)
        temp1 = self.Llim + np.dot(Lamda, psi)
        ang_dot_pub = temp2 + self.ang_dot_ref
        ang_pub = temp1 + self.ang_ref

        ########### Thrust Force(in Kgf) calculation  
        thrust = self.m*(np.cos(ang_pub[1])*np.cos(ang_pub[0]) + (ang_dot_pub[1]/ang_dot_pub[2])*np.sin(ang_pub[0])*np.cos(ang_pub[1]) \
                                - (ang_dot_pub[0]/ang_dot_pub[2])*np.sin(ang_pub[1]))
        thrust_norm = (thrust-1.5)*(0.32/1.35) + 0.72               ##  {linear interpolation} 
        ## try saturation
        
        self.t = self.t + self.h  
        # store error, angles commanded&done
        R_inv = self.rotMat(ang_pub, 1)
        ang_dot_pub  =  np.dot(R_inv, ang_dot_pub)          ## angular velocity in body reference frame
        final = {'ang_vel':np.array(ang_dot_pub), 'ang':np.array(ang_pub), 'thrust':thrust_norm}
        # print(f"final_output : {final}")
        # print("******** end **********")
        return final

        