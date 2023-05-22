#!/usr/bin/env python3

import numpy as np
import sys
import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from std_msgs.msg import *
from mavros_msgs.srv import *
from drone_project.msg import ComputedValues
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time


class offboard_control:

    def __init__(self):
        # Initialise rosnode
        rospy.init_node('offboard_control', anonymous=True)
  
    def setArm(self):
        # Calling to /mavros/cmd/arming to arm the drone and print fail message on failure
        rospy.wait_for_service('mavros/cmd/arming')  # Waiting untill the service starts 
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool) # Creating a proxy service for the rosservice named /mavros/cmd/arming for arming the drone 
            armService(True)
        except rospy.ServiceException as e:
            rospy.loginfo("Service arming call failed: %s"%e)

    # Similarly declare other service proxies 

    def setDisarm(self):
        # Calling to /mavros/cmd/arming to arm the drone and print fail message on failure
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException as e:
            rospy.loginfo("Service disarming call failed: %s"%e)
   
    def offboard_set_mode(self,mode):
        # Call /mavros/set_mode to set the mode the drone to OFFBOARD and print fail message on failure
        rospy.wait_for_service('mavros/set_mode', 5)
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            # flightModeService(custom_mode=mode)
            flightModeService(0, mode)
        except rospy.ServiceException as e:
            rospy.loginfo("service set_mode call failed: %s"%e)

   
class stateMoniter:
    def __init__(self):
        self.state = State()
        self.imu = Imu()
        
        # Instantiate a setpoints message
        
    def stateCb(self, msg):
        # Callback function for topic /mavros/state
        self.state = msg

    # Create more callback functions for other subscribers
    def imuCb(self, msg):
        # Callback function for topic /mavros/...
        self.imu = msg

def main():
    stateMt = stateMoniter()
    ofb_ctl = offboard_control()

    # Initialize publishers
    # In OffBoard Mode only AttitudeTarget/set local/global position messages are only accepted... one message at a time
    act_setpoint_pub = rospy.Publisher('mavros/actuator_control',ActuatorControl,queue_size=10)
    comp_pub = rospy.Publisher('CtrlOut', ComputedValues, queue_size=20)
    
    # Initialize subscriber 
    rospy.Subscriber("/mavros/state",State, stateMt.stateCb)
    # rospy.Subscriber("/mavros/imu/data", Imu, stateMt.imuCb)
    # rospy.Subscriber("~/WindPubTopic", WindPubTopic, stateMt.DisturbCb)

    # Specify the rate 
    h = 400
    rate = rospy.Rate(h)

    # initialize msg values
    act = ActuatorControl()
    act.header = Header()
    act.header.frame_id = "base_footprint"
    act.group_mix = 0
    act.controls = [0,0,0,0,0,0,0,0]
    comp = ComputedValues()

    '''
    NOTE: To set the mode as OFFBOARD in px4, it needs atleast 100 setpoints at rate > 10 hz, so before changing the mode to OFFBOARD, send some dummy setpoints  
    '''
    
    # Send some dummy setpoints before starting offboard mode
    for _ in range(100):
        act_setpoint_pub.publish(act)
        rate.sleep()

    # Switching the state to auto mode
    while not stateMt.state.mode=="OFFBOARD":
        ofb_ctl.offboard_set_mode("OFFBOARD") #this line not working
        # ofb_ctl.set_mode("OFFBOARD", 5)
        rate.sleep()
    rospy.loginfo("OFFBOARD mode activated")

    # Arming the drone
    while not stateMt.state.armed:
        ofb_ctl.setArm()
        # ofb_ctl.set_arm(True, 1)
        rate.sleep()
    rospy.loginfo("Armed Success!!")

    radii = 0.1
    t1 = 0
    J = np.diag([0.019, 0.019, 0.035])
    tau_cap = 2.21
    t0 = time.perf_counter()
    # Publish the setpoints 
    while(not rospy.is_shutdown() ) :
        ang_ref = [radii*np.sin(t1), radii*np.cos(t1), 0.2]
        ang_ref_d = np.rad2deg(ang_ref)
        ang_dot_ref = np.array([radii*np.cos(t1), -radii*np.sin(t1), 0.0])
        ang_ddot_ref = np.array([-radii*np.sin(t1), -radii*np.cos(t1), 0.0])
        
        R = [[1, np.sin(ang_ref[0])*np.tan(ang_ref[1]), np.cos(ang_ref[0])*np.tan(ang_ref[1])],
            [0, np.cos(ang_ref[0]), -np.sin(ang_ref[0])],
            [0, np.sin(ang_ref[0])/np.cos(ang_ref[1]), np.cos(ang_ref[0])/np.cos(ang_ref[1])]]
        R = np.array(R, dtype=np.float64)
        R_inv = np.linalg.inv(R)
        if(t1 == 0.0):
            Rold = R
        R_dot = (R - Rold)/h
        Rold = R
        omega_ref = R_inv @ ang_dot_ref

        tau = J @ R_inv @ ang_ddot_ref + np.cross(-omega_ref, np.dot(J, omega_ref)) - R_dot @ omega_ref
        comp.Torq.x, comp.Torq.y, comp.Torq.z = tau[0], tau[1], tau[2]           
        tau_norm = np.where(abs(tau)>tau_cap, tau_cap, tau)
        
        tau_norm = (2/7.8)*(tau_norm+3.9) - 1
        act.controls = [tau_norm[0], tau_norm[1], tau_norm[2], 0.4, 0,0,0,0]
        act_setpoint_pub.publish(act)

        comp.Time = t1
        comp.AngRef.x, comp.AngRef.y, comp.AngRef.z = ang_ref_d[0], ang_ref_d[1], ang_ref_d[2]
        comp.AngVelRef.x, comp.AngVelRef.y, comp.AngVelRef.z = ang_dot_ref[0], ang_dot_ref[1], ang_dot_ref[2]
        comp_pub.publish(comp)
        t1 += 1/h
        
        template = "{{'Sys_time': {}, 'Ctrl_time': {}, 'ang_ref':{}, 'Omega': {}, 'Tau':{} }}".format(
                (time.perf_counter()-t0), t1, ang_ref, omega_ref, tau )          
        rospy.loginfo(template)
        rate.sleep()
                    
    #Disarm the drone
    # ofb_ctl.offboard_set_mode("RTL")
    while stateMt.state.armed:
        ofb_ctl.setDisarm()
        rate.sleep()
    rospy.loginfo("Disarmed!!")    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("ROSInterrupt")
        sys.exit(0)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(0)       
    except Exception as e:
        print(e)
        sys.exit(0)
        