#!/usr/bin/env python3

import numpy as np
# import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from std_msgs.msg import *
from mavros_msgs.srv import *
from Project_Pack.msg import ComputedValues

from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import time

# from SMC_ctrlr import controller
from SMC_ctrlr_Bnd import controller


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

    ctrl = controller()

    # Initialize publishers
    # In OffBoard Mode only AttitudeTarget/set local/global position messages are only accepted... one message at a time
    att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
    comp_pub = rospy.Publisher('CtrlOut', ComputedValues, queue_size=10)
    # Torq_pub = rospy.Publisher('TorqVals', Vector3, queue_size=10)  
    # ARef_pub = rospy.Publisher('AngRefVals', Vector3, queue_size=10)
    # AComp_pub = rospy.Publisher('AngCompVals', Vector3, queue_size=10)
    # AdRef_pub = rospy.Publisher('AngdotRefVals', Vector3, queue_size=10)  
    
    # Kbar_pub = rospy.Publisher('KbarVals', Vector3, queue_size=10)      #Float32 - high cpu load
    # Sigma_pub = rospy.Publisher('SigmaVals', Vector3, queue_size=10)  
    # Kai1_pub = rospy.Publisher('Kai1Vals', Vector3, queue_size=10)  
    # Initialize subscriber 
    rospy.Subscriber("/mavros/state",State, stateMt.stateCb)
    rospy.Subscriber("/mavros/imu/data", Imu, stateMt.imuCb)
    # rospy.Subscriber("~/WindPubTopic", WindPubTopic, stateMt.DisturbCb)

    # Specify the rate 
    h = 400
    rate = rospy.Rate(h)

    # initialize msg values
    att = AttitudeTarget()
    att.body_rate = Vector3()
    att.header = Header()
    att.header.frame_id = "base_footprint"
    att.orientation = Quaternion(*quaternion_from_euler(0.09, -0.1, -0.1))
    att.thrust = 0.5
    att.body_rate.x, att.body_rate.y, att.body_rate.z = 0.0, 0.0, 0.0
    att.type_mask   = 7     #only attitude        
    # att.type_mask = 128  # only body rate
    comp = ComputedValues()

    # Torqmsg, ARefmsg, ACompmsg, AdotRefmsg, Kai1msg = Vector3(), Vector3(), Vector3(), Vector3(), Vector3()
    # Kbarmsg, Sigmamsg  = Vector3(), Vector3()

    '''
    NOTE: To set the mode as OFFBOARD in px4, it needs atleast 100 setpoints at rate > 10 hz, so before changing the mode to OFFBOARD, send some dummy setpoints  
    '''
    
    # Send some dummy setpoints before starting offboard mode
    for _ in range(100):
        att_setpoint_pub.publish(att)
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

    # Publish the setpoints 
    while(not rospy.is_shutdown() ) :
        '''
        Step 1: Set the setpoint 
        Step 2: Then wait till the drone reaches the setpoint, 
        Step 3: Check if the drone has reached the setpoint by checking the topic /mavros/local_position/pose 
        Step 4: Once the drone reaches the setpoint, publish the next setpoint , repeat the process until all the setpoints are done  
        '''
        # t1 = time.perf_counter()
        quaternion_list = [stateMt.imu.orientation.x,
                           stateMt.imu.orientation.y,
                           stateMt.imu.orientation.z,
                           stateMt.imu.orientation.w]
        ang_list = euler_from_quaternion(quaternion_list)       ## these are in radians, rad/sec
        ang_vel_list = [stateMt.imu.angular_velocity.x,
                        stateMt.imu.angular_velocity.y, stateMt.imu.angular_velocity.z]
        states = np.array([[ang_list], [ang_vel_list]])
        # print(f"states from main():  {states}")

        # final_op = ctrl.controller(states)
        final_op = ctrl.controller(states, h)
        angle = final_op.get('ang')
        att.orientation = Quaternion(*quaternion_from_euler(angle[0][0], angle[1][0], angle[2][0]))
        att.header.frame_id = "base_footprint"

        rates  =  final_op.get('ang_vel')
        att.body_rate.x = rates[0][0]
        att.body_rate.y = rates[1][0]
        att.body_rate.z = rates[2][0]
        
        att.thrust    = final_op.get('thrust')         # Normalised thrust (0.7-grnd level) can't ignore it. it won't takeoff
        att.type_mask        = 7         ##  ignore all rates  {in binary(111)}
        # att.type_mask        = 128         ##  ignore attitude - acro

        att.header.stamp = rospy.Time.now()
        att_setpoint_pub.publish(att)

        angRef = np.rad2deg(final_op.get('a_ref'))
        angRef = np.reshape(angRef, (np.shape(angRef)[0],))
        angle = np.rad2deg(angle)
        angle = np.reshape(angle, (np.shape(angle)[0],))
        torq = final_op.get('torq')
        torq = np.reshape(torq, (np.shape(torq)[0],))
        angdotRef = final_op.get('adot_ref')
        angdotRef = np.reshape(angdotRef, (np.shape(angdotRef)[0],))
        sigma = final_op.get('sigma')
        sigma = np.reshape(sigma, (np.shape(sigma)[0],))
        kai1 = final_op.get('err1')
        kai1 = np.reshape(kai1, (np.shape(kai1)[0],))
        comp.Torq.x, comp.Torq.y, comp.Torq.z = torq[0], torq[1], torq[2]
        comp.KaiErr.x, comp.KaiErr.y, comp.KaiErr.z = kai1[0], kai1[1], kai1[2]
        comp.Sigma.x, comp.Sigma.y, comp.Sigma.z = sigma[0], sigma[1], sigma[2]
        comp.AngRef.x, comp.AngRef.y, comp.AngRef.z = angRef[0], angRef[1], angRef[2]
        comp.AngComp.x, comp.AngComp.y, comp.AngComp.z = angle[0], angle[1], angle[2]
        comp.AngVelRef.x, comp.AngVelRef.y, comp.AngVelRef.z = angdotRef[0], angdotRef[1], angdotRef[2]
        comp.Kbar = final_op.get('kbar')
        comp.Time = final_op.get('time')
        comp_pub.publish(comp)

        # angRef = np.rad2deg(final_op.get('a_ref'))
        # angle = np.rad2deg(angle)
        # torq = final_op.get('torq')
        # angdotRef = final_op.get('adot_ref')
        # sigma = final_op.get('sigma')
        # kai1 = final_op.get('err1')
        
        # Torqmsg.x, Torqmsg.y, Torqmsg.z = torq[0], torq[1], torq[2]
        # Kai1msg.x, Kai1msg.y, Kai1msg.z = kai1[0], kai1[1], kai1[2]
        # Sigmamsg.x, Sigmamsg.y, Sigmamsg.z = sigma[0], sigma[1], sigma[2]
        # Kbarmsg.x, Kbarmsg.y, Kbarmsg.z = final_op.get('kbar'), 0, 0
        # ARefmsg.x, ARefmsg.y, ARefmsg.z = angRef[0][0], angRef[1][0], angRef[2][0]
        # ACompmsg.x, ACompmsg.y, ACompmsg.z = angle[0], angle[1], angle[2]
        # AdotRefmsg.x, AdotRefmsg.y, AdotRefmsg.z = angdotRef[0][0], angdotRef[1][0], angdotRef[2][0]
        # Torq_pub.publish(Torqmsg)
        # ARef_pub.publish(ARefmsg) 
        # AComp_pub.publish(ACompmsg) 
        # AdRef_pub.publish(AdotRefmsg) 

        # Kbar_pub.publish(Kbarmsg)
        # Sigma_pub.publish(Sigmamsg)
        # Kai1_pub.publish(Kai1msg)


        # T = final_op.get('time')
        # count += 1
        # T = (t2-t1)/count
        # freq = 1/T
        # print(freq)
        # # del_t = time.perf_counter() - t1
        
        template = "{{'time': {}, 'ang_vel':{}, 'ang': {}, 'Kai1': {}, 'Torq': {}, 'Kbar': {}, 'sigma': {} }}".format(
            final_op['time'], final_op['ang_vel'], final_op['ang'], final_op['err1'], final_op['torq'], final_op['kbar'], final_op['sigma'])          
        rospy.loginfo(template)
        # rospy.loginfo(f'Time for this loop {time.perf_counter() - t1}')
        # print(final_op,"\nTime taken:",del_t)
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
        print("Ctrtl+C")
