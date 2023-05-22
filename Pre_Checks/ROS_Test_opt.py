#!/usr/bin/env python3

import numpy as np
import sys
import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from std_msgs.msg import *
from mavros_msgs.srv import *
from project_package.msg import ComputedValues
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time

from SMC_ctrlr_Bnd_opt import controller

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
    att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=20)
    comp_pub = rospy.Publisher('CtrlOut', ComputedValues, queue_size=20)
    
    # Initialize subscriber 
    rospy.Subscriber("/mavros/state",State, stateMt.stateCb)
    rospy.Subscriber("/mavros/imu/data", Imu, stateMt.imuCb)
    # rospy.Subscriber("~/WindPubTopic", WindPubTopic, stateMt.DisturbCb)

    # Specify the rate 
    h = 400
    rate = rospy.Rate(h)

    # initialize msg values
    act_ctrl = ActuatorControl()
    # act_ctrl.group_mix=0
    # act_ctrl.controls
    att = AttitudeTarget()
    att.body_rate = Vector3()
    att.header = Header()
    att.header.frame_id = "base_footprint"
    att.orientation = Quaternion(*quaternion_from_euler(0.01, -0.01, 0.01))
    att.thrust = 0.4
    att.type_mask   = 7     #only attitude
    att.body_rate.x, att.body_rate.y, att.body_rate.z = 0.0, 0.0, 0.0
    comp = ComputedValues()

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

    t1 = time.perf_counter()
    t= 0 
    
    # Publish the setpoints 
    while(not rospy.is_shutdown() ) :
        if(time.perf_counter()-t1 <= 3):
            ang_list = [0.0, 0.0, 0.0]
            att.orientation = Quaternion(*quaternion_from_euler(ang_list[0], ang_list[1], ang_list[2]))
            att.thrust = 0.4
            att.type_mask   = 7     #only attitude
            att.header.stamp = rospy.Time.now()
            att_setpoint_pub.publish(att)
            
        else:            
            quaternion_list = [stateMt.imu.orientation.x, stateMt.imu.orientation.y,
                            stateMt.imu.orientation.z, stateMt.imu.orientation.w]
            ang_list = euler_from_quaternion(quaternion_list)       ## these are in radians
            ang_vel_list = [stateMt.imu.angular_velocity.x,
                            stateMt.imu.angular_velocity.y, stateMt.imu.angular_velocity.z]
            states = np.array([ang_list, ang_vel_list])
            final_op = ctrl.controller(states, h)
            
            angle = final_op.get('ang')
            att.orientation = Quaternion(*quaternion_from_euler(angle[0], angle[1], angle[2]))
            rates  =  final_op.get('ang_vel')
            att.body_rate.x, att.body_rate.y, att.body_rate.z = rates[0], rates[1], rates[2]
            att.thrust    = final_op.get('thrust')         # Normalised thrust (0.7-grnd level) can't ignore it. it won't takeoff
            att.type_mask   = 7     #only attitude        
            # att.type_mask = 128  # only body rate
            att.header.stamp = rospy.Time.now()
            att_setpoint_pub.publish(att)

            angRef = np.rad2deg(final_op.get('a_ref'))
            angle = np.rad2deg(angle)
            torq = final_op.get('torq')
            angdotRef = final_op.get('adot_ref')
            sigma = final_op.get('sigma')
            kai1 = final_op.get('err1')
            
            comp.Torq.x, comp.Torq.y, comp.Torq.z = torq[0], torq[1], torq[2]
            comp.KaiErr.x, comp.KaiErr.y, comp.KaiErr.z = kai1[0], kai1[1], kai1[2]
            comp.Sigma.x, comp.Sigma.y, comp.Sigma.z = sigma[0], sigma[1], sigma[2]
            comp.AngRef.x, comp.AngRef.y, comp.AngRef.z = angRef[0], angRef[1], angRef[2]
            comp.AngComp.x, comp.AngComp.y, comp.AngComp.z = angle[0], angle[1], angle[2]
            comp.AngVelRef.x, comp.AngVelRef.y, comp.AngVelRef.z = angdotRef[0], angdotRef[1], angdotRef[2]
            comp.Kbar = final_op.get('kbar')
            comp.Time = final_op.get('time')
            comp_pub.publish(comp)
            
            template = "{{'Sys_time': {}, 'Ctrl_time': {}, 'ang_vel':{}, 'Sigma': {}, 'Kbar': {} }}".format(
                (time.perf_counter()-t1), final_op['time'], final_op['ang_vel'], final_op['sigma'], final_op['kbar'])          
            rospy.loginfo(template)
            # if(any(np.abs(item)>150 or np.isnan(item) for item in rates)):
            #     raise Exception("Out of limit.")
            
            # radii = 0.1
            # angle = np.array([[radii*np.sin(t)], [radii*np.cos(t)], [0.2]])
            # att.orientation = Quaternion(*quaternion_from_euler(angle[0][0], angle[1][0], angle[2][0]))
            # att.header.stamp = rospy.Time.now()
            # att_setpoint_pub.publish(att)
            # t+=1/h

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
        




# Torq_pub = rospy.Publisher('TorqVals', Vector3, queue_size=10)  
# ARef_pub = rospy.Publisher('AngRefVals', Vector3, queue_size=10)
# AComp_pub = rospy.Publisher('AngCompVals', Vector3, queue_size=10)
# AdRef_pub = rospy.Publisher('AngdotRefVals', Vector3, queue_size=10)  

# Kbar_pub = rospy.Publisher('KbarVals', Vector3, queue_size=10)      #Float32 - high cpu load
# Sigma_pub = rospy.Publisher('SigmaVals', Vector3, queue_size=10)  
# Kai1_pub = rospy.Publisher('Kai1Vals', Vector3, queue_size=10)  

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
