#!/usr/bin/env python3


'''
This is a boiler plate script that contains hint about different services that are to be used
to complete the task.
Use this code snippet in your code or you can also continue adding your code in the same file


This python file runs a ROS-node of name offboard_control which controls the drone in offboard mode. 
See the documentation for offboard mode in px4 here() to understand more about offboard mode 
This node publishes and subsribes the following topics:

	 Services to be called                   Publications                                          Subscriptions				
	/mavros/cmd/arming                       /mavros/setpoint_position/local                       /mavros/state
    /mavros/set_mode                         /mavros/setpoint_velocity/cmd_vel                     /mavros/local_position/pose   
         
    
'''

import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *


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
            print ("Service arming call failed: %s"%e)

        # Similarly declare other service proxies 

    def setDisarm(self):
        # Calling to /mavros/cmd/arming to arm the drone and print fail message on failure
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException as e:
            print ("Service disarming call failed: %s"%e)
   
    def offboard_set_mode(self,mode):
        # Call /mavros/set_mode to set the mode the drone to OFFBOARD and print fail message on failure
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode=mode)
        except rospy.ServiceException as e:
            print ("service set_mode call failed: %s"%e)
   
class stateMoniter:
    def __init__(self):
        self.state = State()
        self.pos = PoseStamped()
        self.vel = TwistStamped()
        # Instantiate a setpoints message

        
    def stateCb(self, msg):
        # Callback function for topic /mavros/state
        self.state = msg

    # Create more callback functions for other subscribers
    def posCb(self, msg):
        # Callback function for topic /mavros/local_position/pose
        self.pos = msg

def main():


    stateMt = stateMoniter()
    ofb_ctl = offboard_control()

    # Initialize publishers
    local_pos_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
    local_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    # Specify the rate 
    rate = rospy.Rate(20.0)

    # Make the list of setpoints 
    setpoints = [[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 10.0]] #List to setpoints
    setpoints1 = [0.0, 0.0, 0.0]


    # Similarly initialize other publishers 

    # Create empty message containers 
    pos =PoseStamped()
    pos.pose.position.x = 0
    pos.pose.position.y = 0
    pos.pose.position.z = 0

    # Set your velocity here
    vel = TwistStamped()
    vel.twist.linear.x = 15
    vel.twist.linear.y = 15
    vel.twist.linear.z = 15
    
    # Similarly add other containers 

    # Initialize subscriber 
    rospy.Subscriber("/mavros/state",State, stateMt.stateCb)

    # Similarly initialize other subscribers  
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, stateMt.posCb)

    '''
    NOTE: To set the mode as OFFBOARD in px4, it needs atleast 100 setpoints at rate > 10 hz, so before changing the mode to OFFBOARD, send some dummy setpoints  
    '''
    
    # Send some dummy setpoints before starting offboard mode
    for _ in range(100):
        local_pos_pub.publish(pos)
        rate.sleep()

    # Switching the state to auto mode
    while not stateMt.state.mode=="OFFBOARD":
        ofb_ctl.offboard_set_mode("OFFBOARD")
        rate.sleep()
    print ("OFFBOARD mode activated")

    # Arming the drone
    while not stateMt.state.armed:
        ofb_ctl.setArm()
        rate.sleep()
    print("Armed!!")

    # Publish the setpoints 
    while not rospy.is_shutdown():
        '''
        Step 1: Set the setpoint 
        Step 2: Then wait till the drone reaches the setpoint, 
        Step 3: Check if the drone has reached the setpoint by checking the topic /mavros/local_position/pose 
        Step 4: Once the drone reaches the setpoint, publish the next setpoint , repeat the process until all the setpoints are done  


        Write your algorithm here 
        '''
        thresh=0.5
        count = 0
        while count<2:        
            for i in range(len(setpoints)):
                pos.pose.position.x = setpoints[i][0]
                pos.pose.position.y = setpoints[i][1]
                pos.pose.position.z = setpoints[i][2]
                while not (abs(stateMt.pos.pose.position.x - setpoints[i][0]) < thresh and abs(stateMt.pos.pose.position.y - setpoints[i][1]) < thresh and abs(stateMt.pos.pose.position.z - setpoints[i][2]) < thresh):
                    local_pos_pub.publish(pos)
                    local_vel_pub.publish(vel)
                    rate.sleep()
                    
                #print("Reached setpoint")
            print("All setpoints reached")

            count+=1
        #Disarm the drone
        # pos.pose.position.x = setpoints1[0]
        # pos.pose.position.y = setpoints1[1]
        # pos.pose.position.z = setpoints1[2]
        ofb_ctl.offboard_set_mode("RTL")
        while stateMt.state.armed:
            ofb_ctl.setDisarm()
            rate.sleep()
        print("Disarmed!!")
        break     

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass