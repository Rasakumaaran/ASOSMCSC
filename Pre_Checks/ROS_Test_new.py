#!/usr/bin/env python2

#
# The shebang of this file is currently Python2 because some
# dependencies such as pymavlink don't play well with Python3 yet.
from __future__ import division

PKG = 'px4'

import rospy
import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from mavros_msgs.msg import AttitudeTarget
from Project_Pack.msg import ComputedValues
from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header
from threading import Thread
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion

# from SMC_ctrlr import controller
from SMC_ctrlr_Bnd import controller

class MavrosOffboardAttctlTest(MavrosTestCommon):
    
    def setUp(self):
        super(MavrosOffboardAttctlTest, self).setUp()

        self.att = AttitudeTarget()
        self.att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

        self.comp = ComputedValues()
        self.comp_pub = rospy.Publisher('CtrlOut', ComputedValues, queue_size=10)

        # send setpoints in separate thread to better prevent failsafe
        self.att_thread = Thread(target=self.send_att, args=())
        self.att_thread.daemon = True
        self.att_thread.start()

    def tearDown(self):
        super(MavrosOffboardAttctlTest, self).tearDown()

    #
    # Helper methods
    #
    def send_att(self):
        
        ctrl = controller()

        # timeout = 9  # (int) seconds
        # t = 0
        loop_freq = 40  # Hz
        rate = rospy.Rate(loop_freq)
        
        # while(t < timeout):
        while not rospy.is_shutdown():
            imu_msg = self.imu_data
            # rospy.loginfo("imu_msg inside loop: {} \n type: {}".format(imu_msg, type(imu_msg)))
            quaternion_list = [imu_msg.orientation.x, imu_msg.orientation.y,
                               imu_msg.orientation.z, imu_msg.orientation.w]
            ang_list = euler_from_quaternion(quaternion_list)       ## these are in radians, rad/sec
            ang_vel_list = [imu_msg.angular_velocity.x,
                            imu_msg.angular_velocity.y, imu_msg.angular_velocity.z]
            states = np.array([[ang_list], [ang_vel_list]])
            final_op = ctrl.controller(states, loop_freq)
            
            self.att.header = Header()
            self.att.header.frame_id = "base_footprint"
            rates  =  final_op.get('ang_vel')
            self.att.body_rate.x = rates[0][0]
            self.att.body_rate.y = rates[1][0]
            self.att.body_rate.z = rates[2][0]
            angle = final_op.get('ang')
            self.att.orientation = Quaternion(*quaternion_from_euler(angle[0][0], angle[1][0], angle[2][0]))
            self.att.thrust = final_op.get('thrust')
            self.att.type_mask = 7  # ignore body rate
            # att.type_mask        = 128         ##  ignore attitude - acro
            self.att.header.stamp = rospy.Time.now()
            self.att_setpoint_pub.publish(self.att)

            angRef = np.rad2deg(final_op.get('a_ref'))
            angle = np.rad2deg(angle)
            torq = final_op.get('torq')
            angdotRef = final_op.get('adot_ref')
            sigma = final_op.get('sigma')
            kai1 = final_op.get('err1')
            self.comp.Torq.x, self.comp.Torq.y, self.comp.Torq.z = torq[0], torq[1], torq[2]
            self.comp.KaiErr.x, self.comp.KaiErr.y, self.comp.KaiErr.z = kai1[0], kai1[1], kai1[2]
            self.comp.Sigma.x, self.comp.Sigma.y, self.comp.Sigma.z = sigma[0], sigma[1], sigma[2]
            self.comp.AngRef.x, self.comp.AngRef.y, self.comp.AngRef.z = angRef[0][0], angRef[1][0], angRef[2][0]
            self.comp.AngComp.x, self.comp.AngComp.y, self.comp.AngComp.z = angle[0], angle[1], angle[2]
            self.comp.AngVelRef.x, self.comp.AngVelRef.y, self.comp.AngVelRef.z = angdotRef[0][0], angdotRef[1][0], angdotRef[2][0]
            self.comp.Kbar = final_op.get('kbar')
            self.comp.Time = final_op.get('time')
            self.comp_pub.publish(self.comp)

            # template = "{{'time': {}, 'ang_vel':{}, 'ang': {}, 'Kai1': {}, 'Torq': {}, 'Kbar': {}, 'sigma': {} }}".format(
            #             final_op['time'], final_op['ang_vel'], final_op['ang'], final_op['err1'], final_op['torq'], final_op['kbar'], final_op['sigma'])  
            template = "{{'time': {}, 'ang': {}, 'Torq': {}, 'Kbar': {} }}".format(final_op['time'], final_op['ang'], final_op['torq'], final_op['kbar'] )          
            rospy.loginfo(template)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    #
    # Test method
    #
    def test_attctl(self):
        """Test offboard attitude control"""
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 10, -1)
        # self.log_topic_vars()

        self.set_mode("OFFBOARD", 10)
        self.set_arm(True, 10)

        timeout = 9  # (int) seconds
        t = 0
        loop_freq = 300  # Hz
        rate = rospy.Rate(loop_freq)
        
        # while not rospy.is_shutdown():
        while(t < timeout):
            t += 1/loop_freq
            rospy.loginfo("============ Time from test_attctl : {} ============".format(t))
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        
        self.set_mode("AUTO.LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 90, 0)
        self.set_arm(False, 5)


if __name__ == '__main__':
    import rostest
    rospy.init_node('test_node', anonymous=True)

    rostest.rosrun(PKG, 'mavros_offboard_attctl_test',
                   MavrosOffboardAttctlTest)
