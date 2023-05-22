
#!/usr/bin/env python2
#***************************************************************************
#
#   Copyright (c) 2015 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#***************************************************************************/

#
# @author Andreas Antener <andreas@uaventure.com>
#
# The shebang of this file is currently Python2 because some
# dependencies such as pymavlink don't play well with Python3 yet.
from __future__ import division
import timeit

PKG = 'px4'
import time
import timeit
import rospy
import math
import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from mavros_msgs.msg import AttitudeTarget
from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header,Float32
from threading import Thread
from tf.transformations import quaternion_from_euler


class MavrosOffboardAttctlTest(MavrosTestCommon):
    """
    Tests flying in offboard control by sending attitude and thrust setpoints
    via MAVROS.

    For the test to be successful it needs to cross a certain boundary in time.
    """
    
    def setUp(self):
        super(MavrosOffboardAttctlTest, self).setUp()

        self.att = AttitudeTarget()

        self.att_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        self.custom_pub = rospy.Publisher(
            'custom_msgs', Vector3, queue_size=10)


        # send setpoints in separate thread to better prevent failsafe
        # self.att_thread = Thread(target=self.send_att, args=())
        # self.att_thread.daemon = True
        # self.att_thread.start()

    def tearDown(self):
        super(MavrosOffboardAttctlTest, self).tearDown()

    #
    # Test method
    #
    def test_attctl(self):
        """Test offboard attitude control"""
        rate = rospy.Rate(10)  # Hz
        self.att.body_rate = Vector3()
        self.att.header = Header()
        self.att.header.frame_id = "base_footprint"
        roll, pitch, yaw = -0.25, 0.15, 0.0
        # self.att.body_rate.x, self.att.body_rate.y, self.att.body_rate.z  = roll, pitch, yaw
        self.att.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))
        self.att.thrust = 0.71
        self.att.type_mask = 7  # ignore body rate
        # self.att.type_mask = 0
        
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,10, -1)

        for _ in range(100):
            self.att_setpoint_pub.publish(self.att)
            rate.sleep()
          
        self.log_topic_vars()
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)

        rospy.loginfo("run mission")
        
        loop_freq = 250  # Hz
        crossed = False
        loop_count = 0
        timeout = 15  # (int) seconds
        rate = rospy.Rate(loop_freq)
        count = timeout * loop_freq

        while not rospy.is_shutdown():
            
            t1 = time.time()
            # time.sleep(0.25)#to eliminate initial time delay
            for i in xrange(count):
                if(i>30 and i<600):
                    yaw   = 0.3 + 0.1*np.sin(i/loop_freq)
                    pitch = 0.08*np.cos(i/loop_freq)
                    roll  = -0.2
                else:
                    roll, pitch, yaw = 0.0, 0.0, 0.3
                    
                self.att.orientation = Quaternion(*quaternion_from_euler(roll, pitch,yaw))
                
                # self.att.type_mask = 56
                # self.att.type_mask = 0
                # yaw   = 0
                # pitch = 0.04*np.cos(i/loop_freq)
                # roll  = 0.08*np.sin(i/loop_freq)
                # self.att.body_rate.x = roll
                # self.att.body_rate.y = pitch
                # self.att.body_rate.z = yaw
                
                # rospy.loginfo(f'Current orientation is:{yaw}')
                self.att.header.stamp = rospy.Time.now()
                
                self.custom_msg_angles = Vector3()
                self.custom_msg_angles.x =  np.rad2deg(roll)
                self.custom_msg_angles.y =  np.rad2deg(pitch)
                self.custom_msg_angles.z =  np.rad2deg(yaw)
                # self.custom_msg_angles.x, self.custom_msg_angles.y, self.custom_msg_angles.z = \
                    # self.att.body_rate.x, self.att.body_rate.y, self.att.body_rate.z
                self.att_setpoint_pub.publish(self.att)
                self.custom_pub.publish(self.custom_msg_angles)
                rospy.loginfo("i : {0}".format(i))
                rate.sleep()
                if (i == count-1):
                    rospy.loginfo("Time Reached | seconds: {0} of {1}".format(
                        i / loop_freq, timeout))
                    crossed = True      
                    break
            loop_count += 1
            t2 = time.time()
            rospy.loginfo("freq = {0}".format(1/(t2-t1)))

            t1 = time.perf_counter()
            self.att_setpoint_pub.publish(self.att)
            t2 = time.perf_counter()
            self.custom_pub.publish(self.custom_msg_angles)
            t3 = time.perf_counter()
            rospy.loginfo("AttiTarget{0},  CustomMsg{1}".format((t2-t1),(t3-t2)))
            
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)
        
        self.assertTrue(crossed, (
            "took too long to cross boundaries | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout(seconds): {3}".
            format(self.local_position.pose.position.x,
                   self.local_position.pose.position.y,
                   self.local_position.pose.position.z, timeout)))

        self.set_mode("AUTO.LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,90, 0)
        self.set_arm(False, 5)


if __name__ == '__main__':
    import rostest
    rospy.init_node('test_node', anonymous=True)

    rostest.rosrun(PKG, 'mavros_offboard_attctl_test',
                   MavrosOffboardAttctlTest)

