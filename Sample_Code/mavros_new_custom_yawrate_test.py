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

PKG = 'px4'
import time 
import rospy
import math
import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from mavros_msgs.msg import AttitudeTarget
from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header,Float32
import threading
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
        self.att_thread = threading.Timer(5.0,self.send_att)
        self.att_thread.daemon = True
        self.att_thread.start()

    def tearDown(self):
        super(MavrosOffboardAttctlTest, self).tearDown()

    #
    # Helper methods
    #
    def send_att(self):
        rate = rospy.Rate(10)  # Hz
        self.att.body_rate = Vector3()
        self.att.header = Header()
        self.att.header.frame_id = "base_footprint"
        self.att.orientation = Quaternion(*quaternion_from_euler(-0.25, 0.15,
                                                                 0))
        self.att.thrust = 0.7
        self.att.type_mask = 7  # ignore body rate
        counter = 0
        # time.sleep(1)
        while not rospy.is_shutdown():
            
            timeout = 15  # (int) seconds
            loop_freq = 50  # Hz
            rate = rospy.Rate(loop_freq)
            for i in xrange(timeout * loop_freq):
                if(i>30 and i<600):
                    yaw   = 0.3 + 0.1*np.sin(0.01*i)
                    pitch = 0.08*np.cos(0.01*i)
                    roll  = -0.2
                else:
                    roll, pitch, yaw = 0.0, 0.0, 0.3
                    
                self.att.orientation = Quaternion(*quaternion_from_euler(roll, pitch,yaw))
                self.att_setpoint_pub.publish(self.att)
                self.custom_msg_angles = Vector3()
                self.custom_msg_angles.x = np.rad2deg(roll)
                self.custom_msg_angles.y = np.rad2deg(pitch)
                self.custom_msg_angles.z = np.rad2deg(yaw)
                
                self.custom_pub.publish(self.custom_msg_angles)
                
                # time.sleep(1)
                # self.att.header.stamp = rospy.Time.now()
                # self.att_setpoint_pub.publish(self.att)
            rospy.loginfo("counter in send method : {0}".format(counter))
            counter += 1
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    #
    # Test method
    #
    def test_attctl(self):
        """Test offboard attitude control"""
        
        # make sure the simulation is ready to start the mission
        # self.wait_for_topics(60)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 10, -1)

        self.log_topic_vars()
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)

        rospy.loginfo("run mission")
        
        timeout = 15  # (int) seconds0
        loop_freq = 2  # Hz
        rate = rospy.Rate(loop_freq)
        count = timeout * loop_freq
        crossed = False
        for i in xrange(count):
            rospy.loginfo("in Test method, iter: {0}".format(i))
            if (i == count-1):
                rospy.loginfo("reached count in Test method, Timeout {0}".format(timeout))
                crossed = True
                break
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
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   90, 0)
        self.set_arm(False, 5)


if __name__ == '__main__':
    import rostest
    rospy.init_node('test_node', anonymous=True)

    rostest.rosrun(PKG, 'mavros_offboard_attctl_test',
                   MavrosOffboardAttctlTest)
