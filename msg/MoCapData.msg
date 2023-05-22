# Message for SET_ATTITUDE_TARGET
#
# Some complex system requires all feautures that mavlink
# message provide. See issue #402, #418.

std_msgs/Header header
geometry_msgs/Vector3 orientation
float32[9] Rot
