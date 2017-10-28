#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight, TrafficLightArray
from std_msgs.msg import Int32
import math
import tf
from copy import deepcopy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stop line location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
PUBLISH_RATE = 20 # Publish rate(Hz)
ACCEL = 1.0 # 
STOP_DIST = 2.0 #


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.current_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        self.traffic_light_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        self.current_velocity_sub = rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add state variables
        self.base_waypoints = [] # List of waypoints, received from /base_waypoints
        self.base_wp_orig_v = [] # Starting velocity of waypoints
        self.next_waypoint = None # Next waypoint index in car direction
        self.current_pose = None # Car position
        self.current_velocity = None # Car velocity
        self.red_light_index = None; # The waypoint index of next red light
        self.msg_seq = 0 # Sequence number of /final_waypoints message
        self.max_velocity = None # The maximum velocity 
        self.slow_dist = None # The distance between red light and current car to start slowing down
        self.loop()

    def loop(self):
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            is_initialized = self.base_waypoints and self.current_pose
            print("Initialized")
            if is_initialized:
                cur_idx = self.closest_waypoint()
                self.next_waypoint = self.get_next_waypoint()
                self.publish()
            rate.sleep()


    def publish(self):
        final_waypoints_msg = Lane()
        final_waypoints_msg.header.stamp = rospy.Time.now()
        final_waypoints_msg.header.seq = self.msg_seq
        final_waypoints_msg.header.frame_id = '/world'
        self.msg_seq += 1
        self.set_final_waypoints()
        self.set_final_velocities()
        final_waypoints_msg.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(final_waypoints_msg)


    def set_final_waypoints(self):
        #we just copy the waypoints in base waypoints which are in front of the current position
        self.final_waypoints = deepcopy(self.base_waypoints[self.next_waypoint : self.next_waypoint + LOOKAHEAD_WPS])
        #if self.next_waypoint + LOOKAHEAD_WPS > len(self.base_waypoints), we just add from index 0
        diff = LOOKAHEAD_WPS - len(self.final_waypoints)
        if diff > 0:
            self.final_waypoints = self.final_waypoints + deepcopy(self.base_waypoints[0:diff])

    def set_final_velocities(self):
        if self.red_light_index == None:
            #if no red light ahead, set all the final waypoints' speed at max speed
            speed = self.current_velocity
            target_speed = self.max_velocity
            for wp in self.final_waypoints:
                if speed > target_speed:
                    speed = max(target_speed, speed - ACCEL)
                else:
                    speed = min(target_speed, speed + ACCEL)
                self.set_waypoint_velocity(wp, speed)
        else:
            dist = self.distance(self.base_waypoints, self.next_waypoint, self.red_light_index)
            rospy.logwarn("Next wp: %s, Next TL wp: %s, distance: %s",self.next_waypoint_index, self.light_wp, dist)
            if dist <= self.slow_dist:
                speed = self.current_velocity
                if dist > STOP_DIST:
                    decel = speed * speed / (2 * (dist - STOP_DIST))
                    rospy.logwarn("Decelerating to stop light with : %f",decel)
                else:
                    speed = 0
                    decel = 0
                    rospy.logwarn("Stopped")
                for wp in final_waypoints:
                    speed = max(0, speed - decel)
                    self.set_waypoint_velocity(wp, speed)
            else:
                speed = self.current_velocity
                if speed < self.max_speed:
                    rospy.logwarn("Accelerating to top speed again...")
                for wp in self.final_waypoints:
                    if speed > self.max_speed:              
                        speed = max(self.max_speed, speed-ACCEL)
                    else:
                        speed = min(self.max_speed, speed+ACCEL)
                    self.set_waypoint_velocity(wp, speed)


    def car_yaw(self):
        #return the current car yaw
        quaternion = (
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def closest_waypoint(self):
        #Given the current position, we want to find the closest waypoint index in base_waypoints
        min_dist = 1000000
        ans = -1
        cur_x = self.current_pose.pose.position.x
        cur_y = self.current_pose.pose.position.y
        start = 0
        end = len(self.base_waypoints)
        for idx in range(start, end):
            map_x = self.base_waypoints[idx].pose.pose.position.x
            map_y = self.base_waypoints[idx].pose.pose.position.y
            dist = self.distance_between_any(cur_x, cur_y, map_x, map_y)
            if dist < min_dist:
                min_dist = dist
                ans = idx
        return ans

        

    def distance_between_any(self, x1, y1, x2, y2):
        #return the euclidean distance between(x1, y1) and (x2, y2)
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    def get_next_waypoint(self):
        #Here we make use of the closest waypoint index and the difference between the current car yaw and the
        #heading angle between (cur_x, cur_y) and (map_x, map_y) to determine the waypoint index of the next waypoint in base_waypoints
        ind = self.closest_waypoint()

        map_x = self.base_waypoints[ind].pose.pose.position.x
        map_y = self.base_waypoints[ind].pose.pose.position.y

        cur_x = self.current_pose.pose.position.x
        cur_y = self.current_pose.pose.position.y

        yaw = self.car_yaw() #the current yaw rate of the car
        heading = math.atan2((map_y - cur_y), (map_x - cur_x))
        diff_angle = abs(yaw - heading)
        if diff_angle > math.pi/4:
            ind = (ind + 1) % len(self.base_waypoints)
        return ind


    def get_waypoint_velocity(self, waypoints, waypoint):
        return waypoints[waypoint].twist.twist.linear.x


    def pose_cb(self, msg):
        self.current_pose = msg

    def waypoints_cb(self, msg):
        """
        Receive and store the whole list of waypoints.
        """
        waypoints = msg.waypoints
        self.base_waypoints = waypoints
        if self.base_waypoints:
            #we only need the base waypoints information once, so after receiving it, we unregister the subscriber
            self.base_waypoints_sub.unregister()
            rospy.loginfo("Unregistered from /base_waypoints topic")
            self.max_velocity = self.get_waypoint_velocity(self.base_waypoints[0])
            self.slow_dist = self.max_velocity * 4
        

    def traffic_cb(self, msg):
        """
        Receive and store the waypoint index for the next red traffic light.
        If the index is <0, then there is no red traffic light ahead
        """
        self.red_light_index = msg.data if msg.data >= 0 else None

    def current_velocity_cb(self, msg):
        #Get the current velocity
        self.current_velocity = msg.twist.linear.x


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        if wp2 >= wp1:
            for i in range(wp1, wp2+1):
                dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
                wp1 = i
            return dist
        else:
            for i in range(wp1, len(waypoints)):
                dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
                wp1 = i
            for i in range(0, wp2-1):
                dist += dl(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
            return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
