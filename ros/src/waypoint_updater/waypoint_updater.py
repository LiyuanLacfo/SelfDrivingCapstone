#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight, TrafficLightArray
from std_msgs.msg import Int32
import math
import tf

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
POSITION_SEARCH_RANGE = 10 # Number of waypoints to search for the next waypoint based on the current position
PUBLISH_RATE = 20 # Publish rate(Hz)
debugging = True


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add state variables
        self.base_waypoints = [] # List of waypoints, received from /base_waypoints
        self.base_wp_orig_v = [] # Starting velocity of waypoints
        self.next_waypoint = None # Next waypoint index in car direction
        self.current_pose = None; # Car position
        self.red_light_waypoint = None; # The waypoint index of next red light
        self.msg_seq = 0 # Sequence number of /final_waypoints message

        # Parameters
        self.accel = rospy.get_param('~target_brake_accel', -1.0)
        self.stop_distance = rospy.get_param('~stop_distance', 5.0) # Distance (m) where car will stop before red light

        try:
            # To make brake feasible
            self.accel = max(rospy.get_param('/dbw_node/decel_limit')/2, self.accel)
        except KeyError:
            pass

        rospy.loginfo("Acceleration is {}".format(self.accel))
        rospy.loginfo("Stop distance is {}".format(self.stop_distance))

        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            self.update_and_publish()
            rate.sleep()

    def update_and_publish(self):
        """
        (1)update next waypoint based on the current car position and base waypoints
        (2)generate a list of next LOOKAHEAD_WPS waypoints
        (3)update velocity for them
        (4)publish to /final_waypoints
        """
        self.get_next_waypoint()
        num_base = len(self.base_waypoints)
        last_base_wp = num_base-1
        waypoint_idx = [idx % num_base for idx in range(self.next_waypoint,self.next_waypoint+LOOKAHEAD_WPS)]
        final_waypoints = [self.base_waypoints[wp] for wp in waypoint_idx]

        #if there is a red light ahead, update the velocity
        # Start from original velocities
        self.restore_velocities(waypoint_idx)
        try:
            red_idx = waypoint_idx.index(self.red_light_waypoint)
            self.decelerate(final_waypoints, red_idx, self.stop_distance)
        except ValueError:
            # No red light available: self.red_light_waypoint is None or not in final_waypoints
            red_idx = None
        if debugging:
            v = self.get_waypoint_velocity(final_waypoints, 0)
            rospy.loginfo("Target velocity: %.1f, RL:%s wps ahead", v, str(red_idx))

        # If we are close to the end of the circuit, make sure that we stop there
        if self.base_wp_orig_v[-1] < 1e-5:
            try:
                last_wp_idx = waypoint_idx.index(last_base_wp)
                self.decelerate(final_waypoints, last_wp_idx, 0)
            except ValueError:
                # Last waypoint is not one of the next LOOKAHEAD_WPS
                pass

        # Publish waypoints to "/final_waypoints"
        self.publish_msg(final_waypoints)

    def publish_msg(self, final_waypoints):
        waypoint_msg = Lane()
        waypoint_msg.header.seq = self.msg_seq
        waypoint_msg.header.stamp = rospy.Time.now()
        waypoint_msg.header.frame_id = '/world'
        waypoint_msg.waypoints = final_waypoints
        self.final_waypoints_pub.publish(waypoint_msg)
        self.msg_seq += 1


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
        min_dist = 1000000
        idx = -1

        start = 0
        end = len(self.base_waypoints)
        if(self.next_waypoint):
            start = max (self.next_waypoint - POSITION_SEARCH_RANGE, 0)
            end = min (self.next_waypoint + POSITION_SEARCH_RANGE, end)
        position1 = self.current_pose.pose.position
        for i in range(start, end):
            position2 = self.base_waypoints[i].pose.pose.position
            dist = self.distance_between_two_position(position1, position2)
            if dist < min_dist:
                min_dist = dist
                idx = i
        return idx

    def distance_between_two_position(self, position1, position2):
        a = position1
        b = position2
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


    def get_next_waypoint(self):
        ind = self.closest_waypoint()

        wp_x = self.waypoints[ind].pose.pose.position.x
        wp_y = self.waypoints[ind].pose.pose.position.y

        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y

        yaw = self.car_yaw()

        diff_angle_cos = ((wp_x-x) * math.cos(yaw) + (wp_y-y) * math.sin(yaw))
        if ( diff_angle_cos < 0.0 ):
            ind += 1
        self.next_waypoint = ind

    def restore_velocities(self, indexes):
        """
        Restore original velocities of points
        """
        for idx in indexes:
            self.set_waypoint_velocity(self.base_waypoints, idx, self.base_wp_orig_v[idx])

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


    def decelerate(self, waypoints, stop_index, stop_distance):
        """
        reduce the velocity of  a list of wayponts so that they stop on stop_index
        """
        if stop_index <= 0:
            return
        dist = self.distance(waypoints, 0, stop_index)
        step = dist / stop_index
        #  Everything beyond stop_index will have velocity = 0
        #  Before that, constant (de)cceleration is applied until reaching
        #    previous waypoint velocity.
        #  We assume constant distance between consecutive waypoints for simplicity
        v = 0.0
        d = 0.0
        for idx in reversed(range(len(waypoints))):
            if idx < stop_index:
                d += step
                if d > self.stop_distance:
                    v = math.sqrt(2*abs(self.accel)*(d-stop_distance))
            if v < self.get_waypoint_velocity(waypoints, idx):
                self.set_waypoint_velocity(waypoints, idx, v)

    def get_waypoint_velocity(self, waypoints, waypoint):
        return waypoints[waypoint].twist.twist.linear.x




    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def waypoints_cb(self, waypoints):
        """
        Receive and store the whole list of waypoints.
        """
        t = time.time()
        waypoints = msg.waypoints
        num_wp = len(waypoints)

        if self.base_waypoints != waypoints:
            # Normally we assume that waypoint list doesn't change (or, at least, not
            # in the position where the car is located). If that happens, just handle it.
            self.base_wp_orig_v = [self.get_waypoint_velocity(waypoints, idx) for idx in range(num_wp)]
            self.base_waypoints = waypoints


    def traffic_cb(self, msg):
        """
        Receive and store the waypoint index for the next red traffic light.
        If the index is <0, then there is no red traffic light ahead
        """
        prev_red_light_waypoint = self.red_light_waypoint
        self.red_light_waypoint = msg.data if msg.data >= 0 else None
        if prev_red_light_waypoint != self.red_light_waypoint:
            if debugging:
                rospy.loginfo("TrafficLight changed: %s", str(self.red_light_waypoint))
            self.update_and_publish() # Refresh if next traffic light has changed

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
