# Imports

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np

from sensor_msgs.msg import Imu, Joy, Range, LaserScan
#the IR I guess?
from irobot_create_msgs.msg import IrIntensityVector
from geometry_msgs.msg import Twist 
from nav_msgs.msg import Odometry
import random #to get random stuff for the turn

#for the map
import math
from matplotlib import pyplot as plt 

# Manage states
from enum import Enum

# Pathfinding
import heapq

import time

# Path smoothing
from scipy.ndimage import gaussian_filter1d

class State(Enum):
  stop = 0
  scan = 1
  go_to_multiple_goals_stop_go = 2

class TTBController_Lab03(Node):
    def __init__(self):
        super().__init__('lab3')

        # Initialization Check
        self.ODOM_INIT = False
        self.PATH_INIT = False
        self.state = State.scan

        self.state_counter = 0  # Initialize scan counter (# loops through scan)

        self.dt = 0.1 # 10 times per second 
        
        # odom variables
        self.x_positions = []
        self.y_positions = []
        self.angles = []
        self.velocity = []           # Velocity published to Twist

        ## Occupancy Grid Variables
        self.obstacle_pts_x = []
        self.obstacle_pts_y = []

        # Resolution of occupancy grid
        self.rows = 100                    
        self.cols = 100
        self.grid_size = 0.1   
                                                 # Space between grid particles ie. 0.1 m
        self.grid_prob_list = np.zeros((self.rows, self.cols, 1))       # Occupancy Grid

        self.grid_base_pt_x = 0     # initial pt of the grid
        self.grid_base_pt_y = 0

        # Path Planning
        self.global_goal_index = []
        self.global_goal_pos = []
        self.goal_distance = 8          # The distance to the goal rel to robot pose
        self.goal_list_original = []    # To visualize the goal list befor smoothing
        self.goal_list = []             # list of [x,y] points
        self.curr_goal_index = 0

        # Bounds
        self.max_linear_vel = 0.4               # m/s
        self.max_steering_angle = math.pi / 2

        # go_to_multiple_goals variable
        self.Kv = 1.5                # velocity constant worked with 0.5, 2, 6,
        self.Kp = 1                  # steering constant worked with 1, 2, 2

        # Parameters
        self.path_update_dist = 1           # Distance between stop -> scan -> path updates
        self.SCAN_COUNT = 5                 # Number of scans before update path


       
        # setup subscriber to Imu msg from /TTB03/imu with buffer size = 10
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # setup publisher for Twist msg to /TTB03/cmd_vel with buffer size = 10     // Given in slides
        self.publisher_twist = self.create_publisher(
            Twist,
            '/TTB03/cmd_vel',
            10
        )

        # Pose Data
        self.subscriber_odom = self.create_subscription(
            Odometry,
            '/TTB03/odom',
            self.odom_callback,
            qos_profile
        )

        # Lidar Data
        self.subscriber_LaserScan = self.create_subscription(
            LaserScan,
            '/TTB03/scan',
            self.scan_callback,
            qos_profile
        )

        # setup controller to run at 10hz (period=.1s) and call method controller_callback
        timer_period=self.dt
        self.timer = self.create_timer(timer_period, self.controller_callback)

    ## Plot Helper Functions

    # Plot the Map
    def plot_obstacle_pts(self):
        plt.figure()
        # All Obstacle Pts
        plt.scatter(self.obstacle_pts_x, self.obstacle_pts_y, label="Sampled Objects", marker='o', s=3)

        # Current robot position
        plt.scatter(self.x_positions[-1], self.y_positions[-1], label='Robot Position', marker='o', s=3)

        # Current robot space (assuming width = 0.36m)
        r = 0.36 / 2
        theta = np.linspace(0,2*np.pi,100)
        x = r*np.cos(theta) + self.x_positions[-1]
        y = r*np.sin(theta) + self.y_positions[-1]
        plt.plot(x,y,label='Robot Size')

        # Plot original optimal path
        x_path_pts = [pos[0] for pos in self.goal_list_original]
        y_path_pts = [pos[1] for pos in self.goal_list_original]
        plt.scatter(x_path_pts, y_path_pts, color='green', marker='o', s=0.1)

        # Plot smoothed optimal path
        x_path_pts = [pos[0] for pos in self.goal_list]
        y_path_pts = [pos[1] for pos in self.goal_list]
        plt.scatter(x_path_pts, y_path_pts, color='purple', marker='o', s=0.1)

        # Plot Global Goal
        plt.scatter(self.global_goal_pos[0], self.global_goal_pos[1], color='red', marker='o', s=5)

        # Plot Map Region
        x_length = self.grid_size * self.cols
        y_length = self.grid_size * self.rows
        x_map_coords = [self.grid_base_pt_x, self.grid_base_pt_x + x_length, self.grid_base_pt_x + x_length, self.grid_base_pt_x, self.grid_base_pt_x]
        y_map_coords = [self.grid_base_pt_y, self.grid_base_pt_y, self.grid_base_pt_y + y_length, self.grid_base_pt_y + y_length, self.grid_base_pt_y]
        plt.plot(x_map_coords, y_map_coords, linestyle='--', color='yellow')

        # Describe
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(self.grid_base_pt_x - self.grid_size, self.grid_base_pt_x + (self.cols * self.grid_size) + self.grid_size)
        plt.ylim(self.grid_base_pt_y - self.grid_size, self.grid_base_pt_y + (self.rows * self.grid_size) + self.grid_size)
        plt.savefig("pt_cloud.png")
        plt.close()
        
    def plot_occupancy_grid(self):
        plt.imshow(self.grid_prob_list, cmap='viridis', origin='lower')
        plt.colorbar(label='Intensity')
        plt.savefig("occupancy_grid.png")
        plt.close()

    def plot_velocity(self):
        plt.figure()
        # Some reason this is occasionall one off (race condition? I don't see how?)
        t_sample = np.arange(0, self.dt * (len(self.sampled_velocity)), self.dt)    # Create time (x axis) 
        plt.plot(t_sample[:len(self.sampled_velocity)], self.sampled_velocity, color="Blue", label="Odom Sampled")     # to ensure same length
        t_theoretical = np.arange(0, self.dt * (len(self.velocity)), self.dt)    # Create time (x axis) 
        plt.plot(t_theoretical[:len(self.velocity)], self.velocity, color="Red", label="Twist Instruction")     # to ensure same length
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.savefig("velocity_plot.png")

    def plot_position_goals(self):
        plt.figure()
        plt.scatter(self.x_positions, self.y_positions, s=0.5, c="blue")
        plt.scatter(self.goal_list[:,0], self.goal_list[:,1], s=10, c="red", marker="*")
        plt.xlabel('X')
        plt.ylabel('Y') 
        plt.savefig("position_plot.png")

    def plot_position_one_goal(self):
        plt.figure()
        plt.scatter(self.x_positions, self.y_positions, s=0.5, c="blue")
        plt.scatter(self.x_goal, self.y_goal, s=10, c="red", marker="*")
        plt.xlabel('X')
        plt.ylabel('Y') 
        plt.savefig("position_plot.png")

    def controller_callback(self):
        if(self.ODOM_INIT):
            if(not self.PATH_INIT):
                # Get robot start index
                robot_pt_x = self.quantize(self.x_positions[-1])
                robot_pt_y = self.quantize(self.y_positions[-1])
                robot_index_x = self.convert_quantized_pt_or_pts_to_grid_index(robot_pt_x, self.grid_base_pt_x)
                robot_index_y = self.convert_quantized_pt_or_pts_to_grid_index(robot_pt_y, self.grid_base_pt_y)
                
                start_pt = [robot_index_x,robot_index_y]

                # Get Path in Indexes               
                path_indexes = self.a_star(self.grid_prob_list, start_pt, self.global_goal_index)

                # Convert Path indexes to Path Points
                path_x_indexes, path_y_indexes = zip(*path_indexes) 
                path_x_indexes = list(path_x_indexes)
                path_y_indexes = list(path_y_indexes)

                pts_on_path_x = self.convert_index_to_pt_or_pts(path_x_indexes, self.grid_base_pt_x)
                pts_on_path_y = self.convert_index_to_pt_or_pts(path_y_indexes, self.grid_base_pt_y)
                path = list(zip(pts_on_path_x, pts_on_path_y))

                self.goal_list_original = path

                # Smooth Path if Needed - Increase Sigma for smoother path
                sigma = 1
                x_smooth = gaussian_filter1d(pts_on_path_x, sigma=sigma)
                y_smooth = gaussian_filter1d(pts_on_path_y, sigma=sigma)
                path = list(zip(x_smooth, y_smooth))

                self.goal_list = path           # go to multiplie goals

                # Set first local goal
                goal_step_size = 0.2        # Distance to first local goal
                self.curr_goal_index = min(int(goal_step_size // self.grid_size), len(path) - 1)

                # Update States
                self.PATH_INIT = True
                self.plot_obstacle_pts()
                print('Initialized Path')

            elif self.PATH_INIT and self.state == State.go_to_multiple_goals_stop_go:
                # wait for first Odom measurement
                if len(self.x_positions) != 0:
                    if(self.curr_goal_index < len(self.goal_list)):
                        # Current goal and robot position
                        x_curr_goal = self.goal_list[self.curr_goal_index][0]
                        y_curr_goal = self.goal_list[self.curr_goal_index][1]
                        x_curr = self.x_positions[-1]
                        y_curr = self.y_positions[-1]

                        # Get new velocity
                        v_d = self.Kv * math.sqrt((x_curr_goal - x_curr)**2 + (y_curr_goal - y_curr)**2)
                        vel = min(abs(v_d), self.max_linear_vel)

                        # Get Steering
                        steering_angle = self.get_steering_angle(x_curr_goal, y_curr_goal, x_curr, y_curr)

                        # Publish next Twist
                        self.publish_next_twist(vel, steering_angle)

                        # Go to next goal or Stop if approximately at goal (same as go-to-goal)
                        dist_to_goal = np.sqrt((x_curr_goal - x_curr)**2 + (y_curr_goal - y_curr)**2)
                        if(dist_to_goal < 0.1):
                            print(f"Reached Local Goal {self.curr_goal_index + 1}!")
                            if self.curr_goal_index + 1 < len(self.goal_list):
                                self.curr_goal_index += 1
                            else:
                                print('Reached Destination')
                                self.state = State.stop

                        # Scan again after a little bit
                        dist_to_first_local_goal = np.sqrt((x_curr - self.goal_list[0][0])**2 + (y_curr - self.goal_list[0][1])**2)
                        if dist_to_first_local_goal > self.path_update_dist:
                            self.publish_next_twist(float(0), float(0))   # Stop Robot Movement
                            time.sleep(0.1)                # Pause for better calulations
                            self.state = State.scan

                    else:
                        print(f'Reached all goals!')   
                        self.state = State.stop
                
                # self.state = State.scan # For testing
                self.plot_obstacle_pts()
                    
    def odom_callback(self, msg):
        # x and y position relative to battery charging station
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y 

        # Find Yaw
        q0 = msg.pose.pose.orientation.w
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z

        psi = math.atan2(2*(q0*q3+q1*q2),1-2*(q2**2 + q3**2))   # yaw

        # Update Odom
        self.x_positions.append(x)
        self.y_positions.append(y)
        self.angles.append(psi)

        # Initialize
        if(not self.ODOM_INIT):           
            # Initialize Pose
            start_x = self.x_positions[-1]
            start_y = self.y_positions[-1]
            start_theta = self.angles[-1]

            # Initialize global goal
            global_x, global_y, _ = self.homogenous_transform(start_x, start_y, start_theta, self.goal_distance, 0, 0)

            # Initialize Grid Base pt With center of grid between start and global goal
            grid_center_x = (global_x + start_x) / 2
            grid_center_y = (global_y + start_y) / 2
            
            self.grid_base_pt_x = self.quantize(grid_center_x - (self.grid_size * self.cols / 2))
            self.grid_base_pt_y = self.quantize(grid_center_y - (self.grid_size * self.rows / 2))

            # Initialize global goal pos and indices
            global_x = self.quantize(global_x)
            global_y = self.quantize(global_y)
            global_x_index = self.convert_quantized_pt_or_pts_to_grid_index(global_x, self.grid_base_pt_x)
            global_y_index = self.convert_quantized_pt_or_pts_to_grid_index(global_y, self.grid_base_pt_y)

            self.global_goal_index = [global_x_index, global_y_index]
            self.global_goal_pos = [global_x, global_y]

            # Update State
            self.ODOM_INIT = True
    
    def scan_callback(self, msg):
        if(self.ODOM_INIT and self.state == State.scan):
            # print('Scanning')

            # Current Pose
            x_pos = self.x_positions[-1]
            y_pos = self.y_positions[-1]
            angle = self.angles[-1]

            # List of Range Angles
            range_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges)) #+ angle     # Get angles for each range. 
            
            # Clear Obstacles
            # self.obstacle_pts_x = []
            # self.obstacle_pts_y = []

            range_index = 0
            for range in msg.ranges:
                if(range >= msg.range_min and range <= msg.range_max 
                    # Reduce lidar angles processed    
                    # and range_angles[range_index] >= (-135 * np.pi/180) 
                    # and range_angles[range_index] <= (-45 * np.pi/180)
                    ):

                    # Consider the Robot Pose
                    # dx = range*np.cos(range_angles[range_index] + angle)        # I think
                    # dy = range*np.sin(range_angles[range_index] + angle)
                    # x_pt = x_pos + dx
                    # y_pt = y_pos + dy

                    # Ad hoc Solution: Why did the + 90 degrees work?
                    dx = range*np.cos(range_angles[range_index])
                    dy = range*np.sin(range_angles[range_index])
                    dtheta = 0                                          # obstacles don't need orientation.
                    x_pt, y_pt, _ = self.homogenous_transform(x_pos, 
                                                              y_pos,
                                                              angle + (90 * np.pi/180),
                                                              dx, dy, dtheta)

                    # Might not be best way to update array
                    self.obstacle_pts_x = np.append(self.obstacle_pts_x, x_pt)
                    self.obstacle_pts_y = np.append(self.obstacle_pts_y, y_pt)

                range_index += 1
            
            ## Update occupancy map
            self.obstacle_pts_x = np.array(self.obstacle_pts_x)
            self.obstacle_pts_y = np.array(self.obstacle_pts_y)

            # Quantize pts
            quantized_obstacle_pts_x = self.quantize(self.obstacle_pts_x)
            quantized_obstacle_pts_y = self.quantize(self.obstacle_pts_y)
            robot_x_quantized = self.quantize(x_pos)
            robot_y_quantized = self.quantize(y_pos)
            
            # Convert to grid indices
            x_obstacle_indexes = self.convert_quantized_pt_or_pts_to_grid_index(quantized_obstacle_pts_x, self.grid_base_pt_x)
            y_obstacle_indexes = self.convert_quantized_pt_or_pts_to_grid_index(quantized_obstacle_pts_y, self.grid_base_pt_y)
            x_robot_index = self.convert_quantized_pt_or_pts_to_grid_index(robot_x_quantized, self.grid_base_pt_x)
            y_robot_index = self.convert_quantized_pt_or_pts_to_grid_index(robot_y_quantized, self.grid_base_pt_y)

            # Update Occupancy Grid
            for range_index_np in np.ndindex(x_obstacle_indexes.shape):
                range_index = range_index_np[0]

                self.update_grid_bresenham_algo([x_robot_index,y_robot_index],[x_obstacle_indexes[range_index],y_obstacle_indexes[range_index]])
            
            # Complete Scanning after n loops
            if self.state_counter >= self.SCAN_COUNT:
                self.PATH_INIT = False
                self.state = State.go_to_multiple_goals_stop_go
                self.state_counter = 0

            self.state_counter +=1

            self.plot_obstacle_pts()
            self.plot_occupancy_grid()

    # Get the index or a pt or set or pts
    def convert_quantized_pt_or_pts_to_grid_index(self, pt_or_pts, base_pos):
        rel_pt = pt_or_pts - base_pos
        pt_or_pts_index = np.floor(rel_pt / self.grid_size).astype(int)
        return pt_or_pts_index
    
    # Quantize a pt or set of pts
    def quantize(self, pt_or_pts):
        return np.round(pt_or_pts / self.grid_size) * self.grid_size
    
    # Convert index(es) to pt(s)
    def convert_index_to_pt_or_pts(self, index_or_indexes, base_pos):
        index_or_indexes = np.array(index_or_indexes)
        return (index_or_indexes * self.grid_size) + base_pos

    
    # https://zingl.github.io/bresenham.html
    # Given start (x,y) and end (x,y)
    def update_grid_bresenham_algo(self, start_pt,end_pt):
        # Handle out of bounds lasers (prob do somthing else later)
        if(end_pt[0] < 0 or end_pt[0] >= self.cols or 
           end_pt[1] < 0 or end_pt[1] >= self.rows):
            return

        # For some reason switching these helped
        y0,x0 = start_pt
        y1,x1 = end_pt
        
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy  # error value e_xy

        path = []
        while True:  # loop
            path.append([x0,y0])
            if x0 == x1 and y0 == y1:
                ## Update the prob of the occupancy grid
                # self.grid_prob_list[x0,y0] = 10

                unit_update_size = 4 // 2            # Update a 3x3 grid
                start_row, end_row = max(0, x0 - unit_update_size), min(self.rows, x0 + unit_update_size + 1)
                start_col, end_col = max(0, y0 - unit_update_size), min(self.cols, y0 + unit_update_size + 1)

                for i in range(start_row, end_row):
                    for j in range(start_col, end_col):
                        if(i == x0 and j == y0):
                            self.grid_prob_list[i][j] = 100000
                        else:    
                            self.grid_prob_list[i][j] = min(self.grid_prob_list[i][j] + 2,100000)
                break
            else:
                # Update scanned regions that are not an obstacle
                # // This leads to non-optimal pathing when used sometimes

                # if(self.grid_prob_list[x0,y0] == -1):
                #     self.grid_prob_list[x0,y0] = -2
                # else:
                # self.grid_prob_list[x0,y0] = -1
                pass

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx  # Move in the x direction
            if e2 <= dx:
                err += dx
                y0 += sy  # Move in the y direction


        
    # Get a translated pose with respect to an inital pose 
    def homogenous_transform(self, x, y, theta, x_change, y_change, theta_change):
        initial_pose = np.array([[np.cos(theta), -np.sin(theta), x],
                                 [np.sin(theta), np.cos(theta), y],
                                 [0, 0, 1]])
        rot_matrix = np.array([[np.cos(theta_change), -np.sin(theta_change), x_change],
                                 [np.sin(theta_change), np.cos(theta_change), y_change],
                                 [0, 0, 1]])
        
        final_pose = initial_pose @ rot_matrix      # matrix multipliation
        x_g = final_pose[0,2]
        y_g = final_pose[1,2]
        theta_g = theta + theta_change

        return x_g, y_g, theta_g

   
    ## Helper Functions
    def get_steering_angle(self, x_goal, y_goal, x_curr, y_curr):
        # Update steering
        theta_d = math.atan2((y_goal - y_curr), (x_goal - x_curr))
        gamma = theta_d - self.angles[-1]
        gamma = np.mod(gamma + np.pi, 2*np.pi) - np.pi 
        gamma = self.Kp * gamma   
        steering_angle = min(abs(gamma), self.max_steering_angle) * np.sign(gamma)
        return steering_angle
    
    def publish_next_twist(self, vel, steering_angle):
        # Publish next Twist
        self.velocity.append(vel)
        move = Twist()
        move.linear.x = vel * math.cos(steering_angle)
        move.angular.z = steering_angle
        self.publisher_twist.publish(move)

    def compute_u_for_PID(self):
        # Compute u(t)
        error = self.velocity_ref_point - self.velocity[-1]
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        u_t = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error

        return u_t
    
    # A* Algorithm to return a path
    class Node:
        def __init__(self, position, g=0, h=0):
            self.position = position
            self.g = g                      # Cost from start to this node
            self.h = h                      # Heuristic cost from this node to end
            self.f = g + h                  # Total cost
            self.parent = None

        # Less than comparison
        def __lt__(self, other):
            return self.f < other.f

    def heuristic(self, a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, grid, start_pt, end_pt):
        open_list = []
        closed_list = set()

        start_node = self.Node(tuple(start_pt), g=0, h=self.heuristic(start_pt, end_pt))
        end_node = self.Node(tuple(end_pt))

        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)
            closed_list.add(tuple(current_node.position))

            # Check end reached
            if current_node.position == end_node.position:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]                           # Return reversed path

            x, y = current_node.position

            # Explore neighbors
            neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),                # Adjacent
                         (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]        # Diagonal
            
            for nx, ny in neighbors:
                if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):          # Check grid boundaries
                    if grid[ny][nx] == np.inf or (nx, ny) in closed_list:   # Skip obstacles and closed nodes
                        continue

                    # Calculate g, h, and f values
                    g = current_node.g + grid[ny][nx]
                    h = self.heuristic((nx, ny), end_pt)
                    neighbor_node = self.Node((nx, ny), g=g, h=h)
                    neighbor_node.parent = current_node

                    # Check path is better or not in open list
                    if (nx, ny) not in closed_list and all(neighbor_node.f < node.f for node in open_list if node.position == (nx, ny)):
                        heapq.heappush(open_list, neighbor_node)

        return None  # No path found


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    ttb_controller = TTBController_Lab03()

    # Execute work on the node
    rclpy.spin(ttb_controller)

    # Clean up
    ttb_controller.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# Strategy
'''
Current Strategy In progress:
- Occupancy grid to map area
- A* to find best path (not really working since returns windy path based on measurements)
- follow path algo of sorts. 


- Current problem: My occupany map strategy drifts, even though I believe I am implementing stuff correctly
- I just need to use a better occupancy map because I'm not considering all the world frames etc...
- Soln. Implment gmapping 



'''