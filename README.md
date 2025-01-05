# Maze Planning Node

A ROS2 node for maze navigation and planning, implementing SLAM, path planning, and autonomous movement strategies using occupancy grids and the A* algorithm.

## Dependencies
- `rclpy` for ROS2 Node management.
- `numpy` for numerical computations.
- `matplotlib` for plotting and visualization.
- `scipy` for path smoothing (Gaussian filters).
- ROS2 message types: `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `irobot_create_msgs`.