import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np
import quaternion

class CameraTransformNode(Node):

    def __init__(self):
        super().__init__('camera_transform_node')

        # Initialize extrinsic parameters with identity matrix and zero translation
        self.rotation_matrix = np.identity(3)
        self.translation_vector = np.zeros(3)

        # Create subscriptions to odometry and depth topics
        self.odometry_sub = self.create_subscription(Pose, 'odometry', self.odometry_callback, 10)
        self.depth_sub = self.create_subscription(Image, 'depth', self.depth_callback, 10)

        # Specify the number of cells in the horizontal and vertical directions
        self.num_cells_horizontal = 3
        self.num_cells_vertical = 3

    def odometry_callback(self, msg):
        # Extract position and orientation from odometry message
        odometry_position = [msg.position.x, msg.position.y, msg.position.z]
        odometry_orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        # Update extrinsic parameters based on odometry
        self.rotation_matrix, self.translation_vector = self.update_extrinsic_parameters(odometry_position, odometry_orientation)

    def update_extrinsic_parameters(self, odometry_position, odometry_orientation):
        # Assuming odometry_position is a 3-element array [x, y, z]
        translation_vector = np.array(odometry_position)

        # Assuming odometry_orientation is a 4-element array [x, y, z, w]
        rotation_quaternion = quaternion.as_quat_array(odometry_orientation)
        rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion)

        return rotation_matrix, translation_vector

    def depth_callback(self, msg):
        # Get a pointer to the depth values casting the data pointer to floating point
        depths = list(msg.data)

        # Image size
        width = msg.width
        height = msg.height

        # Calculate cell dimensions
        cell_width = width // self.num_cells_horizontal
        cell_height = height // self.num_cells_vertical

        # Loop through each cell
        for v_cell in range(self.num_cells_vertical):
            for u_cell in range(self.num_cells_horizontal):
                # Calculate the bounding box for the current cell
                u_start = u_cell * cell_width
                u_end = (u_cell + 1) * cell_width
                v_start = v_cell * cell_height
                v_end = (v_cell + 1) * cell_height

                # Calculate the average depth in the current cell
                average_depth = np.mean(depths[u_start:u_end, v_start:v_end])

                # Convert pixel coordinates to 3D point in camera frame
                u_center = (u_start + u_end) // 2
                v_center = (v_start + v_end) // 2
                Z = average_depth
                X = (u_center - width // 2) * Z / width
                Y = (v_center - height // 2) * Z / height
                point_camera_frame = np.array([X, Y, Z])

                # Transform 3D point to robot's base frame
                point_robot_base_frame = np.dot(self.rotation_matrix, point_camera_frame) + self.translation_vector

                # Process or store the real-world coordinates of the cell
                self.process_real_world_cell(u_cell, v_cell, point_robot_base_frame)

    def process_real_world_cell(self, u_cell, v_cell, point_robot_base_frame):
        # Implement your desired processing or store the real-world coordinates of the cell
        self.get_logger().info("Real-world Coordinates of Cell ({}, {}): {}".format(u_cell, v_cell, point_robot_base_frame))

def main(args=None):
    rclpy.init(args=args)

    camera_transform_node = CameraTransformNode()

    rclpy.spin(camera_transform_node)

    camera_transform_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
