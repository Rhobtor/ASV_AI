import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np
import quaternion
import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener
from geometry_msgs.msg import TransformStamped
import time

class CameraTransformNode(Node):

    def __init__(self):
        super().__init__('camera_transform_node')

        # Initialize extrinsic parameters with identity matrix and zero translation
        self.rotation_matrix = np.identity(3)
        self.translation_vector = np.zeros(3)

        # Initialize tf2_ros TransformBroadcaster and TransformListener
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_listener = TransformListener(self)

        # Robot's global position
        self.lat_robot = 37.7749  # Ejemplo, ajusta según la ubicación real del robot
        self.lon_robot = -122.4194  # Ejemplo, ajusta según la ubicación real del robot

        # Create subscriptions to odometry and depth topics
        self.odometry_sub = self.create_subscription(Pose, 'odometry', self.odometry_callback, 10)
        self.depth_sub = self.create_subscription(Image, 'depth', self.depth_callback, 10)

    def odometry_callback(self, msg):
        # Extract position and orientation from odometry message
        odometry_position = [msg.position.x, msg.position.y, msg.position.z]
        odometry_orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        # Update extrinsic parameters based on odometry
        self.rotation_matrix, self.translation_vector = self.update_extrinsic_parameters(odometry_position, odometry_orientation)

        # Broadcast the transformation from the "camera" frame to the "robot_base" frame
        self.broadcast_transform("camera", "robot_base", self.rotation_matrix, self.translation_vector)

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

        # Image coordinates of the center pixel
        u = msg.width // 2
        v = msg.height // 2

        # Linear index of the center pixel
        center_idx = u + msg.width * v

        # Convert pixel coordinates to 3D point in camera frame
        Z = depths[center_idx]
        X = (u - msg.width // 2) * Z / msg.width
        Y = (v - msg.height // 2) * Z / msg.height
        point_camera_frame = np.array([X, Y, Z])

        # Transform 3D point to robot's base frame
        point_robot_base_frame = np.dot(self.rotation_matrix, point_camera_frame) + self.translation_vector

        # Transform point to global coordinates
        lat_global, lon_global, Z_global = self.local_to_global(point_robot_base_frame[0], point_robot_base_frame[1], point_robot_base_frame[2])

        self.get_logger().info("Transformed Point (Global Coordinates): Lat: %s, Lon: %s, Alt: %s", str(lat_global), str(lon_global), str(Z_global))

    def local_to_global(self, X_local, Y_local, Z_local):
        # Radio medio de la Tierra en kilómetros
        radio_de_la_Tierra = 6371.0

        # Conversión de grados a radianes
        lat_robot_rad = np.radians(self.lat_robot)

        # Cálculo de las coordenadas globales
        lat_global = self.lat_robot + (Y_local / radio_de_la_Tierra) * (180.0 / np.pi)
        lon_global = self.lon_robot + (X_local / (radio_de_la_Tierra * np.cos(lat_robot_rad))) * (180.0 / np.pi)

        return lat_global, lon_global, Z_local

    def broadcast_transform(self, frame_id, child_frame_id, rotation_matrix, translation_vector):
        # Construct TransformStamped message
        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = frame_id
        transform_msg.child_frame_id = child_frame_id
        transform_msg.transform.translation.x = translation_vector[0]
        transform_msg.transform.translation.y = translation_vector[1]
        transform_msg.transform.translation.z = translation_vector[2]

        # Convert rotation matrix to quaternion
        rotation_quaternion = quaternion.from_rotation_matrix(rotation_matrix)
        transform_msg.transform.rotation.x = rotation_quaternion[0]
        transform_msg.transform.rotation.y = rotation_quaternion[1]
        transform_msg.transform.rotation.z = rotation_quaternion[2]
        transform_msg.transform.rotation.w = rotation_quaternion[3]

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform_msg)

def main(args=None):
    rclpy.init(args=args)

    camera_transform_node = CameraTransformNode()

    rclpy.spin(camera_transform_node)

    camera_transform_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
