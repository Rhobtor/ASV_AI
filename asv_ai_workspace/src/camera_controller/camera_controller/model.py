import torch
import torch.nn as nn
import torch.optim as optim
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import quaternion
import cv2
import os
import utm
from torch.hub import load
from geometry_msgs.msg import Point
from std_msgs.msg import String

class CameraTransformNode(Node):

    def __init__(self):
        super().__init__('camera_transform_node')

        # Initialize extrinsic parameters with identity matrix and zero translation
        self.rotation_matrix = np.identity(3)
        self.translation_vector = np.zeros(3)
        self.script_folder = os.path.dirname(os.path.abspath(__file__))  # Get the folder containing the script
        
        # Quality of Service Profiles for subscriptions
        depth_qos = rclpy.qos.QoSProfile(depth=10,
                                          reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
                                          durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
                                          history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST)
        pos_qos = rclpy.qos.QoSProfile(depth=10,
                                        reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
                                        durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
                                        history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST)
        info_qos = rclpy.qos.QoSProfile(depth=10,
                                         reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
                                         durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
                                         history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST)
        
        # Create subscriptions to topics
        self.odometry_sub = self.create_subscription(PoseStamped, '/zed/zed_node/pose', self.pose_callback, pos_qos)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, depth_qos)
        self.camera_image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, info_qos)        
        self.camera_info_sub = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, info_qos)
        self.object_distances = self.create_publisher(String, 'object_distances', 10)
        self.object_coordinates = self.create_publisher(String, 'object_coordinates', 10)
        timer_period = 1.0
        self.object_coordinates_timer = self.create_timer(timer_period, self.real_coordinates)
        self.object_distances_timer = self.create_timer(timer_period, self.object_dect)
        
        # Load YOLOv5 model
        self.model = load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.bridge = CvBridge()
        self.distance = 0
        self.u = 0
        self.v = 0
        self.cv_image = None
        self.depth_image = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.RAD2DEG = 57.295779513

    def pose_callback(self, msg):
        # Camera position in map frame
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z

        # Orientation quaternion
        q = quaternion.quaternion(
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        )

        # Convert quaternion to roll, pitch, and yaw
        self.roll, self.pitch, self.yaw = quaternion.as_euler_angles(q)

    def camera_info_callback(self, msg):
        # Store camera information
        self.camera_info = msg

    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.cv_image is None:
            self.get_logger().info("No image available yet")
            return
            
        self.object_dect()  # Perform object detection

    def depth_callback(self, msg):
        # Get depth values
        self.depth_image = memoryview(msg.data).cast('f')
        self.width = msg.width

    def object_dect(self):
        if self.cv_image is None:
            self.get_logger().info("No image available yet")
            return

        # Perform object detection with YOLOv5
        results = self.model(self.cv_image)
        self.draw_boxes(self.cv_image, results.xyxy[0])
        cv2.imshow("Object Detection", self.cv_image)
        cv2.waitKey(1)

    def draw_boxes(self, image, detections):
        for detection in detections:
            xmin, ymin, xmax, ymax, conf, cls_conf = detection  # Adjusted unpacking
            if int(cls_conf) != 0.00:
                continue
            
            # Draw bounding box
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {cls_conf:.2f}', (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate distance from camera (example, replace with actual distance calculation)
            self.distance = self.calculate_distance(xmin, ymin, xmax, ymax)
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # Draw circle at center of detection box
            cv2.circle(image, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # Draw coordinates of detection box center
            cv2.putText(image, f'({center_x}, {center_y})', (center_x, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Publish distance to a topic
            self.get_logger().info(f"Class: {cls_conf:.2f}, Distance: {self.distance} meters")
            distance_msg = String()
            distance_msg.data = f"Class: {cls_conf:.2f}, Distance: {self.distance} meters"

    def calculate_distance(self, xmin, ymin, xmax, ymax):
        if self.depth_image is None:
            return 0

        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)
        depth_image_np = np.array(self.depth_image)
        depth_values = depth_image_np[center_y * self.width + center_x]
        depth_values = depth_values[depth_values != 0]

        if len(depth_values) > 0:
            return np.mean(depth_values)
        else:
            return 0
        
    def real_coordinates(self):
        # Calculate real coordinates from pixel coordinates and camera distance
        Z = self.distance
        X = (self.u - 663.865) * Z / (533.615)
        Y = (self.v - 364.0325) * Z / (533.66)
        point_camera_frame = np.array([X, Y, Z])

        # Apply rotations to camera frame coordinates
        angle_z = np.radians(90)
        angle_y = np.radians(90)
        rotation_matrix_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        rotation_matrix_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        combined_rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_z)
        point_camera_rotated = np.dot(combined_rotation_matrix, point_camera_frame)

        displacement_y = 0.3922
        displacement_z = 0.2401
        point_camera_shifted = point_camera_rotated + np.array([0, displacement_y, displacement_z])

        self.get_logger().info("Point Center Camera: %s " %  str(point_camera_shifted))

def main(args=None):
    rclpy.init(args=args)
    camera_transform_node = CameraTransformNode()
    rclpy.spin(camera_transform_node)
    camera_transform_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
