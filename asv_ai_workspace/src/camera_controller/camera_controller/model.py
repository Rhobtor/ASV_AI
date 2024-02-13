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
import torch
import torch.backends.cudnn as cudnn

class CameraTransformNode(Node):

    def __init__(self):
        super().__init__('camera_transform_node')

        # Initialize extrinsic parameters with identity matrix and zero translation
        self.rotation_matrix = np.identity(3)
        self.translation_vector = np.zeros(3)
        self.script_folder = os.path.dirname(os.path.abspath(__file__))  # Get the folder containing the script
        depth_qos=rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
            )
        pos_qos=rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
            )
        info_qos=rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
            )
        # Create subscriptions to odometry and depth topics
        self.odometry_sub = self.create_subscription(Odometry, '/zed2i/zed_node/odom', self.odometry_callback, pos_qos)
        self.depth_sub = self.create_subscription(Image, '/zed2i/zed_node/depth/depth_registered', self.depth_callback, depth_qos)
        self.camera_image_sub = self.create_subscription(Image,'/zed2i/zed_node/left/image',self.camera_image_callback,info_qos)        
        self.camera_info_sub = self.create_subscription(CameraInfo,'/zed2i/zed_node/left/camera_info',self.camera_info_callback,info_qos)
    def odometry_callback(self, msg):
        # Extract position and orientation from odometry message
        odometry_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        odometry_orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        # self.get_logger().info("Test translation point: %s " %  str(odometry_orientation))
       
        # Update extrinsic parameters based on odometry
        self.rotation_matrix, self.translation_vector = self.update_extrinsic_parameters(odometry_position, odometry_orientation)
        # rotation_matrix_str = "\n".join(" ".join(map(str, row)) for row in self.rotation_matrix)
        # self.get_logger().info("Rotation Matrix:\n%s" % rotation_matrix_str)
    def update_extrinsic_parameters(self, odometry_position, odometry_orientation):
        # Assuming odometry_position is a 3-element array [x, y, z]
        translation_vector = np.array(odometry_position)

        # Assuming odometry_orientation is a 4-element array [x, y, z, w]
        rotation_quaternion = quaternion.as_quat_array(odometry_orientation)
        rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion)

        return rotation_matrix, translation_vector
    def camera_info_callback(self, msg):
        # Almacenar la información de la cámara
        self.camera_info = msg

    def camera_image_callback(self,msg):
        self.images=memoryview(msg.data).cast('f')

    def depth_callback(self, msg):
        # Get a pointer to the depth values casting the data pointer to floating point
        depths = memoryview(msg.data).cast('f')

        # Image coordinates of the center pixel
        u = msg.width // 2
        v = msg.height // 2

        # Linear index of the center pixel
        center_idx = u + msg.width * v

        # Convert pixel coordinates to 3D point in camera frame
        Z = depths[center_idx]
        X = (u - 663.865) * Z / (533.615)
        Y = (v - 364.0325) * Z / (533.66)
        point_camera_frame = np.array([X, Y, Z])

        self.get_logger().info("Point Frame (Base Frame): %s " %  str(point_camera_frame))

        # Definir los ángulos de rotación en radianes
        angle_z = np.radians(90)  # Rotación alrededor del eje z
        angle_y = np.radians(90)  # Rotación alrededor del nuevo eje y

        # Matrices de rotación
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

        # Combinar las matrices de rotación
        combined_rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_z)

        # Aplicar las rotaciones a las coordenadas en el marco de la cámara
        point_camera_rotated = np.dot(combined_rotation_matrix, point_camera_frame)

        # Desplazamiento en el eje y y z
        displacement_y = 0.3922
        displacement_z = 0.2401

        # Aplicar desplazamientos
        point_camera_shifted = point_camera_rotated + np.array([0, displacement_y, displacement_z])

        self.get_logger().info("Point Center Camera: %s " %  str(point_camera_shifted))

        

        # Convertir de latitud y longitud a coordenadas UTM, (sub a gps node ASV)
        latitude = 37.4191666
        longitude = -6.0005347
        utm_coords = utm.from_latlon(latitude, longitude)
        utm_x, utm_y, zone_number, zone_letter = utm_coords

        print(f'UTM Coordinates: ({utm_x}, {utm_y}) in zone {zone_number}{zone_letter}')

        # Convertir de coordenadas UTM a latitud y longitud
        back_to_latlon = utm.to_latlon(point_camera_shifted[0]+utm_x, point_camera_shifted[1]+utm_y, zone_number, zone_letter)
        back_latitude, back_longitude = back_to_latlon

        print(f'Back to Latitud, Longitude: ({back_latitude}, {back_longitude})')

        ##
        
        
       # Capture the image with the point marked
        # self.visualize_point(msg, u, v)

    def visualize_point(self, msg, u, v):
        # Convert ROS Image message to OpenCV format
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # Mark the point with a circle
        img_with_point = img.copy()
        cv2.circle(img_with_point, (u, v), 30, (0, 0, 0), -1)  # Circle in red

         # Save the image in the script's folder
        image_path = os.path.join(self.script_folder, 'marked_image.png')
        cv2.imwrite(image_path, img_with_point)

        self.get_logger().info(f"Image saved at: {image_path}")




    def model (self,msg):
        
        # Model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Images
        imgs = self.images  # batch of images

        # Inference
        results = model(imgs)

        # Results
        results.print()
        results.save()  # or .show()

        results.xyxy[0]  # img1 predictions (tensor)
        results.pandas().xyxy[0]  # img1 predictions (pandas)


def main(args=None):
    rclpy.init(args=args)

    camera_transform_node = CameraTransformNode()

    rclpy.spin(camera_transform_node)

    camera_transform_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
