import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import numpy as np
import quaternion
import cv2
import os
import utm
import torch
import torch.backends.cudnn as cudnn
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch
from torch.hub import load
from geometry_msgs.msg import Point
from std_msgs.msg import String
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R


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
        self.odometry_sub = self.create_subscription(PoseStamped, '/zed2i/zed_node/pose', self.pose_callback, pos_qos)
        self.depth_sub = self.create_subscription(Image, '/zed2i/zed_node/depth/depth_registered', self.depth_callback, depth_qos)
        self.camera_image_sub = self.create_subscription(Image,'/zed2i/zed_node/left/image_rect_color',self.image_callback,info_qos)        
        self.camera_info_sub = self.create_subscription(CameraInfo,'/zed2i/zed_node/left/camera_info',self.camera_info_callback,info_qos)
        self.object_distances = self.create_publisher(String, 'object_distances', 10)
        self.object_coordinates = self.create_publisher(String, 'object_coordinates', 10)
        timer_period=1.0
        self.object_coordinates_timer=self.create_timer(timer_period,self.real_coordinates)
        self.object_distances_timer=self.create_timer(timer_period,self.object_dect)
        self.model = YOLO('./utils/yolov5su.engine', task='detect')
        self.bridge = CvBridge()
        np.bool = np.bool_
        self.model(np.zeros((640,640,3)))
        self.rotMat = r = R.from_rotvec(np.pi/2*np.array([1,0,1]))
        self.distance=None
        self.i = 0  
        self.cv_image=None
        self.depth_image=None
        self.point_cloud_data=None



    def image_callback(self, camera_info_msg, camera_image_msg, depth_msg):

        self.camera_info = camera_info_msg # To retrieve calibration data
        self.cv_image = self.bridge.imgmsg_to_cv2(camera_image_msg, desired_encoding='bgr8') # left camera color image to pass YOLO

        # Get a pointer to the depth values casting the data pointer to floating point
        self.depth_image = memoryview(depth_msg.data).cast('f')
        self.width=depth_msg.width
        
        # Infer with YOLO
        np.bool = np.bool_ # Quick fix for some deprecation problems
        self.get_logger().info(f"------------------------------------------------------------------------------\n")
        results = self.model(self.cv_image)

        detected_classes = [results[0].names[i] for i in results[0].boxes.cls.to('cpu').numpy()]
        bounding_boxes = results[0].boxes.xyxy.to('cpu').numpy()
        self.distance = self.calculate_distance(detected_classes, bounding_boxes)

        #self.draw_boxes(self.cv_image, results.xyxy[0])
        cv2.imshow("Object Detection", results[0].plot(conf=False))
        for cls in self.distance.keys():
            self.get_logger().info(f"Class: {cls}, Distance: {self.distance[cls]} meters")
        cv2.waitKey(1)

        #self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)

    def object_dect(self):
        if self.cv_image is None:
            self.get_logger().info("No image available yet")
            
    
        results = self.model(self.cv_image)
        self.draw_boxes(self.cv_image, results.xyxy[0])
        cv2.imshow("Object Detection", self.cv_image)
        cv2.waitKey(1)

    def draw_boxes(self, image, detections):
        for detection in detections:
            xmin, ymin, xmax, ymax, conf, cls_conf = detection  # Adjusted unpacking
            # Filtrar detecciones para la clase 0.00
            if int(cls_conf) != 0.00:
                continue
            
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {cls_conf:.2f}', (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate distance from camera (example, replace with actual distance calculation)
            self.distance = self.calculate_distance(xmin, ymin, xmax, ymax)
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

                # Dibujar un círculo en el centro del cuadro de detección
            cv2.circle(image, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # Dibujar las coordenadas del centro del cuadro de detección
            cv2.putText(image, f'({center_x}, {center_y})', (center_x, center_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # Publish distance to a topic
            self.get_logger().info(f"Class: {cls_conf:.2f}, Distance: {self.distance} meters")
            distance_msg = String()
            distance_msg.data = f"Class: {cls_conf:.2f}, Distance: {self.distance} meters"
            # self.publisher.publish(distance_msg)

    def calculate_distance(self, detected_classes, bboxes): # returns a dict with the distance of each detection in meters
        if self.depth_image is None:
            return -1
        
        real_distance = {cls_id: np.nan for cls_id in detected_classes}
        for i, cls in enumerate(detected_classes):
            xmin, ymin, xmax, ymax= bboxes[i] 
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            depth_image_np = np.array(self.depth_image)
            depth_value = depth_image_np[int(center_y * self.width + center_x)]

            real_dist, _ = self.real_coordinates(depth_value,center_x,center_y)
            real_distance[cls] = real_dist
        
        return real_distance
        
    def real_coordinates(self, z_distance, u, v):

        # Convert pixel coordinates to 3D point in camera frame (Right handed, z-up, x-forward (ROS))
        Z = z_distance
        X = (u - self.camera_info.k[2]) * Z / (self.camera_info.k[0]) # from https://github.com/stereolabs/zed-ros2-wrapper/blob/master/zed_components/src/zed_camera/src/zed_camera_component.cpp line 3067
        Y = (v - self.camera_info.k[5]) * Z / (self.camera_info.k[4])
        point_camera_frame = np.array([X, Y, Z])

        point_camera_shifted = self.rotMat.apply(point_camera_frame)

        #self.get_logger().info("Point Center Camera: %s " %  str(point_camera_shifted))
        real_distance = np.sqrt(X**2 + Y**2 + Z**2)
        
        return real_distance, point_camera_shifted






def main(args=None):
    rclpy.init(args=args)

    camera_transform_node = CameraTransformNode()

    rclpy.spin(camera_transform_node)

    camera_transform_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()