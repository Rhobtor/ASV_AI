import os 
# Set the OPENBLAS_CORETYPE environment variable , TO WORK WITH ARM THAT USE CUDA AND CPU APLICATIONS LIKE ZED
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import pyzed.sl as sl
from time import sleep
import threading
import numpy as np
from camera_interface.msg import  Nodeupdate
import json, traceback
from rclpy.executors import MultiThreadedExecutor

class ZEDNode(Node):
    def __init__(self):
        super().__init__('zed_node')
        self.bridge = CvBridge()
        self.init_zed()

        self.image_publisher = self.create_publisher(Image, 'zed_image', 10)
        self.depth_publisher = self.create_publisher(Image, 'zed_depth', 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'zed_pointcloud', 10)

        self.zed_thread = threading.Thread(target=self.zed_loop)
        self.zed_thread.start()

    def init_zed(self):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.sdk_verbose = True

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {err}")
            rclpy.shutdown()

    def zed_loop(self):
        runtime_params = sl.RuntimeParameters()
        image_msg = Image()
        depth_msg = Image()

        while rclpy.ok():
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.publish_image(image_msg, sl.VIEW.LEFT)
                self.publish_image(depth_msg, sl.VIEW.DEPTH)
                self.Pointcloud_publish()
                self.publish_camera_info()
                sleep(0.01)

    def publish_image(self, msg, view_type):
        image = sl.Mat()
        self.zed.retrieve_image(image, view_type)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'zed_camera'
        msg = self.bridge.cv2_to_imgmsg(image.get_data(), encoding='bgr8')
        msg.header = header
        self.image_publisher.publish(msg)

    def Pointcloud_publish(self):
        point_cloud = sl.Mat()
        self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        point_cloud_data = point_cloud.get_data()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'  # Adjust the frame_id according to your application

        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = point_cloud.get_width() * point_cloud.get_height()
        pc_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12  # 3 floats (x, y, z) * 4 bytes each
        pc_msg.row_step = pc_msg.point_step * pc_msg.width
        pc_msg.is_dense = True

        pc_msg.data = np.array(point_cloud_data).astype(np.float32).tobytes()

        self.pointcloud_publisher.publish(pc_msg)

def main():
    rclpy.init()
    try:
        Zed_Node = ZEDNode()
        rclpy.spin(Zed_Node, executor=MultiThreadedExecutor())
        # After finish close the camera
        Zed_Node.get_logger().info("normal finish")
        Zed_Node.zed.close()
        Zed_Node.destroy_node()
    except:
        #There has been an error with the program, so we will send the error log to the watchdog
        zed=sl.Camera()
        zed.close()
        x = rclpy.create_node('Zed_Node') #we state what node we are
        publisher = x.create_publisher(Nodeupdate, '_internal_error', 10) #we create the publisher
        #we create the message
        msg = Nodeupdate()
        msg.node = "Zed_Node" #our identity
        msg.message = traceback.format_exc() #the error
        x.get_logger().error(f"error:{msg.message}")
        #to be sure the message reaches, we must wait till wathdog is listening (publisher needs time to start up)
        #TODO: Vulnerable si alguien esta haciendo echo del topic, el unico subscriptor debe ser wathdog
        # este topic está oculto en echo al usar _
        while publisher.get_subscription_count() == 0: #while no one is listening
            sleep(0.01) #we wait
        publisher.publish(msg) #we send the message
        x.destroy_node() #we destroy node and finish



if __name__ == '__main__':
    main()