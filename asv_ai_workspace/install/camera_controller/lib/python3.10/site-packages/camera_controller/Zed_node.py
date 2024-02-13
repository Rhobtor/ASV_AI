import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ZedVideoTutorial(Node):
    def __init__(self):
        super().__init__('zed_video_tutorial')
        
        info_qos=rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
            )
        
        # Create right image subscriber
        self.right_sub = self.create_subscription(
            Image, '/zed2i/zed_node/right/image_rect_color', self.image_right_rectified_callback, info_qos)
        
        # Create left image subscriber
        self.left_sub = self.create_subscription(
            Image, '/zed2i/zed_node/left/image_rect_color', self.image_left_rectified_callback, info_qos)
        
    def image_right_rectified_callback(self, msg):
        self.get_logger().info(
            f"Right Rectified image received from ZED\tSize: {msg.width}x{msg.height} - Ts: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} sec")
        
    def image_left_rectified_callback(self, msg):
        self.get_logger().info(
            f"Left Rectified image received from ZED\tSize: {msg.width}x{msg.height} - Ts: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} sec")

def main(args=None):
    rclpy.init(args=args)
    zed_video_tutorial = ZedVideoTutorial()
    rclpy.spin(zed_video_tutorial)
    zed_video_tutorial.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
