###########################################################
import os 
# Set the OPENBLAS_CORETYPE environment variable , TO WORK WITH ARM THAT USE CUDA AND CPU APLICATIONS LIKE ZED
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
##############################################################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
import math
import numpy as np
import sys
import math
import sys
from rclpy.node import Node
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from time import sleep
from camera_interface.msg import Obstacles, Nodeupdate,Camera
from rcl_interfaces.msg import Log
# from submodulos.asv_identity import get_asv_identity
from datetime import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
from threading import Lock, Thread
import traceback
import pyzed.sl as sl
import cv2
import numpy as np
import threading
from math import atan2, degrees
import torch
import torch.backends.cudnn as cudnn
sys.path.insert(0, '/home/xavier/repositorios/yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class Camera_node(Node):

    def parameters(self):
        # self.vehicle_id = get_asv_identity()
        self.declare_parameter('img_size', 640)
        self.img_size = self.get_parameter('img_size').get_parameter_value().integer_value
        self.declare_parameter('enable_obstacle_avoidance', True)
        self.enable_obstacle_avoidance = self.get_parameter('enable_obstacle_avoidance').get_parameter_value().bool_value
        self.declare_parameter('weights_filename', 'best.pt')
        path = "/home/workspace/weight" + self.get_parameter('weights_filename').get_parameter_value().string_value### cambiar 
        self.weights_filepath = os.path.expanduser(path)
        self.declare_parameter('confidence', 0.4)
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
#done
    def declare_services(self):
        self.sendinfo = self.create_service(CommandBool, 'camera_recording', self.camera_recording_callback) #### hacia el barco de okey
        self.reset_home_service = self.create_service(CommandBool, 'enable_obstacle_avoidance', self.obstacle_avoidance_enable)#### para resetear la deteccion
#done
    def declare_topics(self):
        self.obstacles_publisher = self.create_publisher(Obstacles, 'camera_obstacles', 10)
        self.depth_publisher=self.create_publisher(Obstacles, 'camera_obstacles', 10) #### cambiar el msg
        self.Pointcloud_publish = self.create_publisher(PointCloud2, 'spatial_map', 10)
        # timer_period = 1.0  # seconds
        # self.sonar_publisher = self.create_publisher(Sonar, 'sonar', 10)
        # self.sonar_publisher_timer = self.create_timer(timer_period, self.sonar_publish)
#done
    #def declare_actions(self):

    def __init__(self):
        #start the node
        super().__init__('camera_node')

        #call the parameters
        self.parameters()

        #declare variables
        
        self.recording=False

        #call services
        self.declare_services()

        #declare topics
        self.declare_topics()
        
        ##connect to camera

        self.get_logger().info("Initializing camera")

        # Create a Camera object
        self.zed = sl.Camera()
        

        #define objects for threads
        self.lock = threading.Lock()
        self.run_signal = False
        self.stop_camera_detection=False

############################################################################################################################
######### Create a InitParameters object and set configuration parameters ##################################################
        ##General parameters
        #init_params = sl.InitParameters(svo_real_time_mode=True)
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        init_params.coordinate_units = sl.UNIT.METER # Set units in meters
        init_params.camera_fps = 30  # Set fps at 30
        init_params.sdk_verbose = 1 #disable verbose
        
        ##Positional tracking
        #enable positional tracking
        positional_tracking_param = sl.PositionalTrackingParameters() #load default parameters
        positional_tracking_param.set_floor_as_origin = True
        self.zed.enable_positional_tracking(positional_tracking_param)

        # Set initialization parameters
        detection_parameters = sl.ObjectDetectionParameters()
        detection_parameters.enable_tracking = True # Objects will keep the same ID between frames
        detection_parameters.enable_segmentation = True # Outputs 2D masks over detected objects

        # Set runtime parameters
        detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
        detection_parameters_rt.detection_confidence_threshold = 70
        detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX  

        #

        err = self.zed.enable_object_detection(detection_parameters)
        if err != sl.ERROR_CODE.SUCCESS :
            self.get_logger().error("obstacle detenction couldnt be initialized, closing camera")
            self.zed.close()
            return


        # # Enable spatial mapping
        # spatial_mapping_param = sl.SpatialMappingParameters()
        # spatial_mapping_param.resolution_meter = 0.1  # Set the desired spatial mapping resolution
        # spatial_mapping_param.range_meter = 10.0  # Set the desired spatial mapping range

        # err = self.zed.enable_spatial_mapping(spatial_mapping_param)
        # if err != sl.ERROR_CODE.SUCCESS:
        #     self.get_logger().error("Spatial mapping couldn't be initialized, closing camera")
        #     self.zed.close()
        #     return


        #shared variables
        self.objects = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.runtime_params = sl.RuntimeParameters()
        #self.obj_runtime_param.detection_confidence_threshold = 70
        self.image_left_tmp = sl.Mat()
###############################################################################################################################
        ## aqui no, en ASV
        # #declare threads
         
        self.camera_detection_thread = threading.Thread(target=self.camera_perception, args=(self.weights_filepath,self.img_size, self.confidence,)) #weights, img_size, confidence
        self.camera_recording_thread = threading.Thread(target=self.camera_recording)


        # Open the camera
        try:
            err = self.zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                self.get_logger().error(f"camera couldn't be initialized:{err}")
                self.get_logger().fatal("Camera module is dead")
                self.destroy_node()
                return
        except:
            error = traceback.format_exc()
            self.get_logger().error(f"Connection to camera could not be made, unknown error:\n {error}")
            self.get_logger().fatal("Camera module is dead")
            self.destroy_node()



        # #if defined at start, start obstacle avoidance thread   
        # if self.enable_obstacle_avoidance:
        #     self.camera_detection_thread.start()
        # else:
        #     self.get_logger().info("obstacle avoidance not enabled")

        # while rclpy.ok():
        #     if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        #         # -- Get the image
        #         self.lock.acquire()
        #         self.zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
        #         self.image_net = image_left_tmp.get_data()
        #         self.lock.release()
        #         self.run_signal = True
        #         # -- Detection running on the other thread
        #         while self.run_signal:
        #             sleep(0.001)
        #         # Wait for detections
        #         self.lock.acquire()
        #         # -- Ingest detections
        #         self.zed.ingest_custom_box_objects(self.detections)
        #         self.lock.release()
        #         self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
        #         if self.objects.is_new :
        #             obj_array = self.objects.object_list
        #             #self.get_logger().info(f"{len(obj_array)} Object(s) detected")
        #             obstacles=Obstacles()
        #             obstacles.angle_increment=1.5 # we will cover an area of 110º (camera aperture) starting from -54 to 54º with an increment of 1.5 degrees as we can send at most 72 values
        #             obstacles.distance=[2000 for i in range(72)] #2000 or greater is no obstacle
        #             try:
        #                 for objeto in obj_array: #we will store everything of the object
        #                     obj = Camera()
        #                     obj.id = int(objeto.id) 
        #                     label=str(repr(objeto.label))
        #                     sublabel=str(repr(objeto.sublabel))
        #                     #self.get_logger().info(f"label: {label}, sublabel {sublabel}",once=True)
        #                     obj.label = label
        #                     obj.position = [objeto.position[0], objeto.position[1], objeto.position[2]]
        #                     #self.get_logger().info(f"pos: {[objeto.position[0], objeto.position[1], objeto.position[2]]}")
        #                     obj.confidence = objeto.confidence
        #                     #obj.tracking_state = objeto.tracking_state
        #                     #self.get_logger().info(f"track: {objeto.tracking_state}",once=True)
        #                     obj.dimensions = [objeto.dimensions[0], objeto.dimensions[1], objeto.dimensions[2]]
        #                     obj.velocity = [objeto.velocity[0], objeto.velocity[1], objeto.velocity[2]]
        #                     for i in range(4):
        #                         for j in range(2):
        #                             obj.bounding_box_2d.append(objeto.bounding_box_2d[i][j])
        #                     for i in range(8):
        #                         for j in range(3):
        #                             obj.bounding_box.append(objeto.bounding_box[i][j])
        #                     obstacles.objects.append(obj)

        #                     #for obstacle detection we apply pythagoras theorem
        #                     try:
        #                         minangle=degrees(atan2(objeto.bounding_box_2d[0][0],objeto.position[2]))
        #                         maxangle=degrees(atan2(objeto.bounding_box_2d[1][0],objeto.position[2]))
        #                     except:
        #                         continue
        #                     self.get_logger().info(f"{objeto.bounding_box_2d}, {objeto.position} , angles: {[minangle, maxangle]}")

        #                     #self.get_logger().info(f"{label}, {sublabel} , angles: {[minangle, maxangle]}")
        #                     if abs(minangle)>55 or maxangle>55:
        #                         self.get_logger().error("object trepassed camera limits")
        #                     else:
        #                         for i in range(int((53+minangle)/1.5),int((53+maxangle)/1.5)):                        
        #                             if obstacles.distance[i]>int(objeto.position[2]*100):
        #                                 obstacles.distance[i]=int(objeto.position[2]*100)
        #                 self.obstacles_publisher.publish(obstacles)
        #             except:
        #                 pass
        # self.zed.close()

        
    # def obstacle_avoidance_enable(self, request, response):
    #     if request.value:
    #         self.get_logger().info("obstacle avoidance enabled")
    #         self.stop_camera_detection=False
    #         self.camera_detection_thread.start()
    #     else:
    #         self.stop_camera_detection=True
    #         self.get_logger().info("obstacle avoidance disabled")
    #     return response
########## no se puede hacer aqui el evitar obstaculo hacer en el asv y dedicar este para sacar la info de la camara como el wrapper

#######################################################################################################
#### DEPTH DATA########################################################################################
        
    def depth_publisher(self):
        image = sl.Mat()
        depth_map = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
            # A new image and depth is available if grab() returns SUCCESS
            self.zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve left image
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth

        # Convert the depth_map to a numpy array
        depth_data = depth_map.get_data()

        # Create an Image message
        depth_msg = Image()
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.height = depth_map.get_height()
        depth_msg.width = depth_map.get_width()
        depth_msg.encoding = '32FC1'  # 32-bit floating point, single channel
        depth_msg.is_bigendian = False
        depth_msg.step = depth_map.get_width() * 4  # Assuming 32-bit float

        # Flatten the depth_data and set it as the data field of the Image message
        depth_msg.data = np.array(depth_data).astype(np.float32).tobytes()

        # Publish the Image message
        self.depth_publisher.publish(depth_msg)


    def Pointcloud_publish(self):
        ## pointcloud map
        point_cloud = sl.Mat()
        self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Convertir la point_cloud a una matriz numpy
        point_cloud_data = point_cloud.get_data()

        # Crear un mensaje PointCloud2
        pc_msg = PointCloud2()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = 'base_link'  # Ajusta el marco de referencia según tu aplicación
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
        pc_msg.is_dense = True  # Assumiendo que todos los puntos son válidos

        # Flatten la matriz point_cloud_data y establecerla como el campo de datos de PointCloud2
        pc_msg.data = np.array(point_cloud_data).astype(np.float32).tobytes()

        # Publicar el mensaje PointCloud2
        self.pointcloud_publisher.publish(pc_msg)



##################################################################################################
######################## RECORD PART ################################################################### 
        
    def camera_recording_callback(self, request, response):
        if request.value:
            if self.recording:
                self.get_logger().info("camera already recording")
                response.success=False
                return response #we are already recording
            self.recording=True
            self.camera_recording_thread.start()
        else:
            if not self.recording: #we arent recording
                response.success=False
                self.get_logger().info("camera wasnt recording")
                return response
            self.get_logger().info("stop recording")
            self.recording=False
        return response
   
    def camera_recording(self):
        name=datetime.today()
        name=name.strftime('%Y.%m.%d..%H.%M')
        name=str("/home/xavier/zed_datasets/recording"+name+".svo")
        recording_param = sl.RecordingParameters(name, sl.SVO_COMPRESSION_MODE.H264)
        err = self.zed.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().info("camera could not start recording")
            return #error
        self.get_logger().info("started recording")
        runtime = sl.RuntimeParameters()
        while True:
            if self.zed.grab(runtime):
                pass #we will enter this if each new frame
            if self.recording==False:
                self.zed.disable_recording()
                return
############################################################################################


##########################################################################################
############ OBJECT DETECTION ###############################

    # def obstacles_publisher(self):
    #     while rclpy.ok() and not self.destroyed:
    #         if self.run_signal:
    #             lock.acquire()

    #             # Convert detections to Obstacles message
    #             obstacles_msg = Obstacles()
    #             obstacles_msg.angle_increment = 1.5

    #             for objeto in self.detections:
    #                 obj_msg = Camera()  # Replace Camera with the actual name of your message
    #                 obj_msg.id = int(objeto.label)
    #                 obj_msg.label = str(objeto.label)
    #                 obj_msg.position = [objeto.bounding_box_2d[0][0], objeto.bounding_box_2d[0][1], 0.0]
    #                 obj_msg.confidence = objeto.probability
    #                 obj_msg.dimensions = [objeto.bounding_box_2d[1][0] - objeto.bounding_box_2d[0][0],
    #                                       objeto.bounding_box_2d[1][1] - objeto.bounding_box_2d[0][1], 0.0]
    #                 obj_msg.velocity = [0.0, 0.0, 0.0]

    #                 obstacles_msg.objects.append(obj_msg)

    #             # Publish the Obstacles message
    #             self.obstacles_publisher.publish(obstacles_msg)

    #             lock.release()
    #             self.run_signal = False

    #         sleep(0.01)


    def img_preprocess(self, img, device, half, net_size):
        net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
        net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        net_image = np.ascontiguousarray(net_image)

        img = torch.from_numpy(net_image).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, ratio, pad

    def xywh2abcd(self, xywh, im_shape):
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
        x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
        y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
        y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_min
        output[2][1] = y_max

        output[3][0] = x_max
        output[3][1] = y_max
        return output

    def detections_to_custom_box(self, detections, im, im0):
        output = []
        for i, det in enumerate(detections):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    # Creating ingestable objects for the ZED SDK
                    obj = sl.CustomBoxObjectData()
                    obj.bounding_box_2d = self.xywh2abcd(xywh, im0.shape)
                    obj.label = cls
                    obj.probability = conf
                    obj.is_grounded = False
                    output.append(obj)
        return output

    def camera_perception(self, weights, img_size, conf_thres=0.2, iou_thres=0.45):

        self.get_logger().info("Intializing Network...")

        device = select_device()
        half = device.type != 'cpu'  # half precision only supported on CUDA
        imgsz = img_size

        # Load model
        model = attempt_load(weights, device=device)  # load FP32
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16
        cudnn.benchmark = True

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        self.get_logger().info("Network initialized")

        while rclpy.ok() and self.stop_camera_detection==False:
            if self.run_signal:
                self.lock.acquire()
                img, ratio, pad = self.img_preprocess(self.image_net, device, half, imgsz)

                pred = model(img)[0]
                det = non_max_suppression(pred, conf_thres, iou_thres)

                # ZED CustomBox format (with inverse letterboxing tf applied)
                self.detections = self.detections_to_custom_box(det, img, self.image_net)
                self.lock.release()
                self.run_signal = False
            sleep(0.01)
##############################################################################################################################
#############################
    # def run_spatial_mapping(self):
    #     spatial_mapping_params = sl.SpatialMappingParameters()
    #     depth_map = sl.Mat()
    #     while rclpy.ok() and not self.destroyed:
    #         # Grab the left image and depth map
    #         if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
    #             self.zed.retrieve_image(self.image_left_tmp, sl.VIEW.LEFT)
    #             self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    #             # Get the spatial map
    #             err = self.zed.update_spatial_map()
    #             if err == sl.ERROR_CODE.SUCCESS:
    #                 # Retrieve the spatial map
    #                 spatial_map = sl.Mat()
    #                 self.zed.retrieve_spatial_map(spatial_map)

    #                 # Convert spatial map to PointCloud2 ROS message
    #                 pc_msg = PointCloud2()
    #                 pc_msg.header.stamp = self.get_clock().now().to_msg()
    #                 pc_msg.header.frame_id = 'base_link'  # Adjust the frame_id according to your application
    #                 pc_msg.height = 1
    #                 pc_msg.width = spatial_map.get_width() * spatial_map.get_height()
    #                 pc_msg.fields = [
    #                     PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    #                     PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    #                     PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    #                 ]
    #                 pc_msg.is_bigendian = False
    #                 pc_msg.point_step = 12  # 3 floats (x, y, z) * 4 bytes each
    #                 pc_msg.row_step = pc_msg.point_step * pc_msg.width
    #                 pc_msg.is_dense = True

    #                 # Flatten the spatial map data and set it as the data field of the PointCloud2 message
    #                 pc_msg.data = np.array(spatial_map.get_data()).astype(np.float32).tobytes()

    #                 # Publish the PointCloud2 message
    #                 self.pointcloud_publisher.publish(pc_msg)

    #             sleep(0.01)


####################################

def main():
    rclpy.init()
    try:
        camera_node = Camera_node()
        rclpy.spin(camera_node, executor=MultiThreadedExecutor())
        # After finish close the camera
        camera_node.get_logger().info("normal finish")
        camera_node.zed.close()
        camera_node.destroy_node()
    except:
        #There has been an error with the program, so we will send the error log to the watchdog
        zed=sl.Camera()
        zed.close()
        x = rclpy.create_node('camera_node') #we state what node we are
        publisher = x.create_publisher(Nodeupdate, '_internal_error', 10) #we create the publisher
        #we create the message
        msg = Nodeupdate()
        msg.node = "camera_node" #our identity
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