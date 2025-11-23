import rclpy
from rclpy.node import Node
import cv2
import time
import os
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
from .aruco_lib import Aruco
from .usb_camera_reconnect import USBCameraReconnect

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # 1. 声明核心参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('aruco_dict_type', 'DICT_7X7_1000'),
                ('marker_length', 0.15),
                ('image_topic', '/usb_cam/image_raw'),
                ('debug_save_path', './aruco_debug/'),
                ('frame_buffer_size', 5),
                ('detection_cache_size', 10)
            ]
        )
        
        # 2. 读取参数
        self.params = {
            'dict_type': self.get_parameter('aruco_dict_type').value,
            'marker_len': self.get_parameter('marker_length').value,
            'img_topic': self.get_parameter('image_topic').value,
            'debug_path': self.get_parameter('debug_save_path').value,
            'buffer_size': self.get_parameter('frame_buffer_size').value,
            'cache_size': self.get_parameter('detection_cache_size').value
        }
        
        # 3. 初始化组件
        self.bridge = CvBridge()
        self._init_directories()
        self.aruco = Aruco(aruco_type=self.params['dict_type'], if_draw=False)
        self._set_aruco_params()
        self.camera_reconnect = USBCameraReconnect(
            node=self,
            camera_device="/dev/video10",
            usb_hub_pci="0000:00:14.0",
            sudo_password="qing"
        )
        
        # 4. 初始化变量
        self.frame_count = 0
        self.continuous_detect = {}
        self.min_continuous = 1
        self.last_log_time = self.get_clock().now()
        
        # 帧缓存和识别结果缓存
        self.frame_buffer = []
        self.detection_cache = []
        
        # 5. 初始化ROS通信
        self._init_ros_communication()
        
        self.get_logger().info("✅ Aruco检测节点启动(精简版，无预处理)")
        self.get_logger().info(f"帧缓存大小: {self.params['buffer_size']}, 识别缓存大小: {self.params['cache_size']}")

    # ------------------------------
    # 初始化辅助函数
    # ------------------------------
    def _init_directories(self):
        """创建图像保存目录"""
        self.dirs = {
            'raw': os.path.join(self.params['debug_path'], 'raw/'),
            'detected': os.path.join(self.params['debug_path'], 'detected/')
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def _init_ros_communication(self):
        """初始化订阅者和发布者"""
        self.image_sub = self.create_subscription(
            Image, self.params['img_topic'], self.image_callback, 10
        )
        self.marker_pub = self.create_publisher(MarkerArray, '/aruco_markers', 10)
        self.detected_img_pub = self.create_publisher(Image, '/aruco/detected_img', 10)

    def _set_aruco_params(self):
        """配置Aruco检测器参数"""
        try:
            params = self.aruco.detector.getDetectorParameters()
            params.minMarkerPerimeterRate = 0.003
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            params.adaptiveThreshConstant = 7
            self.get_logger().info("✅ Aruco参数配置完成")
        except Exception as e:
            self.get_logger().warn(f"⚠️ 参数配置失败: {str(e)}")

    # ------------------------------
    # 核心回调函数
    # ------------------------------
    def image_callback(self, msg):
        """图像消息回调处理（删除所有预处理步骤）"""
        self.frame_count += 1
        current_time = time.time()
        timestamp = self._get_timestamp(current_time)

        # 1. 图像转换
        try:
            # 读取原始图像并加入缓存
            bgr_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._update_frame_buffer(bgr_img, timestamp)
            self._save_image(bgr_img, self.dirs['raw'], timestamp, 'raw')
            
            # 直接转灰度
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            
        except Exception as e:
            self.get_logger().error(f"❌ 图像处理失败: {str(e)}")
            return

        # 2. Aruco检测
        detected_img = self.aruco.detect_image(gray_img, self.params['dict_type'], if_draw=True)
        if len(detected_img.shape) == 2:
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_GRAY2BGR)
        results = self.aruco.update(gray_img)

        # 3. 处理检测结果并暂存
        self._update_detection_cache(results, timestamp)
        self._handle_detection_results(results, detected_img, timestamp, msg.header)

        # 4. 可视化
        cv2.imshow("Aruco Detection", detected_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    # ------------------------------
    # 暂存相关方法
    # ------------------------------
    def _update_frame_buffer(self, frame, timestamp):
        """更新帧缓存，保持最新的N帧"""
        self.frame_buffer.append({
            'timestamp': timestamp,
            'frame': frame.copy()
        })
        if len(self.frame_buffer) > self.params['buffer_size']:
            self.frame_buffer.pop(0)

    def _update_detection_cache(self, results, timestamp):
        """更新识别结果缓存"""
        if results:
            self.detection_cache.append({
                'timestamp': timestamp,
                'frame_count': self.frame_count,
                'results': results.copy()
            })
            if len(self.detection_cache) > self.params['cache_size']:
                self.detection_cache.pop(0)

    # ------------------------------
    # 辅助处理函数
    # ------------------------------
    def _get_timestamp(self, current_time):
        """生成毫秒级时间戳字符串"""
        return time.strftime("%Y%m%d_%H%M%S_", time.localtime(current_time)) + \
               f"{int(current_time % 1 * 1000):03d}"

    def _save_image(self, img, dir_path, timestamp, prefix):
        """保存图像到指定目录"""
        cv2.imwrite(f"{dir_path}{timestamp}_{prefix}.jpg", img)

    def _handle_detection_results(self, results, detected_img, timestamp, header):
        """处理检测结果并发布消息"""
        # 保存检测结果图像
        if results:
            self._save_image(detected_img, self.dirs['detected'], timestamp, 'detected')
            self.get_logger().info(f"帧{self.frame_count}: 识别到{len(results)}个码")
        
        # 发布Marker
        self._publish_markers(results, header)
        
        # 发布图像
        self._publish_images(detected_img=detected_img, header=header)

    def _publish_markers(self, results, header):
        """发布Marker消息（保留不变）"""
        if not results:
            return
            
        marker_array = MarkerArray()
        for result in results:
            marker_id = result['id']
            self.continuous_detect[marker_id] = self.continuous_detect.get(marker_id, 0) + 1
            if self.continuous_detect[marker_id] < self.min_continuous:
                continue
                
            marker = Marker()
            marker.header = header
            marker.header.frame_id = "camera"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = self.params['marker_len']
            marker.scale.z = 0.01
            marker.color.g = 1.0
            marker.color.a = 0.5
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)

    def _publish_images(self, detected_img, header):
        """发布图像"""
        try:
            # 发布检测结果图像
            detected_msg = self.bridge.cv2_to_imgmsg(detected_img, 'bgr8')
            detected_msg.header = header
            self.detected_img_pub.publish(detected_msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ 图像发布失败: {str(e)}")

# ------------------------------
# 主函数
# ------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 节点停止")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == "__main__":
    main()
