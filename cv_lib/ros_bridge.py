import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_lib.zmq_bridge import CvBridge
import numpy as np
import time
import threading

class ImagePublish_t_ros(Node):
    """
    将OpenCV图像发布到ROS 2话题(未压缩图像)
    :param topic: 话题名称
    :param queue_size: 消息队列大小
    """
    def __init__(self, topic: str, node_name="image_publisher"):
        super().__init__(node_name)
        self._topic = topic
        self._publisher = self.create_publisher(Image, topic, 10)
        self._bridge = CvBridge()
        self.get_logger().info(f"图像发布器初始化，话题: {topic}")
        
    def update(self, image: np.ndarray, content: dict = None):
        """
        将OpenCV图像发布到ROS 2话题
        :param image: OpenCV图像对象 (np.ndarray)
        :param content: 附加内容，如时间戳、坐标系等
        """
        try:
            msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            if content and 'timestamp' in content:
                msg.header.stamp = self.get_clock().now().to_msg()
            self._publisher.publish(msg)
            self.get_logger().debug(f"图像发布成功，尺寸: {image.shape}")
        except Exception as e:
            self.get_logger().error(f"图像发布失败: {str(e)}")

class ImageSubscribe_t_ros(Node):
    """
    订阅ROS 2图像话题并转为OpenCV图像
    """
    def __init__(self, topic: str, node_name="image_subscriber"):
        super().__init__(node_name)
        self._topic = topic
        self._subscriber = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )
        self._bridge = CvBridge()
        self.latest_image = None  # 存储最新的图像
        self.get_logger().info(f"图像订阅器初始化，话题: {topic}")

    def image_callback(self, msg: Image):
        """回调函数：将ROS图像转为OpenCV图像"""
        try:
            # 将ROS的Image消息转为OpenCV的np.ndarray
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_image
            self.get_logger().debug(f"接收到图像，尺寸: {cv_image.shape}")
            # 在这里可以添加图像处理逻辑
            # cv2.imshow("Received Image", cv_image)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")

