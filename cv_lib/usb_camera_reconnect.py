import os
import subprocess
import time
import rclpy

class USBCameraReconnect:
    def __init__(self, node, camera_device="/dev/video10", usb_hub_pci="0000:00:14.0", sudo_password="qing"):
        self.node = node  # 接收主节点的logger
        self.camera_device = camera_device
        self.usb_hub_pci = usb_hub_pci
        self.sudo_password = sudo_password
        self.reconnect_count = 0
        self.max_reconnect = 3
        # 重连脚本路径（需与主节点脚本位置对应）
        self.script_dir = os.path.join(os.path.dirname(__file__), "../cv_lib/")
        self.unbind_script = os.path.join(self.script_dir, "usb_unbind.sh")
        self.bind_script = os.path.join(self.script_dir, "usb_bind.sh")

    def is_camera_online(self):
        """检查摄像头设备是否存在"""
        return os.path.exists(self.camera_device)

    def reset_usb_hub(self):
        """重置USB Hub以恢复摄像头连接"""
        self.node.get_logger().warn(f"⚠️ 重置USB Hub（PCI：{self.usb_hub_pci}）")
        if not os.path.exists(self.unbind_script) or not os.path.exists(self.bind_script):
            self.node.get_logger().error(f"❌ 未找到重连脚本，路径：{self.script_dir}")
            return False
        try:
            # 卸载USB Hub
            cmd_unbind = f"echo '{self.sudo_password}' | sudo -S sh {self.unbind_script} {self.usb_hub_pci}"
            subprocess.run(cmd_unbind, shell=True, check=True, capture_output=True, text=True)
            self.node.get_logger().info("✅ USB Hub卸载成功")
            time.sleep(2)
            # 重新绑定USB Hub
            cmd_bind = f"echo '{self.sudo_password}' | sudo -S sh {self.bind_script} {self.usb_hub_pci}"
            subprocess.run(cmd_bind, shell=True, check=True, capture_output=True, text=True)
            self.node.get_logger().info("✅ USB Hub绑定成功")
            time.sleep(3)
            return True
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"❌ USB重置失败：{e.stderr}")
            return False

    def restart_usb_cam_node(self):
        """重启usb_cam节点"""
        try:
            cmd = f"echo '{self.sudo_password}' | sudo -S ros2 service call /usb_cam_node/reset std_srvs/srv/Empty"
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            time.sleep(2)
            self.node.get_logger().info("✅ usb_cam节点重启成功")
            return True
        except Exception as e:
            self.node.get_logger().error(f"❌ 节点重启失败：{str(e)}")
            return False

    def reconnect(self):
        """统一重连入口：先检查设备，再执行对应重连逻辑"""
        self.reconnect_count += 1
        if self.reconnect_count > self.max_reconnect:
            self.node.get_logger().error(f"❌ 重连失败（{self.max_reconnect}次上限），检查USB线/端口/摄像头")
            return False

        self.node.get_logger().warn(f"⚠️ 第{self.reconnect_count}次重连摄像头...")
        if self.is_camera_online():
            return self.restart_usb_cam_node()
        else:
            if self.reset_usb_hub() and self.is_camera_online():
                self.reconnect_count = 0  # 重连成功重置计数器
                self.node.get_logger().info(f"✅ 摄像头恢复：{self.camera_device}")
                return True
            else:
                self.node.get_logger().error(f"❌ 重置后仍未找到{self.camera_device}")
                return False
