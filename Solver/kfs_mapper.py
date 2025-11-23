import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String

class KFSMapperNode(Node):
    def __init__(self):
        super().__init__('kfs_mapper_node')
        
        # 核心存储：4个Aruco码的前8位二进制
        self.marker_binaries = {1: None, 2: None, 3: None, 4: None}
        # 12个位置的最终状态
        self.position_states = {i: "未知" for i in range(1, 13)}
        # 状态映射规则
        self.status_map = {"00": "空", "01": "R1KFS", "10": "R2KFS", "11": "假KFS"}
        
        # 稳定性过滤
        self.unrecognized_counters = {i: 0 for i in range(1, 13)}
        self.stable_threshold = 3
        
        self.last_parsed_result = None
        
        # 订阅Aruco识别结果
        self.aruco_sub = self.create_subscription(
            MarkerArray, '/aruco_markers', self.aruco_callback, 10
        )
        # 发布状态话题
        self.state_pub = self.create_publisher(String, "/kfs_states", 10)
        
        self.get_logger().info("✅ KFS状态解析节点启动（10位ID→前8位，4码对应12位置）")

    def aruco_callback(self, msg):
        """接收Aruco码，更新4个码的前8位二进制"""
        # 临时存储当前帧识别到的码
        current_frame_binaries = self.marker_binaries.copy()
        
        for marker in msg.markers:
            marker_id = marker.id
            try:
                # 10位ID转二进制（补0至10位），提取前8位
                binary_str_10bit = bin(marker_id)[2:].zfill(10)
                first_8bit = binary_str_10bit[:8]
                
                # 按前2位判断码序号（11=1号，00=2号，01=3号，10=4号）
                prefix = first_8bit[:2]
                seq = None
                if prefix == '11':
                    seq = 1
                elif prefix == '00':
                    seq = 2
                elif prefix == '01':
                    seq = 3
                elif prefix == '10':
                    seq = 4
                
                # 仅更新有效序号的码（1-4），且前8位变化时才更新
                if seq in [1,2,3,4] and first_8bit != current_frame_binaries[seq]:
                    current_frame_binaries[seq] = first_8bit
                    self.get_logger().debug(f"📥 更新{seq}号码：前8位={first_8bit}（ID={marker_id}）")
            
            except Exception as e:
                self.get_logger().warn(f"⚠️ 解析ID={marker_id}失败：{str(e)}")
                continue
        
        # 更新全局存储
        self.marker_binaries = current_frame_binaries
        # 合并解析12个位置状态
        self.merge_and_parse()

    def merge_and_parse(self):
        """合并4个码的信息，解析12个位置状态"""
        # 临时存储当前解析的位置状态
        current_pos_states = {}
        
        # 1号码：前8位后6位 → 位置1-3
        if self.marker_binaries[1]:
            bin1 = self.marker_binaries[1][2:]  # 去掉前2位前缀
            current_pos_states[1] = self.get_status(bin1[:2]) if len(bin1)>=2 else "无效"
            current_pos_states[2] = self.get_status(bin1[2:4]) if len(bin1)>=4 else "无效"
            current_pos_states[3] = self.get_status(bin1[4:6]) if len(bin1)>=6 else "无效"
        # 2号码：前8位后6位 → 位置4-6
        if self.marker_binaries[2]:
            bin2 = self.marker_binaries[2][2:]
            current_pos_states[4] = self.get_status(bin2[:2]) if len(bin2)>=2 else "无效"
            current_pos_states[5] = self.get_status(bin2[2:4]) if len(bin2)>=4 else "无效"
            current_pos_states[6] = self.get_status(bin2[4:6]) if len(bin2)>=6 else "无效"
        # 3号码：前8位后6位 → 位置7-9
        if self.marker_binaries[3]:
            bin3 = self.marker_binaries[3][2:]
            current_pos_states[7] = self.get_status(bin3[:2]) if len(bin3)>=2 else "无效"
            current_pos_states[8] = self.get_status(bin3[2:4]) if len(bin3)>=4 else "无效"
            current_pos_states[9] = self.get_status(bin3[4:6]) if len(bin3)>=6 else "无效"
        # 4号码：前8位后6位 → 位置10-12
        if self.marker_binaries[4]:
            bin4 = self.marker_binaries[4][2:]
            current_pos_states[10] = self.get_status(bin4[:2]) if len(bin4)>=2 else "无效"
            current_pos_states[11] = self.get_status(bin4[2:4]) if len(bin4)>=4 else "无效"
            current_pos_states[12] = self.get_status(bin4[4:6]) if len(bin4)>=6 else "无效"
        
        # 稳定性过滤：处理已识别/未识别的位置
        for pos in range(1, 13):
            if pos in current_pos_states:
                # 该位置已识别：更新状态，重置未识别计数器
                self.position_states[pos] = current_pos_states[pos]
                self.unrecognized_counters[pos] = 0
            else:
                # 该位置未识别：计数器累加，超过阈值才置为"未知"
                self.unrecognized_counters[pos] += 1
                if self.unrecognized_counters[pos] >= self.stable_threshold:
                    self.position_states[pos] = "未知"
        
        # 仅结果变化时输出日志（避免刷屏）
        current_result_str = str([(pos, self.position_states[pos]) for pos in range(1,13)])
        if current_result_str != self.last_parsed_result:
            self.last_parsed_result = current_result_str
            # 打印12个位置状态
            self.get_logger().info("🔍 当前12个位置状态：")
            for pos in range(1, 13):
                self.get_logger().info(f"位置{pos}：{self.position_states[pos]}")
            # 发布状态话题
            state_str = ",".join([f"位置{i}:{self.position_states[i]}" for i in range(1,13)])
            self.state_pub.publish(String(data=state_str))

    def get_status(self, bit_str):
        """2位二进制转状态"""
        # 过滤非2位的无效输入
        if len(bit_str) != 2 or not all(c in ['0','1'] for c in bit_str):
            return "无效"
        return self.status_map.get(bit_str, "无效")

def main(args=None):
    rclpy.init(args=args)
    node = KFSMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 KFS状态解析节点已停止")
    finally:
        node.destroy_node()
if __name__ == '__main__':
    main()
