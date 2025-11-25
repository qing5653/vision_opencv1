import cv2
import time
import os
import re
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import multiprocessing

# ------------------------------
# 核心配置参数（新增总播放时长，删除冗余参数）
# ------------------------------
CONFIG = {
    "aruco_dict_type": "DICT_7X7_1000",
    "physical_size_cm": 15,
    "dpi": 300,
    "save_dir": "./new_aruco_markers",
    "status_map": {"00": "空", "01": "R1KFS", "10": "R2KFS", "11": "假KFS"},
    "reverse_status_map": {"空": "00", "R1": "01", "R2": "10", "假": "11"},
    "camera_index": 10,
    "camera_width": 320,
    "camera_height": 240,
    "camera_fps": 120,
    "marker_length": 0.15,
    "stable_threshold": 3,
    "total_play_duration_ms": 200,  # 4个码总播放时长（固定200ms）
    "final_pause_ms": 200,          # 最后停留时间（可选，不包含在200ms内）
}

# ------------------------------
# 合并核心工具类（无修改）
# ------------------------------
class KFSArucoCore:
    def __init__(self):
        # 编码解码相关
        self.status_map = CONFIG["status_map"]
        self.reverse_status_map = CONFIG["reverse_status_map"]
        self.marker_binaries = {1: None, 2: None, 3: None, 4: None}
        self.position_states = {i: "未知" for i in range(1, 13)}
        self.unrecognized_counters = {i: 0 for i in range(1, 13)}
        
        # Aruco检测相关
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, CONFIG["aruco_dict_type"])
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.minMarkerPerimeterRate = 0.003
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector_params.adaptiveThreshConstant = 7
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    # 编码：状态→10位二进制
    def encode_states(self, input_states: List[str]) -> List[str]:
        if len(input_states) != 12:
            raise ValueError("必须输入12个位置的状态")
        
        valid_states = list(self.reverse_status_map.keys())
        for i, state in enumerate(input_states):
            if state not in valid_states:
                raise ValueError(f"位置{i+1}无效状态：{state}（有效：{valid_states}）")
        
        # 3个位置一组，生成4个10位二进制
        groups = [input_states[i*3:(i+1)*3] for i in range(4)]
        prefixes = ["11", "00", "01", "10"]
        binary_strings = []
        
        for i, (group, prefix) in enumerate(zip(groups, prefixes)):
            group_bin = "".join([self.reverse_status_map[s] for s in group])
            full_bin = prefix + group_bin + "00"  # 前8位+补02位
            binary_strings.append(full_bin)
            print(f"✅ 编码{i+1}号：{full_bin}（位置{i*3+1}-{i*3+3}）")
        
        return binary_strings

    # 解码：Marker ID→位置状态
    def decode_markers(self, marker_ids: List[int]) -> Dict[int, str]:
        current_bin = self.marker_binaries.copy()
        
        for marker_id in marker_ids:
            try:
                bin_10bit = bin(marker_id)[2:].zfill(10)[:8]  # 取前8位
                prefix = bin_10bit[:2]
                seq = {"11":1, "00":2, "01":3, "10":4}.get(prefix)
                if seq:
                    current_bin[seq] = bin_10bit
            except Exception as e:
                print(f"⚠️ 解析ID={marker_id}失败：{e}")
        
        self.marker_binaries = current_bin
        self._parse_pos_states()
        return self.position_states

    # 检测：图像→Marker ID列表
    def detect_markers(self, frame: np.ndarray) -> List[int]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        marker_ids = [int(id_) for id_ in ids.flatten()] if ids is not None else []
        if marker_ids:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return marker_ids

    # 内部：解析位置状态
    def _parse_pos_states(self):
        current_pos = {}
        # 4个码对应12个位置
        for seq in range(1,5):
            if not self.marker_binaries[seq]:
                continue
            bin_data = self.marker_binaries[seq][2:]  # 去掉前缀
            pos_start = (seq-1)*3 + 1
            for i in range(3):
                pos = pos_start + i
                bit_str = bin_data[i*2:(i+1)*2] if len(bin_data)>=i*2+2 else ""
                current_pos[pos] = self.status_map.get(bit_str, "无效") if len(bit_str)==2 else "无效"
        
        # 稳定性过滤
        for pos in range(1,13):
            if pos in current_pos:
                self.position_states[pos] = current_pos[pos]
                self.unrecognized_counters[pos] = 0
            else:
                self.unrecognized_counters[pos] += 1
                if self.unrecognized_counters[pos] >= CONFIG["stable_threshold"]:
                    self.position_states[pos] = "未知"

# ------------------------------
# 简化Aruco生成函数（无修改）
# ------------------------------
def generate_aruco(binary_str: str) -> str:
    """根据10位二进制生成Aruco码"""
    if len(binary_str)!=10 or not all(c in "01" for c in binary_str):
        raise ValueError(f"无效二进制：{binary_str}（需10位01）")
    
    marker_id = int(binary_str, 2)
    if marker_id > 999:
        raise ValueError(f"ID={marker_id}超过DICT_7X7_1000上限（999）")
    
    # 生成图像
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    marker_size = int(CONFIG["physical_size_cm"] * CONFIG["dpi"] / 2.54)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG["aruco_dict_type"]))
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, borderBits=1)
    
    # 保存路径
    prefix = binary_str[:2]
    seq = {"11":1, "00":2, "01":3, "10":4}[prefix]
    save_path = os.path.join(CONFIG["save_dir"], f"aruco_{binary_str}_id{marker_id}_seq{seq}.png")
    
    try:
        Image.fromarray(img).save(save_path, dpi=(CONFIG["dpi"], CONFIG["dpi"]))
    except:
        cv2.imwrite(save_path, img)
    print(f"📁 生成Aruco：{os.path.basename(save_path)}")
    return save_path

# ------------------------------
# 独立进程播放Aruco（核心修改：总时长200ms分配给4个码）
# ------------------------------
def play_aruco_process(aruco_paths: List[str]):
    """Aruco播放函数（独立进程，4个码总播放200ms）"""
    # 加载并缩放图像
    screen_w, screen_h = _get_screen_res()
    target_size = int(CONFIG["physical_size_cm"] * _get_pixel_per_cm(screen_w, screen_h))
    aruco_imgs = []
    for path in aruco_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 播放进程：无法加载{path}")
            return
        aruco_imgs.append(cv2.resize(img, (target_size, target_size), cv2.INTER_NEAREST))
    
    # 关键：总时长200ms，平均分配给4个码（每个码50ms）
    total_duration = CONFIG["total_play_duration_ms"] / 1000  # 200ms → 0.2秒
    num_markers = len(aruco_imgs)
    if num_markers != 4:
        print(f"⚠️  检测到{num_markers}个Aruco码（预期4个），总时长仍按200ms分配")
    single_duration = total_duration / num_markers  # 每个码的精准时长（50ms）
    
    # 初始化播放窗口
    window_name = "Aruco Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)
    x = (screen_w - target_size) // 2
    y = (screen_h - target_size) // 2
    blank_bg = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

    print(f"📽️  播放进程启动：共{num_markers}个码，总时长{total_duration*1000:.0f}ms（每个{single_duration*1000:.0f}ms）")
    print(f"📍 显示位置：屏幕中央（{x},{y}）")
    
    try:
        # 只播放一次（精准控制总时长）
        start_total = time.time()  # 记录总播放开始时间
        for i, img in enumerate(aruco_imgs):
            # 叠加Aruco到背景
            frame = blank_bg.copy()
            frame[y:y+target_size, x:x+target_size] = img
            
            # 快速刷新窗口（1ms监听按键，不占用播放时长）
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            
            # 打印播放信息
            bin_str = os.path.basename(aruco_paths[i]).split("_")[1]
            print(f"▶️  正在播放：{bin_str}（{i+1}/{num_markers}）")
            
            # 精准控制当前码的播放时长（避免累积误差）
            start_single = time.time()
            while time.time() - start_single < single_duration:
                # 循环刷新窗口，避免画面冻结
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
        
        # 计算实际总播放时长（验证是否符合预期）
        actual_total = (time.time() - start_total) * 1000
        print(f"⏱️  实际总播放时长：{actual_total:.0f}ms（预期{CONFIG['total_play_duration_ms']}ms）")
        
        # 最后停留（可选，不包含在200ms内）
        if CONFIG["final_pause_ms"] > 0:
            print(f"⏸️  播放完成，额外停留{CONFIG['final_pause_ms']}ms...")
            start_pause = time.time()
            while time.time() - start_pause < CONFIG["final_pause_ms"] / 1000:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
        
    finally:
        cv2.destroyAllWindows()
        print("🗑️  播放进程退出（已完成一次播放）")

# 播放进程辅助函数（无修改）
def _get_screen_res() -> Tuple[int, int]:
    """获取屏幕分辨率"""
    try:
        output = os.popen("xrandr").read()
        match = re.search(r"current (\d+) x (\d+)", output)
        return (int(match.group(1)), int(match.group(2))) if match else (1920, 1080)
    except:
        return (1920, 1080)

def _get_pixel_per_cm(screen_w: int, screen_h: int) -> float:
    """计算像素密度（像素/厘米）"""
    screen_size_inch = 15.6  # 可根据实际修改
    diagonal_px = np.sqrt(screen_w**2 + screen_h**2)
    return diagonal_px / (screen_size_inch * 2.54)

# ------------------------------
# 主流程（无修改，仅适配配置参数）
# ------------------------------
def main():
    print("="*60)
    print("📋 KFS-Aruco 编码-识别系统（总时长200ms版）")
    print("="*60)
    print(f"有效状态：{list(CONFIG['reverse_status_map'].keys())} | 格式：12个状态空格分隔")
    print(f"提示：4个Aruco码总播放200ms（每个50ms），播放完成后自动关闭窗口")
    print("="*60)
    
    # 1. 输入12个状态
    core = KFSArucoCore()
    while True:
        input_states = input("请输入12个位置状态：").strip().split()
        if len(input_states) == 12:
            try:
                for s in input_states:
                    if s not in CONFIG["reverse_status_map"]:
                        raise ValueError(f"无效状态：{s}")
                break
            except ValueError as e:
                print(f"❌ {e}")
        else:
            print(f"❌ 需12个状态（当前{len(input_states)}个）")
    
    # 2. 编码生成Aruco
    print("\n🔧 编码生成Aruco码...")
    try:
        binary_strs = core.encode_states(input_states)
        aruco_paths = [generate_aruco(bin_str) for bin_str in binary_strs]
    except Exception as e:
        print(f"❌ 生成失败：{e}")
        return
    
    # 3. 启动摄像头
    print("\n📹 启动摄像头...")
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, CONFIG["camera_fps"])  # 按配置设置FPS
    
    if not cap.isOpened():
        print("❌ 摄像头启动失败（检查camera_index是否正确）")
        return
    print(f"✅ 摄像头就绪：{CONFIG['camera_width']}×{CONFIG['camera_height']} @ {CONFIG['camera_fps']}FPS")
    
    # 4. 启动独立进程播放Aruco（总时长200ms）
    print("\n📽️  启动Aruco单次播放...")
    play_process = multiprocessing.Process(target=play_aruco_process, args=(aruco_paths,))
    play_process.start()
    
    # 5. 识别解码主循环
    print("\n🔍 开始识别解码（播放完成后仍可继续识别，按'q'退出）")
    last_result = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取摄像头画面")
            time.sleep(0.5)
            continue
        
        # 检测+解码
        marker_ids = core.detect_markers(frame)
        current_states = core.decode_markers(marker_ids)
        
        # 显示识别信息
        msg = f"IDs: {marker_ids}" if marker_ids else "No markers"
        cv2.putText(frame, msg, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        # 打印变化结果
        current_result = str([(pos, current_states[pos]) for pos in range(1,13)])
        if current_result != last_result:
            last_result = current_result
            print("\n🔍 解码结果：")
            for pos in range(1,13):
                print(f"  位置{pos}：{current_states[pos]}")
        
        # 显示摄像头画面
        cv2.imshow("Camera Detection", frame)
        
        # 退出逻辑
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 正在退出...")
            if play_process.is_alive():
                play_process.terminate()
            play_process.join()
            break
        
        # 播放进程结束后提示
        if not play_process.is_alive() and not hasattr(main, "play_ended_flag"):
            main.play_ended_flag = True
            print("\n📢 Aruco码单次播放已完成！")
            print("💡 可继续移动摄像头对准已生成的Aruco图像进行识别，按'q'退出")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 程序完全退出")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序中断")
    except Exception as e:
        print(f"\n❌ 异常退出：{e}")
