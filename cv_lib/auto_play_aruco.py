import os
import re
import time
import cv2
import numpy as np

def get_screen_resolution():
    """自动获取当前屏幕分辨率"""
    try:
        import subprocess
        output = subprocess.check_output(["xrandr"]).decode("utf-8")
        for line in output.splitlines():
            if "current" in line:
                match = re.search(r"current (\d+) x (\d+)", line)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
    except:
        pass
    return (2560, 1600)  # 默认值

def get_screen_pixel_per_cm():
    screen_size_inch = 16.0
    screen_resolution = get_screen_resolution()  # 使用自动获取的分辨率
    print(f"🖥️  检测到屏幕分辨率：{screen_resolution[0]}×{screen_resolution[1]}")

    diagonal_pixels = np.sqrt(screen_resolution[0]**2 + screen_resolution[1]** 2)
    pixel_per_cm = diagonal_pixels / (screen_size_inch * 2.54)
    print(f"📏 屏幕像素密度：{pixel_per_cm:.1f}像素/厘米")
    return pixel_per_cm

def auto_play_aruco(
    relative_aruco_dir="./new_aruco_markers",
    total_duration_ms=200,
    target_physical_size_cm=15.0
):
    # 1. 计算目标尺寸
    pixel_per_cm = get_screen_pixel_per_cm()
    target_pixel_size = int(target_physical_size_cm * pixel_per_cm)
    print(f"🎯 目标显示尺寸：{target_physical_size_cm}cm × {target_physical_size_cm}cm（{target_pixel_size}px × {target_pixel_size}px）")

    # 2. 处理路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aruco_dir = os.path.join(script_dir, relative_aruco_dir)
    
    if not os.path.isdir(aruco_dir):
        print(f"❌ 目录不存在：{aruco_dir}")
        return

    # 3. 预加载并缩放图像
    aruco_files = []
    id_pattern = re.compile(r"aruco_bin_\d{10}_id_(\d+)_seq_\d+_15cm\.png")
    
    for filename in os.listdir(aruco_dir):
        match = id_pattern.match(filename)
        if match:
            file_path = os.path.join(aruco_dir, filename)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️  无法加载图像：{filename}")
                continue
            if np.sum(img) == 0:
                print(f"⚠️  图像{filename}为全黑")
                continue
            # 缩放至目标尺寸
            img = cv2.resize(img, (target_pixel_size, target_pixel_size), interpolation=cv2.INTER_NEAREST)
            aruco_files.append((int(match.group(1)), img, filename))
            print(f"✅ 图像{filename}已缩放到目标尺寸")

    if not aruco_files:
        print(f"❌ 未找到符合格式的Aruco码文件")
        return

    # 4. 排序并计算单张时长
    aruco_files.sort(key=lambda x: x[0])
    num_markers = len(aruco_files)
    single_duration_ms = total_duration_ms / num_markers
    print(f"📽️  开始播放（共{num_markers}张，总时长{total_duration_ms}ms，每张停留{single_duration_ms:.0f}ms）")
    print(f"📄 播放顺序：{[f[2] for f in aruco_files]}")

    # 5. 显示逻辑
    window_name = "Aruco Player (15cm×15cm)"
    # 关键优化：使用全屏无边框模式，禁用窗口装饰
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 预渲染空白帧
    screen_w, screen_h = get_screen_resolution()
    blank_img = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255  # 全屏白色背景
    cv2.imshow(window_name, blank_img)
    cv2.waitKey(200)  # 延长预渲染时间，确保窗口稳定

    # 计算码在全屏中的居中位置
    x = (screen_w - target_pixel_size) // 2
    y = (screen_h - target_pixel_size) // 2
    print(f"📍 码将显示在屏幕中央：({x}, {y})")

    # 高精度计时器
    start_time = time.perf_counter()
    planned_end_times = [start_time + (i+1)*single_duration_ms/1000 for i in range(num_markers)]

    try:
        for i, (_, img, filename) in enumerate(aruco_files):
            # 核心优化：在全屏白色背景上叠加码（避免窗口刷新冲突）
            frame = blank_img.copy()
            frame[y:y+target_pixel_size, x:x+target_pixel_size] = img  # 叠加码到中央
            
            # 显示并强制刷新
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)  # 最小延迟，确保立即刷新
            print(f"▶️  显示：{filename}")

            # 精准控制停留时间
            current_time = time.perf_counter()
            sleep_time = planned_end_times[i] - current_time

            if sleep_time > 0:
                cv2.waitKey(int(sleep_time * 1000))
            else:
                print(f"⚠️  延迟：{filename} 停留时间不足（{sleep_time*1000:.1f}ms）")

        # 最后停留2秒
        print("⏸️  最后一张图片停留2秒...")
        cv2.waitKey(2000)

    finally:
        # 清理窗口
        cv2.destroyAllWindows()
        print("🗑️  播放清理完成")

    # 输出总时长
    total_elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"✅ 播放完成！实际总时长：{total_elapsed_ms:.0f}ms（目标：{total_duration_ms}ms）")

if __name__ == "__main__":
    auto_play_aruco()
