# 导入所需的库
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import time
import cv2
import numpy as np
from cv_lib.ros.cv_bridge import ImagePublish_t, ImageReceive_t
from PoseSolver.Aruco import Aruco
from PoseSolver.PoseSolver import PoseSolver
from YOLOv11.yolo_lib import MyYOLO

def main():
    # 相机内参矩阵
    camera_matrix = np.array([
        [606.634521484375, 0, 433.2264404296875],
        [0, 606.5910034179688, 247.10369873046875],
        [0.000000, 0.000000, 1.000000]
    ], dtype=np.float32)
    
    # 畸变系数
    dist_coeffs = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
    
    # 创建处理管道
    pipe = []
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(MyYOLO("/home/Elaina/yolo/best.pt", show=True))
    pipe.append(ImagePublish_t("yolo"))
    
    content = {}
    print_time = True
    
    while True:
        # 创建一个空的图像对象
        image = np.zeros((480, 640, 3), dtype=np.uint8)  # 修改为实际图像尺寸
        
        # 启动处理管道
        for p in pipe:
            if print_time:
                start_time = time.time()
            
            p.update(image, content)
            
            if print_time:
                end_time = time.time()
                print(f"name:{type(p).__name__}: {(end_time - start_time)*1000:.2f} ms")
        
        # 处理并显示角点信息
        if "corners" in content and len(content["corners"]) > 0:
            print("\n" + "="*50)
            print(f"检测到 {len(content['corners'])} 个高置信度目标:")
            
            for i, corner_data in enumerate(content["corners"]):
                corners = corner_data["corners"]
                conf = corner_data["confidence"]
                
                print(f"\n目标 {i+1} (置信度: {conf:.2f}):")
                print(f"左上: ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
                print(f"右上: ({corners[1][0]:.1f}, {corners[1][1]:.1f})") 
                print(f"右下: ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
                print(f"左下: ({corners[3][0]:.1f}, {corners[3][1]:.1f})")
            
            print("="*50 + "\n")
            
            # 清空当前帧的角点数据，避免重复处理
            content["corners"] = []
        
        # 添加适当的延迟或退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()