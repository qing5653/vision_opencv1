from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path


class MyYOLO():
    def __init__(self, model_path, show=False, use_intel=False):
        self.model = YOLO(model_path)
        self.show = show
        if use_intel:
            import openvino.runtime as ov
            from openvino.runtime import Core
            import openvino.properties.hint as hints
            self.model = YOLO(Path(model_path).parent)
            config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
            core = Core()
            model = core.read_model(model_path)
            quantized_seg_compiled_model = core.compile_model(model, config=config)
            if self.model.predictor is None:
                custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
                args = {**self.model.overrides, **custom}
                self.model.predictor = self.model._smart_load("predictor")(overrides=args, _callbacks=self.model.callbacks)
                self.model.predictor.setup_model(model=self.model.model)
            self.model.predictor.model.ov_compiled_model = quantized_seg_compiled_model

    def update(self, image: np.ndarray, content: dict):
        results = self.model(image)
        content["corners"] = []

        for result in results:
            if result.masks is None:
                continue

            # 安全获取置信度
            conf = 0.0
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'conf'):
                conf_tensor = result.boxes.conf
                if conf_tensor is not None and len(conf_tensor) > 0:
                    conf = float(conf_tensor[0])

            if conf >= 0.9:
                for mask in result.masks.xy:
                    try:
                        # 转换为二维数组 (N,2)
                        mask_points = np.array(mask, dtype=np.float32).reshape(-1, 2)
                        if len(mask_points) < 4:
                            continue

                        # 使用优化的角点检测方法
                        final_corners = self._optimized_corner_detection(mask_points, image.shape[:2])

                        content["corners"].append({
                            "corners": final_corners,
                            "confidence": conf,
                            "raw_points": mask_points
                        })
                    except Exception as e:
                        print(f"角点检测出错: {str(e)}")
                        continue

        # 可视化结果
        if self.show and len(results) > 0:
            self._visualize_results(results[0], image, content)

    def _optimized_corner_detection(self, points: np.ndarray, image_shape) -> np.ndarray:
        """优化的角点检测方法，结合轮廓分析和透视变换原理"""
        # 1. 创建掩膜并查找轮廓
        mask_img = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [points.astype(np.int32)], 255)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((4, 2), dtype=np.float32)
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 2. 应用Douglas-Peucker算法进行多边形逼近
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果逼近结果刚好是四边形，直接使用
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)
        
        # 3. 计算凸包
        hull = cv2.convexHull(largest_contour)
        
        # 4. 使用改进的角点检测方法
        if len(hull) >= 4:
            # 使用Shi-Tomasi角点检测
            corners = cv2.goodFeaturesToTrack(mask_img, 4, 0.01, 10)
            if corners is not None:
                corners = corners.reshape(-1, 2)
                
                # 确保有4个角点
                if len(corners) == 4:
                    return self._sort_corners(corners)
        
        # 5. 作为后备，使用方向投影法
        contour_points = largest_contour.reshape(-1, 2)
        center = np.mean(contour_points, axis=0)
        
        # 定义四个方向：上、下、左、右
        directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        
        extreme_points = []
        for dir in directions:
            projections = np.dot(contour_points - center, dir)
            idx = np.argmax(projections)
            extreme_points.append(contour_points[idx])
        
        return self._sort_corners(np.array(extreme_points, dtype=np.float32))

    def _sort_corners(self, corners):
        """将四个点按左上、右上、右下、左下排序"""
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 计算每个点与中心点的夹角
        angles = []
        for pt in corners:
            dx = pt[0] - center[0]
            dy = pt[1] - center[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # 根据夹角排序角点
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        return sorted_corners

    def _visualize_results(self, result, image, content):
        """可视化检测结果"""
        try:
            image_result = result.plot()

            # 绘制极值点
            for corner_data in content["corners"]:
                confidence = corner_data["confidence"]
                if confidence >= 0.9:  # 只绘制置信度满足要求的掩膜
                    corners = corner_data["corners"].astype(np.int32)

                    # 绘制连接线
                    cv2.polylines(image_result, [corners.reshape((-1, 1, 2))],
                                  True, (255, 0, 255), 2)

                    # 标记四个极值点（不同颜色）
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
                    labels = ["TL", "TR", "BR", "BL"]  # 左上、右上、右下、左下
                    for i, pt in enumerate(corners):
                        cv2.circle(image_result, tuple(pt), 8, colors[i], -1)
                        cv2.putText(image_result, labels[i],
                                    (pt[0] + 10, pt[1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            image[:] = image_result
        except Exception as e:
            print(f"可视化出错: {str(e)}")
