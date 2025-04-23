import cv2
import numpy as np
import time

class OpenCVObjectDetector:
    def __init__(self, model_cfg='yolov3.cfg', model_weights='yolov3.weights', 
                 classes_file='coco.names', conf_threshold=0.5, nms_threshold=0.4,
                 show=True):
        """
        使用OpenCV的dnn模块实现目标检测
        :param model_cfg: 模型配置文件路径
        :param model_weights: 模型权重文件路径
        :param classes_file: 类别名称文件路径
        :param conf_threshold: 置信度阈值
        :param nms_threshold: 非极大值抑制阈值
        :param show: 是否显示检测结果
        """
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        with open(classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.show = show
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        
    def update(self, image, content):
        """
        执行目标检测
        :param image: 输入图像
        :param content: 存储结果的字典
        """
        if image is None or image.size == 0:
            return
            
        (H, W) = image.shape[:2]
        
        # 获取输出层
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # 构建blob并前向传播
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)
        
        # 初始化检测结果
        boxes = []
        confidences = []
        classIDs = []
        
        # 解析输出
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                if confidence > self.conf_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
        # 应用非极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        # 存储结果
        content['detections'] = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                content['detections'].append({
                    'class': self.classes[classIDs[i]],
                    'confidence': confidences[i],
                    'box': boxes[i]
                })
        
        # 可视化
        if self.show and len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = f"{self.classes[classIDs[i]]}: {confidences[i]:.2f}"
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("Object Detection", image)
            cv2.waitKey(1)

class BRISKCornerDetector:
    def __init__(self, show_result=False):
        self.brisk = cv2.BRISK_create()
        self.show_result = show_result

    def update(self, image, content):
        if image is None or image.size == 0:
            return
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.brisk.detectAndCompute(gray, None)
        
        content["brisk_keypoints"] = keypoints
        content["brisk_descriptors"] = descriptors
        
        if self.show_result:
            img_kp = cv2.drawKeypoints(
                image, keypoints, None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imshow("BRISK Corners", img_kp)
            cv2.waitKey(1)

def main():
    # 初始化处理管道
    pipe = []
    pipe.append(OpenCVObjectDetector(show=True))
    pipe.append(BRISKCornerDetector(show_result=True))
    
    content = {}
    
    # 测试图像
    image = cv2.imread("example.jpg")
    if image is None:
        print("无法加载测试图像")
        return
    
    # 处理管道
    for p in pipe:
        start_time = time.time()
        p.update(image, content)
        print(f"{type(p).__name__} processing time: {(time.time() - start_time)*1000:.2f} ms")
    
    # 打印结果
    print("\nDetection Results:")
    for det in content.get('detections', []):
        print(f"- {det['class']}: confidence={det['confidence']:.2f}, box={det['box']}")
    
    print(f"\nDetected {len(content.get('brisk_keypoints', []))} corners")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()