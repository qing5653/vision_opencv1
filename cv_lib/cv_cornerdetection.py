import cv2
import time

class BRISKCornerDetector:
    def __init__(self, show_result=True):
        """
        BRISK角点检测器
        :param show_result: 是否实时显示角点检测结果
        """
        self.brisk = cv2.BRISK_create()
        self.show_result = show_result

    def detect(self, image):
        """
        执行角点检测
        :param image: 输入图像（BGR格式）
        :return: (keypoints, descriptors)
        """
        if image is None or image.size == 0:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.brisk.detectAndCompute(gray, None)

        if self.show_result:
            img_kp = cv2.drawKeypoints(
                image, keypoints, None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imshow("BRISK Corners", img_kp)
            cv2.waitKey(1)

        return keypoints, descriptors

def main():
    # 初始化检测器
    detector = BRISKCornerDetector(show_result=True)
    
    # 读取测试图像
    image = cv2.imread("example.jpg")
    if image is None:
        print("错误：无法加载图像！")
        return

    # 执行角点检测
    start_time = time.time()
    keypoints, descriptors = detector.detect(image)
    processing_time = (time.time() - start_time) * 1000

    # 打印结果
    print(f"检测到 {len(keypoints)} 个角点")
    print(f"处理时间: {processing_time:.2f} ms")

    # 等待退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()