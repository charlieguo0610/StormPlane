import cv2
import numpy as np
import time
import threading


class communication:
    def __init__(self, x: int, y: int):
        self.x_value = x
        self.y_value = y

    def write_x(self):
        # 存储为数据库
        with open("x_data(for_training).txt", "a") as f:
            f.write(self.x_value)
        with open("x.txt", "w") as f:
            f.write(self.x_value)
        with open("xb.txt", "w") as f:
            f.write(self.x_value)

    def write_y(self):
        with open("y_data(for_training).txt", "a") as f:
            f.write(self.y_value)

        with open("y.txt", "w") as f:
            f.write(self.y_value)

        with open("yb.txt", "w") as f:
            f.write(self.y_value)

    def get_x():
        try:
            with open("x.txt", 'r') as f:
                contents = f.read()
                return contents

        except:
            with open("xb.txt", 'r') as f:
                contents = f.read()
                return contents

    def get_y():
        try:
            with open("y.txt", 'r') as f:
                y_contents = f.read()
                return y_contents



        except:
            with open("yb.txt", 'r') as f:
                y_contents = f.read()
                return y_contents


class Vision(communication):

    def __init__(self, x: int, y: int):
        super().__init__(x, y)

    @staticmethod
    def Face_Detect():
        cv2.namedWindow('Face_Detect')  # 定义一个窗口
        cap = cv2.VideoCapture(0)  # 捕获摄像头图像

        # 判断视频是否打开

        if cap.isOpened():
            print('Open')
        else:
            print('camra is not opened')

        (success, frame) = cap.read()  # 读入第一帧

        classifier = \
            cv2.CascadeClassifier(
                r"C:\Users\Simon\Desktop\12Ucpt\opencvdata\data\haarcascades/haarcascade_frontalface_alt.xml"
                )

        # 定义人脸识别的分类数据集，需要自己查找，在opencv的目录下，参考上面我的路径**

        while success:  # 如果读入帧正常
            size = frame.shape[:2]
            image = np.zeros(size, dtype=np.float16)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)
            divisor = 8
            (h, w) = size
            minSize = (int(w / divisor), int(h / divisor))  # 像素一定是整数，或者用w//divisor

            faceRects = classifier.detectMultiScale(image, 1.2, 2,
                                                    cv2.CASCADE_SCALE_IMAGE, minSize)

            # 人脸识别

            if len(faceRects) > 0:
                for faceRect in faceRects:
                    (x, y, w, h) = faceRect
                    x_data = str(x)
                    y_data = str(x)
                    now = communication(x_data, y_data)
                    now.write_x()
                    now.write_y()
                    print(x, y, w, h)
                    cv2.circle(frame, (x + w // 2, y + h // 2), min(w
                                                                    // 2, h // 2), (0xFF, 0, 0), 2)  # 圆形轮廓
                    cv2.circle(frame, (x + w // 4, y + 2 * h // 5),
                               min(w // 8, h // 8), (0, 0xFF, 0), 2)  # 左眼轮廓
                    cv2.circle(frame, (x + 3 * w // 4, y + 2 * h // 5),
                               min(w // 8, h // 8), (0, 0xFF, 0), 2)  # 右眼轮廓
                    cv2.circle(frame, (x + w // 2, y + 2 * h // 3),
                               min(w // 8, h // 8), (0, 0xFF, 0), 2)  # 鼻子轮廓
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0,
                                                                  0xFF), 2)  # 矩形轮廓

                    # y通道B（冗余通道）

                    with open('yb.txt', 'w') as f:
                        f.write(str(y))

                    # x通道B（冗余通道）

                    with open('xb.txt', 'w') as f:
                        f.write(str(x))
            cv2.imshow('Face_Detect', frame)

            # 显示轮廓

            (success, frame) = cap.read()  # 如正常则读入下一帧

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 循环结束则清零

        cap.release()
        cv2.destroyAllWindows()


class MyThread(threading.Thread):
    def run(self):
        Vision.Face_Detect()


class My_secound_Thread(threading.Thread):
    def run(self):
        try:
            with open("x.txt", 'r') as f:
                contents = f.read()
                print(contents)

        except:
            with open("xb.txt", 'r') as f:
                contents = f.read()
                print(contents)


def main():
    print("Start main threading")
    # 创建三个线程
    vision = MyThread()
    game = My_secound_Thread()
    # 启动三个线程
    vision.start()
    game.start()
    vision.join()
    game.join()
    print("End Main threading")


if __name__ == '__main__':
    main()