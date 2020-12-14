#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 30 00:37:11 2019

@author: Simon
"""

import cv2
import numpy as np


class Vision:

    def Face_Detect(self):
        cv2.namedWindow('Face_Detect')  # 定义一个窗口
        cap = cv2.VideoCapture(0)  # 捕获摄像头图像

        # 判断视频是否打开

        if cap.isOpened():
            print ('Open')
        else:
            print ('camra is not opened')

        (success, frame) = cap.read()  # 读入第一帧

        classifier = \
            cv2.CascadeClassifier("opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml")

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
                    with open('x_data(for_training).txt', 'a') as f:
                        f.write(x_data)

                    with open('y_data(for_training).txt', 'a') as f:
                        f.write(y_data)

                    with open('x.txt', 'w') as f:
                        f.write(x_data)

                    with open('y.txt', 'w') as f:
                        f.write(y_data)

                    print (x, y, w, h)
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

    def face_calibration(self):
        nearest = int()
        fearest = int()
        cv2.namedWindow('face_calibration')  # 定义一个窗口
        cap = cv2.VideoCapture(1)  # 捕获摄像头图像

        # 判断视频是否打开

        if cap.isOpened():
            print ('Open')
        else:
            print ('camra is not opened')

        (success, frame) = cap.read()  # 读入第一帧

        classifier = \
            cv2.CascadeClassifier("opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml")


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
                    print (x, y, w, h)
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
            

                        
                        
            if cv2.waitKey(1) == ord('p'):
                key = cv2.waitKey(0)
                if key == ord('y'):
                    fearest = h
                    print ('fearest')
                if key == ord('j'):
                    nearest = h
                    print ('nearest',nearest)
            
            
                    
            cv2.imshow('face_calibration', frame)
                    
            if len(faceRects) <= 0:
                print("not face there")

         # 显示轮廓
            
         
            (success, frame) = cap.read()  # 如正常则读入下一帧
            
            if cv2.waitKey(1) == ord('q'):
                break
            if nearest > 0 and fearest > 0:
                break
            elif nearest == 0 and fearest > 0:
                print("fearest point not found")
            elif nearest > 0 and fearest == 0:
                print("nearest point not found")
            else:
                print("fearest and nearest point not found")
         # 循环结束则清零

        cap.release()
        cv2.destroyAllWindows()
        return (nearest, fearest)
    
    def get_mid_point(self, nearest, fearest):
        mid_point = (nearest+fearest)/2
        return mid_point
    
    def range_swich(self, mid_point, h):
        swich = False
        if h >= mid_point:
            swich = True
        else:
            swich = False
            
        return swich
    

    
if __name__ == '__main__':
    vision = Vision()
    vision.Face_Detect()
    nearest,fearest = vision.face_calibration()
    print (vision.get_mid_point(nearest,fearest))