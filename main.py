import sys
from PyQt5 import QtCore
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
import cv2
import copy
import time
import numpy as np
import os
import utils
from analyzer import Analyzer, PlotCanvas


SIZE = (640, 360)
IMG_SIZE = (300, 200)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('mainwindow.ui', self)

        self.timer_camera = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0

        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        self.history = 10
        self.bs.setHistory(self.history)

        self.analyzer = Analyzer()

        self.slot_init()
        self.is_first_frame = True
        self.current_frame = 0
        self.last_frame = None

        self.g_events.setFixedSize(*IMG_SIZE)
        self.g_gray.setFixedSize(*IMG_SIZE)
        self.g_binary.setFixedSize(*IMG_SIZE)

        self.g_events.move(10, 10)

        self.video_path = None

        self.m = PlotCanvas(self, width=8, height=2)
        self.m.move(10, 500)

    def slot_init(self):
        self.b_Camera.clicked.connect(
            self.button_open_camera_clicked)
        self.b_Video.clicked.connect(self.close_camera)

        self.timer_camera.timeout.connect(self.show_camera)
        self.b_Play.clicked.connect(self.select_video)
        self.timer_video.timeout.connect(self.show_video)

    def select_video(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self, "选取文件",
                                                                os.getcwd(),
                                                                "Videos(*.avi)")
        if fileName_choose == "":
            self.video_path = None
            return
        else:
            self.is_first_frame = True
            self.current_frame = 0
            self.last_frame = None
            self.video_path = fileName_choose
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer_video.start(200)

    def show_video(self):
        flag, self.image = self.cap.read()

        if not flag:
            self.timer_video.stop()
            self.cap.release()
            self.g_binary.clear()
            self.g_events.clear()
            self.g_gray.clear()
            self.analyzer.clean()

        self.current_frame += 1

        if self.current_frame <= self.history:
            fg_mask = self.bs.apply(self.image)
            fg_mask = np.zeros_like(fg_mask)
        else:
            fg_mask = self.bs.apply(self.image, learningRate=0.005)

        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        status = self.analyzer.add(image)
        self.m.update_figure(self.analyzer.ys)
        if status:
            self.l_status.setText("FALL!")
        else:
            self.l_status.setText("NORMAL")
        info = utils.get_box(image)
        if info:
            x_min, y_min, x_max, y_max = info
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 255, 2)

            self.image = cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), 255, 2)
            show = cv2.resize(self.image, IMG_SIZE)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.g_events.setPixmap(QtGui.QPixmap.fromImage(showImage))

            gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            showImage = QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                     QtGui.QImage.Format_Indexed8)
            self.g_gray.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            show = cv2.resize(self.image, IMG_SIZE)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.g_events.setPixmap(QtGui.QPixmap.fromImage(showImage))

            gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            showImage = QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                     QtGui.QImage.Format_Indexed8)
            self.g_gray.setPixmap(QtGui.QPixmap.fromImage(showImage))
        show = cv2.resize(image, IMG_SIZE)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_Indexed8)
        self.g_binary.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def close_camera(self):
        self.timer_camera.stop()
        self.cap.release()
        self.g_binary.clear()
        self.g_events.clear()
        self.g_gray.clear()

    def show_camera(self):
        flag, self.image = self.cap.read()

        if not flag:
            self.timer_video.stop()
            self.cap.release()
            self.g_binary.clear()
            self.g_events.clear()
            self.g_gray.clear()
            self.analyzer.clean()

        self.current_frame += 1

        show = cv2.resize(self.image, IMG_SIZE)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.g_events.setPixmap(QtGui.QPixmap.fromImage(showImage))

        gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
        showImage = QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                 QtGui.QImage.Format_Indexed8)
        self.g_gray.setPixmap(QtGui.QPixmap.fromImage(showImage))

        if self.current_frame <= self.history:
            fg_mask = self.bs.apply(self.image)
            fg_mask = np.zeros_like(fg_mask)
        else:
            fg_mask = self.bs.apply(self.image, learningRate=0.01)

        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        status = self.analyzer.add(image)
        self.m.update_figure(self.analyzer.ys)
        if status:
            self.l_status.setText("FALL!")
        else:
            self.l_status.setText("NORMAL")
        info = utils.get_box(image)
        if info:
            x_min, y_min, x_max, y_max = info
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 255, 2)
        show = cv2.resize(image, IMG_SIZE)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_Indexed8)
        self.g_binary.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def button_open_camera_clicked(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                msg = QtWidgets.QMessageBox.warning(self, '错误', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(100)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec())