import numpy as np


def get_y_estimate(y_feature, min_support=200,
                   min_conf=50):
    indices = np.nonzero(y_feature)[0]
    ys = []
    for y in indices:
        if min_support <= 0:
            break
        num = y_feature[y]
        if num > min_support:
            ys.extend([y] * min_support)
            break
        else:
            ys.extend([y] * num)
            min_support -= num
    if len(ys) < min_conf:
        return 0
    ys = ys[min_conf:]
    return np.mean(ys)


class Analyzer:
    def __init__(self, n_frame=10, change_ratio=0.7):
        self.n_frame = n_frame
        self.change_ratio = change_ratio
        self.ys = []
        self.is_fall = False

    def add(self, image):
        y_feature = np.where(image != 255, 0, 1).sum(axis=1)
        y_est = get_y_estimate(y_feature)
        self.ys.append(y_est)
        print(y_est)

        if len(self.ys) < self.n_frame * 2:
            return 0
        cut = self.ys[-10:]
        if cut == sorted(cut) and \
            (cut[-1] - cut[0]) / self.n_frame > self.change_ratio:
            self.is_fall = True
            return 1
        return 1 if self.is_fall else 0

    def clean(self):
        self.ys = []


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QTimer
from numpy import *
from scipy import interpolate


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.init_plot()

    def plot(self):
        timer = QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(100)

    def init_plot(self):
        n = 200
        x = [0]
        y = [0]
        self.axes.plot(x, y)
        self.axes.set_xlabel("time")
        self.axes.set_ylabel("height")
        self.axes.set_ylim(0, 270)

    def update_figure(self, ys):
        self.axes.cla()
        self.axes.scatter([i for i in range(len(ys))], ys)
        self.axes.set_ylim(0, 270)
        self.axes.set_ylabel("height")
        self.draw()
