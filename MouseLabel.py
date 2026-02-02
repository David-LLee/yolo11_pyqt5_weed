from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QCursor

class LabelMouse(QLabel):
    double_clicked = pyqtSignal()
    mouse_entered = pyqtSignal()
    mouse_left = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        self.mouse_moved.emit(event.pos())
        super().mouseMoveEvent(event)

    def enterEvent(self, event):
        """鼠标进入控件区域"""
        self.mouse_entered.emit()
        self.setStyleSheet("border: 2px solid blue;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开控件区域"""
        self.mouse_left.emit()
        self.setStyleSheet("")
        super().leaveEvent(event)

class Label_click_Mouse(QLabel):
    clicked = pyqtSignal(Qt.MouseButton)
    right_clicked = pyqtSignal()
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        self.clicked.emit(event.button())
        # 区分左右键
        if event.button() == Qt.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)