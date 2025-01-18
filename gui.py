# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2025/01/18
import shutil
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QMessageBox, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from cartoonizer_onnx import CartoonizerONNX

model = CartoonizerONNX(onnx_model_path="frozen_cartoonizer.onnx")


def handle(image_path):
    save_path = "___tmp_out." + image_path.split(".")[-1]
    shutil.copy(image_path, save_path)  # 复制一份原始图像，防止路径中有中文导致 cv2 读取失败
    image = cv2.imread(save_path)
    onnx_output = model.inference(image)
    cv2.imwrite(save_path, onnx_output)
    return save_path


class ImageStyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.original_image_path = None
        self.processed_image_path = None

    def initUI(self):
        self.setWindowTitle('Image to Cartoon Style')
        self.setGeometry(100, 100, 800, 600)  # 设置窗口大小

        # 使用样式表美化界面
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border-radius: 5px;
                padding: 5px 15px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #005ea6;
            }
            QLabel {
                border: 1px solid #ccc;
                padding: 5px;
            }
        """)

        # Create widgets
        self.load_button = QPushButton('Load Image', self)
        self.transfer_button = QPushButton('Apply Style Transfer', self)
        self.save_button = QPushButton('Save Result', self)
        self.progress_bar = QProgressBar(self)

        self.original_label = QLabel("Original Image", self)
        self.processed_label = QLabel("Processed Image", self)

        # Set up layout
        vlayout = QVBoxLayout()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.transfer_button)
        button_layout.addWidget(self.save_button)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.processed_label)

        vlayout.addLayout(button_layout)
        vlayout.addWidget(self.progress_bar)
        vlayout.addLayout(image_layout)

        self.setLayout(vlayout)

        # Connect buttons to functions
        self.load_button.clicked.connect(self.load_image)
        self.transfer_button.clicked.connect(self.apply_style_transfer)
        self.save_button.clicked.connect(self.save_result)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        if file_name:
            self.original_image_path = file_name
            pixmap = QPixmap(self.original_image_path)
            self.original_label.setPixmap(
                pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def apply_style_transfer(self):
        if not self.original_image_path:
            QMessageBox.warning(self, 'Warning', 'Please load an image first.')
            return

        self.progress_bar.setValue(0)
        self.processed_image_path = handle(self.original_image_path)
        if self.processed_image_path:
            self.progress_bar.setValue(100)
            pixmap = QPixmap(self.processed_image_path)
            self.processed_label.setPixmap(
                pixmap.scaled(self.processed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_result(self):
        if not self.processed_image_path:
            QMessageBox.warning(self, 'Warning', 'No processed image available to save.')
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Processed Image As", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        if file_name:
            try:
                pixmap = QPixmap(self.processed_image_path)
                pixmap.save(file_name)
                QMessageBox.information(self, 'Success', 'Image saved successfully.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image: {str(e)}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageStyleTransferApp()
    ex.show()
    sys.exit(app.exec_())
