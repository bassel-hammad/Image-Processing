import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QGridLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from task2ui import Ui_MainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QGridLayout, QSizePolicy
import matplotlib.pyplot as plt

class ImageProcessor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.rowsSpinBox.setMaximum(300)
        self.ui.columnSpinBox.setMaximum(300)

        self.ui.generateImageButton.clicked.connect(self.generate_random_image)
        self.ui.ApplyButton.clicked.connect(self.apply_bayer_filter)
        self.ui.InterpolateButton.clicked.connect(self.generate_interpolated_image)
        self.ui.TransformButton.clicked.connect(self.transform_image)

    def generate_random_image(self):
        """ Generates a random RGB image where each pixel is its own square, perfectly filling the layout """
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()

        # Generate a random RGB matrix
        self.matrix = np.random.randint(0, 256, (rows, cols, 3), dtype=np.uint8)

        # Clear the layout before updating
        while self.ui.randomMatrixLayout.count():
            item = self.ui.randomMatrixLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create a new grid layout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)  # Ensure no spacing between squares
        grid_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Determine the pixel size dynamically
        layout_width = self.ui.randomMatrixLayout.geometry().width()
        layout_height = self.ui.randomMatrixLayout.geometry().height()
        pixel_size = min(layout_width // cols, layout_height // rows)

        for i in range(rows):
            for j in range(cols):
                color = self.matrix[i, j]

                # Create a label as a square pixel
                label = QLabel()
                label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
                label.setMinimumSize(pixel_size, pixel_size)  # Allow resizing
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # Stretch to fit
                
                grid_layout.addWidget(label, i, j)
        for i in reversed(range(self.ui.randomMatrixLayout.count())):
            self.ui.randomMatrixLayout.itemAt(i).widget().setParent(None)
        self.ui.randomMatrixLayout.addLayout(grid_layout)
  # Return QLabel instead of adding to layout here

    
    def apply_bayer_filter(self):
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()
            # Determine pixel size dynamically
        layout_width = self.ui.BayerFilterImageLayout.geometry().width()
        layout_height = self.ui.BayerFilterImageLayout.geometry().height()

        pixel_size = min(layout_width // cols, layout_height // rows)
        self.bayer_filtered_image = np.zeros_like(self.matrix, dtype=np.uint8)
        # Red channel: even rows, even columns
        self.bayer_filtered_image[0::2, 0::2, 0] = self.matrix[0::2, 0::2, 0]
        # Green channel: even rows, odd columns and odd rows, even columns
        self.bayer_filtered_image[0::2, 1::2, 1] = self.matrix[0::2, 1::2, 1]
        self.bayer_filtered_image[1::2, 0::2, 1] = self.matrix[1::2, 0::2, 1]
        # Blue channel: odd rows, odd columns
        self.bayer_filtered_image[1::2, 1::2, 2] = self.matrix[1::2, 1::2, 2]



        # Clear previous layout contents
        while self.ui.BayerFilterImageLayout.count():
            item = self.ui.BayerFilterImageLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create a new grid layout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)  # Remove gaps
        grid_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Display each pixel as a QLabel with background-color
        for i in range(rows):
            for j in range(cols):
                color = self.bayer_filtered_image[i, j]

                # Convert numpy array values to integers
                r, g, b = int(color[0]), int(color[1]), int(color[2])

                label = QLabel()
                label.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
                label.setMinimumSize(pixel_size, pixel_size)
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

                grid_layout.addWidget(label, i, j)

        self.ui.BayerFilterImageLayout.addLayout(grid_layout)

    def bilinear_interpolation(self, x_left, x_right, y_top, y_bottom,  top_left, top_right, bottom_left, bottom_right,x_target, y_target):
            if x_left == x_right or y_top == y_bottom:  # Prevent division by zero
                return (top_left + top_right + bottom_left + bottom_right) / 4

            interp_top = ((x_right - x_target) / (x_right - x_left)) * top_left + ((x_target - x_left) / (x_right - x_left)) * top_right
            interp_bottom = ((x_right - x_target) / (x_right - x_left)) * bottom_left + ((x_target - x_left) / (x_right - x_left)) * bottom_right

            interp_final = ((y_bottom - y_target) / (y_bottom - y_top)) * interp_top + ((y_target - y_top) / (y_bottom - y_top)) * interp_bottom
            return interp_final
    
    def generate_interpolated_image(self):
        """Generates a color interpolated image using bilinear interpolation and displays it."""
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()

        # Convert the Bayer-filtered image to grayscale (max intensity across RGB channels)
        grayscale_image = self.bayer_filtered_image.max(axis=2).astype(np.float32)
        self.interpolated_image = np.zeros((rows, cols, 3), dtype=np.float32)
        padded_grayscale_image = cv2.copyMakeBorder(grayscale_image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

        for row in range(rows):
            for col in range(cols):
                padded_row, padded_col = row + 1, col + 1  # Offset due to padding
                if (col % 2 == 1) and (row % 2 == 1):  # Blue pixel location
                    blue = padded_grayscale_image[padded_row, padded_col]
                    green = self.bilinear_interpolation(padded_col-1, padded_col+1, padded_row-1, padded_row+1,
                                                        padded_grayscale_image[padded_row-1, padded_col],
                                                        padded_grayscale_image[padded_row+1, padded_col],
                                                        padded_grayscale_image[padded_row, padded_col-1],
                                                        padded_grayscale_image[padded_row, padded_col+1],
                                                        col, row)
                    red = self.bilinear_interpolation(padded_col-1, padded_col+1, padded_row-1, padded_row+1,
                                                    padded_grayscale_image[padded_row-1, padded_col-1],
                                                    padded_grayscale_image[padded_row-1, padded_col+1],
                                                    padded_grayscale_image[padded_row+1, padded_col-1],
                                                    padded_grayscale_image[padded_row+1, padded_col+1],
                                                    col, row)
                elif (col % 2 == 0) and (row % 2 == 0):  # Red pixel location
                    red = padded_grayscale_image[padded_row, padded_col]
                    green = self.bilinear_interpolation(padded_col-1, padded_col+1, padded_row-1, padded_row+1,
                                                        padded_grayscale_image[padded_row-1, padded_col],
                                                        padded_grayscale_image[padded_row+1, padded_col],
                                                        padded_grayscale_image[padded_row, padded_col-1],
                                                        padded_grayscale_image[padded_row, padded_col+1],
                                                        padded_col, padded_row)
                    blue = self.bilinear_interpolation(padded_col-1, padded_col+1, padded_row-1, padded_row+1,
                                                    padded_grayscale_image[padded_row-1, padded_col-1],
                                                    padded_grayscale_image[padded_row-1, padded_col+1],
                                                    padded_grayscale_image[padded_row+1, padded_col-1],
                                                    padded_grayscale_image[padded_row+1, padded_col+1],
                                                    col, row)
                else:  # Green pixel location
                    green = padded_grayscale_image[padded_row, padded_col]
                    if row % 2 == 0:  # Green in red row
                        red = (padded_grayscale_image[padded_row, padded_col-1] + padded_grayscale_image[padded_row, padded_col+1]) / 2
                        blue = (padded_grayscale_image[padded_row-1, padded_col] + padded_grayscale_image[padded_row+1, padded_col]) / 2
                    else:  # Green in blue row
                        red = (padded_grayscale_image[padded_row-1, padded_col] + padded_grayscale_image[padded_row+1, padded_col]) / 2
                        blue = (padded_grayscale_image[padded_row, padded_col-1] + padded_grayscale_image[padded_row, padded_col+1]) / 2
                self.interpolated_image[row, col] = [red, green, blue]

        # Clear previous layout contents
        while self.ui.interpolatedImageLayout_3.count():
            item = self.ui.interpolatedImageLayout_3.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create a new grid layout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)  # Remove gaps
        grid_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        layout_width = self.ui.interpolatedImageLayout_3.geometry().width()
        layout_height = self.ui.interpolatedImageLayout_3.geometry().height()

        # Calculate dynamic pixel size to fit the entire layout
        pixel_size = min(layout_width // cols, layout_height // rows)

        # Display each pixel as a QLabel with background-color
        for row in range(rows):
            for col in range(cols):
                color = self.interpolated_image[row, col]

                # Convert numpy array values to integers
                red, green, blue = int(color[0]), int(color[1]), int(color[2])

                label = QLabel()
                label.setStyleSheet(f"background-color: rgb({red}, {green}, {blue});")
                label.setMinimumSize(pixel_size, pixel_size)
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

                grid_layout.addWidget(label, row, col)

        self.ui.interpolatedImageLayout_3.addLayout(grid_layout)

    def transform_image(self):
        """Converts the image to HSV and displays each channel."""
        hsv_image = cv2.cvtColor(self.matrix, cv2.COLOR_RGB2HSV)

        # Extract channels
        hue_channel = hsv_image[:, :, 0]  # Hue: [0, 179]
        saturation_channel = hsv_image[:, :, 1]  # Saturation: [0, 255]
        value_channel = hsv_image[:, :, 2]  # Value: [0, 255]
        
        # Use grid display for Hue
        self.display_channel_as_grid(hue_channel, is_hue=True,layout=self.ui.hueLayout)
        self.display_channel_as_grid(saturation_channel,layout=self.ui.saturationLayout)
        self.display_channel_as_grid(value_channel,layout=self.ui.valueLayout)
        self.display_channel_as_grid(hsv_image,layout=self.ui.valueLayout_4)
        

    def display_channel_as_grid(self, channel, layout, is_hue=False):
        """Displays the given channel as a grid of QLabel widgets."""
        
        # Clear previous layout contents
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Check if channel is 3D (RGB Image)
        if len(channel.shape) == 3 and channel.shape[2] == 3:
            rows, cols, _ = channel.shape
        else:
            rows, cols = channel.shape

        # Create a new grid layout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)  # Remove gaps
        grid_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        for i in range(rows):
            for j in range(cols):
                if is_hue:
                    # Convert Hue to RGB color representation
                    hsv_colored = np.uint8([[[channel[i, j], 255, 255]]])  # Full saturation & value
                    rgb_colored = cv2.cvtColor(hsv_colored, cv2.COLOR_HSV2RGB)[0][0]
                    r, g, b = int(rgb_colored[0]), int(rgb_colored[1]), int(rgb_colored[2])
                elif len(channel.shape) == 3:  # RGB case
                    r, g, b = int(channel[i, j, 0]), int(channel[i, j, 1]), int(channel[i, j, 2])
                else:
                    # Grayscale case (Saturation & Value)
                    r = g = b = int(channel[i, j])

                # QLabel for displaying color
                label = QLabel()
                label.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
                label.setMinimumSize(10, 10)  # Adjust pixel size
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

                grid_layout.addWidget(label, i, j)

        layout.addLayout(grid_layout)  # Add to layout


    def display_image(self, image, layout):
        """ Helper function to display images in a given layout """
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)

        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

        layout.addWidget(label)

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
