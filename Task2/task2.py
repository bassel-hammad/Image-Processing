import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from task2ui import Ui_MainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QGridLayout, QSizePolicy

class ImageProcessor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.rowsSpinBox.setMaximum(300)
        self.ui.columnSpinBox.setMaximum(300)

        self.ui.generateImageButton.clicked.connect(self.generate_random_image)
        self.ui.ApplyButton.clicked.connect(self.create_bayer_filter)
        self.ui.InterpolateButton.clicked.connect(self.generate_interpolated_image)

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



    from PyQt5.QtWidgets import QLabel, QGridLayout, QSizePolicy

    def create_bayer_filter(self):
        """ Creates and displays a Bayer filter pattern (RGGB) in the bayerLayout """
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()

        # Clear previous layout contents
        while self.ui.bayerLayout.count():
            item = self.ui.bayerLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create a grid layout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)  # Remove spacing between cells
        grid_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Determine pixel size dynamically
        layout_width = self.ui.bayerLayout.geometry().width()
        layout_height = self.ui.bayerLayout.geometry().height()
        pixel_size = min(layout_width // cols, layout_height // rows)

        # Bayer Filter Pattern (RGGB)
        for i in range(rows):
            for j in range(cols):
                if (i % 2 == 0) and (j % 2 == 0):  # Red pixel (R)
                    color = (255, 0, 0)  # Red
                elif (i % 2 == 0) and (j % 2 == 1):  # Green pixel (G)
                    color = (0, 255, 0)  # Green
                elif (i % 2 == 1) and (j % 2 == 0):  # Green pixel (G)
                    color = (0, 255, 0)  # Green
                else:  # Blue pixel (B)
                    color = (0, 0, 255)  # Blue

                # Create a QLabel for each pixel
                label = QLabel()
                label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
                label.setMinimumSize(pixel_size, pixel_size)
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

                grid_layout.addWidget(label, i, j)

        self.ui.bayerLayout.addLayout(grid_layout)
        self.apply_bayer_filter()

 

    
    def apply_bayer_filter(self):
        """ Applies a Bayer filter (RGGB) while ensuring it fully fills the layout """
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()

        # Get available dimensions of BayerFilterImageLayout
        layout_width = self.ui.BayerFilterImageLayout.geometry().width()
        layout_height = self.ui.BayerFilterImageLayout.geometry().height()

        # Calculate dynamic pixel size to fit the entire layout
        pixel_size = min(layout_width // cols, layout_height // rows)

        # Generate a Bayer filter mask
        bayer_filter = np.zeros((rows, cols, 3), dtype=np.uint8)

        # RGGB Pattern
        for i in range(rows):
            for j in range(cols):
                if (i % 2 == 0) and (j % 2 == 0):  # Red pixel (R)
                    bayer_filter[i, j] = [1, 0, 0]
                elif (i % 2 == 0) and (j % 2 == 1):  # Green pixel (G)
                    bayer_filter[i, j] = [0, 1, 0]
                elif (i % 2 == 1) and (j % 2 == 0):  # Green pixel (G)
                    bayer_filter[i, j] = [0, 1, 0]
                else:  # Blue pixel (B)
                    bayer_filter[i, j] = [0, 0, 1]

        # Apply the Bayer filter
        self.bayer_filtered_image = self.matrix * bayer_filter

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
        """ Generates a color interpolated image using bilinear interpolation and displays it. """
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()

        # Convert the Bayer-filtered image to grayscale (max intensity across RGB channels)
        grayscale_image = self.bayer_filtered_image.max(axis=2).astype(np.float32)
        Interpolated_image = np.zeros((rows, cols, 3), dtype=np.float32)
        bayer_padded = cv2.copyMakeBorder(grayscale_image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        for y in range(0, rows):
            for x in range(0, cols):
                y_p, x_p = y + 1, x + 1  # Offset due to padding
                if (x % 2 == 1) and (y % 2 == 1):  # Blue pixel location
                    B = bayer_padded[y_p, x_p]
                    G = self.bilinear_interpolation(x_p-1, x_p+1, y_p-1, y_p+1,
                                            bayer_padded[y_p-1, x_p],
                                            bayer_padded[y_p+1, x_p],
                                            bayer_padded[y_p, x_p-1],
                                            bayer_padded[y_p, x_p+1],
                                            x, y)
                    R = self.bilinear_interpolation(x_p-1, x_p+1, y_p-1, y_p+1,
                                            bayer_padded[y_p-1, x_p-1],
                                            bayer_padded[y_p-1, x_p+1],
                                            bayer_padded[y_p+1, x_p-1],
                                            bayer_padded[y_p+1, x_p+1],
                                            x, y)
                elif (x % 2 == 0) and (y % 2 == 0):  # Red pixel location
                    R = bayer_padded[y_p, x_p]
                    G = self.bilinear_interpolation(x_p-1, x_p+1, y_p-1, y_p+1,
                                            bayer_padded[y_p-1, x_p],
                                            bayer_padded[y_p+1, x_p],
                                            bayer_padded[y_p, x_p-1],
                                            bayer_padded[y_p, x_p+1],
                                            x_p, y_p)
                    B = self.bilinear_interpolation(x_p-1, x_p+1, y_p-1, y_p+1,
                                            bayer_padded[y_p-1, x_p-1],
                                            bayer_padded[y_p-1, x_p+1],
                                            bayer_padded[y_p+1, x_p-1],
                                            bayer_padded[y_p+1, x_p+1],
                                            x, y)
                else:  # Green pixel location
                    G = bayer_padded[y_p, x_p]
                    if y % 2 == 0:  # Green in red row
                        R = (bayer_padded[y_p, x_p-1] + bayer_padded[y_p, x_p+1]) / 2
                        B = (bayer_padded[y_p-1, x_p] + bayer_padded[y_p+1, x_p]) / 2
                    else:  # Green in blue row
                        R = (bayer_padded[y_p-1, x_p] + bayer_padded[y_p+1, x_p]) / 2
                        B = (bayer_padded[y_p, x_p-1] + bayer_padded[y_p, x_p+1]) / 2
                Interpolated_image[y,x] = [R,G,B]
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
        for i in range(rows):
            for j in range(cols):
                color = Interpolated_image[i, j]

                # Convert numpy array values to integers
                r, g, b = int(color[0]), int(color[1]), int(color[2])

                label = QLabel()
                label.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
                label.setMinimumSize(pixel_size, pixel_size)
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

                grid_layout.addWidget(label, i, j)

        self.ui.interpolatedImageLayout_3.addLayout(grid_layout)
        


        

    # def display_hue_component(self, rgb_image):
    #     """ Extracts and displays the Hue component of the image in the hueLayout """
        
    #     # Convert RGB to HSV
    #     hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
    #     # Extract the Hue channel
    #     hue_channel = hsv_image[:, :, 0]  # OpenCV stores Hue in the first channel

    #     # Normalize hue values to 0-255 for better visualization
    #     hue_normalized = cv2.normalize(hue_channel, None, 0, 255, cv2.NORM_MINMAX)

    #     # Convert single-channel grayscale image to RGB format for Qt display
    #     hue_colormap = cv2.applyColorMap(hue_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    #     # Display the Hue image
    #     self.display_image(hue_colormap, self.ui.hueLayout)


    # def display_saturation_component(self, rgb_image):
    #     """ Extracts and displays the Saturation component in the saturationLayout """
        
    #     # Convert to HSV
    #     hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
    #     # Extract the Saturation channel
    #     saturation_channel = hsv_image[:, :, 1]  # Second channel

    #     # Normalize to 0-255
    #     saturation_normalized = cv2.normalize(saturation_channel, None, 0, 255, cv2.NORM_MINMAX)

    #     # Apply colormap for better visualization
    #     saturation_colormap = cv2.applyColorMap(saturation_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    #     # Display in saturationLayout
    #     self.display_image(saturation_colormap, self.ui.saturationLayout)

    # def display_value_component(self, rgb_image):
    #     """ Extracts and displays the Value component in the valueLayout """
        
    #     # Convert to HSV
    #     hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
    #     # Extract the Value channel
    #     value_channel = hsv_image[:, :, 2]  # Third channel

    #     # Normalize to 0-255
    #     value_normalized = cv2.normalize(value_channel, None, 0, 255, cv2.NORM_MINMAX)

    #     # Apply colormap for better visualization
    #     value_colormap = cv2.applyColorMap(value_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    #     # Display in valueLayout
    #     self.display_image(value_colormap, self.ui.valueLayout)




    # def display_image(self, image, layout):
    #     """ Helper function to display images in a given layout """
    #     height, width, channels = image.shape
    #     qimage = QImage(image.data, width, height, channels * width, QImage.Format_RGB888)
    #     pixmap = QPixmap.fromImage(qimage)

    #     label = QLabel()
    #     label.setPixmap(pixmap)
    #     label.setScaledContents(True)

    #     for i in reversed(range(layout.count())):
    #         layout.itemAt(i).widget().setParent(None)

    #     layout.addWidget(label)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
