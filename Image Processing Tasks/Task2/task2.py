import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
from task2ui import Ui_MainWindow

class ImageProcessor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.rowsSpinBox.setMaximum(300)
        self.ui.columnSpinBox.setMaximum(300)

        self.ui.generateImageButton.clicked.connect(self.generate_random_image)

    def generate_random_image(self):
        """ Generates a random RGB image and displays it """
        rows = self.ui.rowsSpinBox.value()
        cols = self.ui.columnSpinBox.value()

        # Generate a random color image (instead of grayscale)
        matrix = np.random.randint(0, 256, (rows, cols, 3), dtype=np.uint8)

        height, width, _ = matrix.shape
        qimage = QImage(matrix.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)

        # Clear previous image before displaying the new one
        for i in reversed(range(self.ui.randomMatrixLayout.count())):
            self.ui.randomMatrixLayout.itemAt(i).widget().setParent(None)

        self.ui.randomMatrixLayout.addWidget(label)

        # Convert random image to Bayer pattern and process it
        self.generate_bayer_filter(matrix)
        self.generate_bayer_image(rows, cols)

    def generate_bayer_filter(self, rgb_image):
        """ Converts an RGB image to a Bayer pattern visualization """
        rows, cols, _ = rgb_image.shape

        # Convert to a single-channel Bayer matrix for interpolation
        bayer_pattern = np.zeros((rows, cols), dtype=np.uint8)
        bayer_pattern[::2, ::2] = rgb_image[::2, ::2, 2]  # Blue
        bayer_pattern[::2, 1::2] = rgb_image[::2, 1::2, 1]  # Green
        bayer_pattern[1::2, ::2] = rgb_image[1::2, ::2, 1]  # Green
        bayer_pattern[1::2, 1::2] = rgb_image[1::2, 1::2, 0]  # Red

        self.generate_interpolated_image(bayer_pattern)


    
    def generate_bayer_image(self, rows, cols):
        
        bayer_colored = np.zeros((rows, cols, 3), dtype=np.uint8)
        bayer_colored[::2, ::2, 0] = 255  
        bayer_colored[::2, 1::2, 1] = 255  
        bayer_colored[1::2, ::2, 1] = 255  
        bayer_colored[1::2, 1::2, 2] = 255  

        height, width, _ = bayer_colored.shape
        qimage = QImage(bayer_colored.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)

        for i in reversed(range(self.ui.bayerLayout.count())):
            self.ui.bayerLayout.itemAt(i).widget().setParent(None)

        self.ui.bayerLayout.addWidget(label)

    def generate_interpolated_image(self, bayer_pattern):
        """ Applies OpenCV demosaicing separately on each channel and averages the results """

        # Interpolate Blue Channel
        blue_interpolated = cv2.cvtColor(bayer_pattern, cv2.COLOR_BAYER_BG2BGR)[:, :, 0] 

        # Interpolate Green Channel
        green_interpolated = cv2.cvtColor(bayer_pattern, cv2.COLOR_BAYER_BG2BGR)[:, :, 1]

        # Interpolate Red Channel
        red_interpolated = cv2.cvtColor(bayer_pattern, cv2.COLOR_BAYER_BG2BGR)[:, :, 2]

        # Average the three interpolated channels to create the final image
        final_interpolated = np.stack((blue_interpolated, green_interpolated, red_interpolated), axis=2).astype(np.uint8)

        # Display the final interpolated image
        self.display_image(final_interpolated, self.ui.interpolatedImageLayout)

        # Display the Hue component of the image
        self.display_hue_component(final_interpolated)
        self.display_saturation_component(final_interpolated)
        self.display_value_component(final_interpolated)

    def display_hue_component(self, rgb_image):
        """ Extracts and displays the Hue component of the image in the hueLayout """
        
        # Convert RGB to HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Extract the Hue channel
        hue_channel = hsv_image[:, :, 0]  # OpenCV stores Hue in the first channel

        # Normalize hue values to 0-255 for better visualization
        hue_normalized = cv2.normalize(hue_channel, None, 0, 255, cv2.NORM_MINMAX)

        # Convert single-channel grayscale image to RGB format for Qt display
        hue_colormap = cv2.applyColorMap(hue_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Display the Hue image
        self.display_image(hue_colormap, self.ui.hueLayout)


    def display_saturation_component(self, rgb_image):
        """ Extracts and displays the Saturation component in the saturationLayout """
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Extract the Saturation channel
        saturation_channel = hsv_image[:, :, 1]  # Second channel

        # Normalize to 0-255
        saturation_normalized = cv2.normalize(saturation_channel, None, 0, 255, cv2.NORM_MINMAX)

        # Apply colormap for better visualization
        saturation_colormap = cv2.applyColorMap(saturation_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Display in saturationLayout
        self.display_image(saturation_colormap, self.ui.saturationLayout)

    def display_value_component(self, rgb_image):
        """ Extracts and displays the Value component in the valueLayout """
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Extract the Value channel
        value_channel = hsv_image[:, :, 2]  # Third channel

        # Normalize to 0-255
        value_normalized = cv2.normalize(value_channel, None, 0, 255, cv2.NORM_MINMAX)

        # Apply colormap for better visualization
        value_colormap = cv2.applyColorMap(value_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Display in valueLayout
        self.display_image(value_colormap, self.ui.valueLayout)




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
