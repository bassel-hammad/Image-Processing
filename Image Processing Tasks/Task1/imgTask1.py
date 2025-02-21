import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QColorDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout
from imgTask1UI import Ui_MainWindow  

class ImageProcessor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  
        self.ui.setupUi(self)  

        # Add QLabel inside imgFrame for displaying the image
        self.imgLabel = QtWidgets.QLabel(self.ui.imgFrame)
        self.imgLabel.setGeometry(0, 0, self.ui.imgFrame.width(), self.ui.imgFrame.height())
        self.imgLabel.setScaledContents(True)

        # Default colors 
        self.color1 = (0, 0, 0)  # White
        self.color2 = (0, 0, 0)  # White

        # Connect button click to function
        self.ui.createImgButton.clicked.connect(self.generate_chessboard)
        self.ui.color1Button.clicked.connect(self.choose_color1)
        self.ui.color2Button.clicked.connect(self.choose_color2)

    def choose_color1(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color1 = (color.red(), color.green(), color.blue())  # Store RGB color

    def choose_color2(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color2 = (color.red(), color.green(), color.blue())  # Store RGB color
        

    def generate_chessboard(self):
        rows, cols = 10, 10 # Chessboard size
        square_size = 50   # Size of each square in pixels
        height, width = rows * square_size, cols * square_size   

        chessboard = np.full((height, width, 3), 255, dtype=np.uint8)  # White background image

        # Convert colors to numpy arrays for interpolation
        color1_np = np.array(self.color1, dtype=np.float32)
        color2_np = np.array(self.color2, dtype=np.float32)

        # Generate the gradient chessboard
        for y in range(height):
            # Compute the interpolation factor (0 at top, 1 at bottom)
            t = y / (height - 1)  
            current_color = (1 - t) * color1_np + t * color2_np  # Linear interpolation

            for x in range(width):
                i, j = y // square_size, x // square_size  # Determine chessboard position
                if (i + j) % 2 == 0:
                    chessboard[x, y] = current_color  # Apply interpolated color

       # If warpCheckBox is checked, apply the wavy warp effect
        if self.ui.warpCheckBox.isChecked():
            # Create mapping arrays
            map_x = np.zeros((height, width), dtype=np.float32)
            map_y = np.zeros((height, width), dtype=np.float32)

            # Amplitude and frequency of the waves
            amplitude = 5 # Adjust for stronger waves
            frequency = 2 * np.pi / 80  # Controls how many waves appear

            for y in range(height):
                for x in range(width):
                    shift_x = amplitude * np.sin(frequency * y)
                    shift_y = amplitude * np.sin(frequency * x)
                    map_x[y, x] = x + shift_x
                    map_y[y, x] = y + shift_y

            # Apply remap to distort the chessboard
            chessboard = cv2.remap(chessboard, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Convert to QImage
        chessboard = chessboard.astype(np.uint8)
        q_image = QImage(chessboard.data, width, height, 3 * width, QImage.Format_RGB888)

        # Display in QLabel
        self.imgLabel.setPixmap(QPixmap.fromImage(q_image))

        # Compute mean and standard deviation
        mean = chessboard.mean()
        std = chessboard.std()
        self.ui.lcdNumber.display(mean)
        self.ui.lcdNumber_2.display(std)

        self.display_histogram(chessboard)


    def display_histogram(self, image):
        
        # If the canvas already exists, clear the previous plot
        if hasattr(self, 'canvas'):
            self.canvas.figure.clear()  # Clear the figure
            ax = self.canvas.figure.add_subplot(111)  # Create a new subplot
        else:
            # Create a new Matplotlib figure and canvas if it doesn't exist
            fig = Figure(figsize=(6, 3), dpi=100)
            ax = fig.add_subplot(111)
            self.canvas = FigureCanvas(fig)
            self.ui.histogramLayout.addWidget(self.canvas)

        # Set up the histogram plot
        ax.set_xlim([0, 256])
        ax.set_title("Color Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")

        # Compute and plot histograms
        if len(image.shape) == 3:  # RGB image
            colors = ('b', 'g', 'r')  # OpenCV uses BGR format
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)
        else:  # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black')
        # Redraw the canvas with the updated plot
        self.canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
