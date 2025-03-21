# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\DELL\Desktop\Spring 2025\Image Processing\Image Processing Tasks\Task1\imgTask1UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(976, 744)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.imgFrame = QtWidgets.QFrame(self.frame_2)
        self.imgFrame.setMinimumSize(QtCore.QSize(400, 300))
        self.imgFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imgFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgFrame.setObjectName("imgFrame")
        self.gridLayout_4.addWidget(self.imgFrame, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 0, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.controlBox = QtWidgets.QGroupBox(self.frame_3)
        self.controlBox.setObjectName("controlBox")
        self.gridLayout = QtWidgets.QGridLayout(self.controlBox)
        self.gridLayout.setObjectName("gridLayout")
        self.createImgButton = QtWidgets.QPushButton(self.controlBox)
        self.createImgButton.setObjectName("createImgButton")
        self.gridLayout.addWidget(self.createImgButton, 0, 0, 1, 2)
        self.warpCheckBox = QtWidgets.QCheckBox(self.controlBox)
        self.warpCheckBox.setObjectName("warpCheckBox")
        self.gridLayout.addWidget(self.warpCheckBox, 0, 2, 1, 1)
        self.color1Button = QtWidgets.QPushButton(self.controlBox)
        self.color1Button.setObjectName("color1Button")
        self.gridLayout.addWidget(self.color1Button, 1, 0, 1, 1)
        self.color2Button = QtWidgets.QPushButton(self.controlBox)
        self.color2Button.setObjectName("color2Button")
        self.gridLayout.addWidget(self.color2Button, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.controlBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.lcdNumber = QtWidgets.QLCDNumber(self.controlBox)
        self.lcdNumber.setObjectName("lcdNumber")
        self.gridLayout.addWidget(self.lcdNumber, 2, 1, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.controlBox)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.lcdNumber_2 = QtWidgets.QLCDNumber(self.controlBox)
        self.lcdNumber_2.setObjectName("lcdNumber_2")
        self.gridLayout.addWidget(self.lcdNumber_2, 3, 1, 1, 2)
        self.gridLayout_5.addWidget(self.controlBox, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_3, 0, 1, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.histogramLayout = QtWidgets.QVBoxLayout()
        self.histogramLayout.setObjectName("histogramLayout")
        self.gridLayout_6.addLayout(self.histogramLayout, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_4, 1, 0, 1, 2)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 976, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.controlBox.setTitle(_translate("MainWindow", "Controls"))
        self.createImgButton.setText(_translate("MainWindow", "Create Image"))
        self.warpCheckBox.setText(_translate("MainWindow", "Warp Image"))
        self.color1Button.setText(_translate("MainWindow", "Color #1"))
        self.color2Button.setText(_translate("MainWindow", "Color #2"))
        self.label.setText(_translate("MainWindow", "Mean:"))
        self.label_2.setText(_translate("MainWindow", "STD:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
