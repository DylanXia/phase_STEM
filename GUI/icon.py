# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# importing the required libraries

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtGui
import os
import sys

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon = QtGui.QIcon(os.path.join(script_dir, 'logo.png'))
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(128,128))  # Set the icon size to 48x48 pixels
        self.setWindowTitle("Icon")
        self.setGeometry(0, 0, 400, 300)

        self.label = QLabel("Icon is set", self)
        self.label.move(200, 200)
        self.label.setStyleSheet("border: 1px solid black;")

        self.show()
# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())
