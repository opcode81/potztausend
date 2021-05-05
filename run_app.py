import sys

from PyQt5.QtWidgets import QApplication

from src.app import PotztausendApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    potztausendApp = PotztausendApp()
    potztausendApp.show()
    app.exec()
