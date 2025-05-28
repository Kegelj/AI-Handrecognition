import sys
import os
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
import keyboard

class SignalEmitter(QObject):
    trigger_flash = pyqtSignal(str)  # 'left', 'up', 'right'

class ArrowOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(600, 100, 400, 150)

        base_path = os.path.join(os.path.dirname(__file__), "assets")
        self.icons = {
            "left": self.create_icon(os.path.join(base_path, "left.png"), 10, 10),
            "up": self.create_icon(os.path.join(base_path, "up.png"), 140, 0),
            "right": self.create_icon(os.path.join(base_path, "right.png"), 270, 10)
        }

        self.signals = SignalEmitter()
        self.signals.trigger_flash.connect(self.flash_icon)

        self.listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.listener_thread.start()

    def create_icon(self, path, x, y):
        label = QLabel(self)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            print(f"[FEHLER] Konnte Bild nicht laden: {path}")
        pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.move(x, y)
        label.setVisible(False)
        return label

    def flash_icon(self, direction):
        label = self.icons.get(direction)
        if label:
            label.setVisible(True)
            QTimer.singleShot(300, lambda: label.setVisible(False))

    def keyboard_listener(self):
        keyboard.on_press(self.handle_key_press)

    def handle_key_press(self, event):
        if event.name == 'a':
            self.signals.trigger_flash.emit("left")
        elif event.name == 'w':
            self.signals.trigger_flash.emit("up")
        elif event.name == 'd':
            self.signals.trigger_flash.emit("right")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = ArrowOverlay()
    overlay.show()
    sys.exit(app.exec_())
