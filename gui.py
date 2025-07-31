import sys

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QPainter,
    QImage,
    QPixmap,
    QPen,
    QColor,
    QBrush,
    QKeySequence
)
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QFileDialog,
    QPushButton,
    QGraphicsView,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsScene,
    QShortcut
)
from inference import fnet_inference, sam2_inference
from utils.plot import overlay_mask_edge
from constants import *

output_dir = EVAL_RESULT_ROOT / "New"
output_dir.mkdir(parents=True, exist_ok=True)

def pil2pixmap(pil_image):
    """
    Convert a PIL.Image to QPixmap without external dependencies.
    """
    img_rgb = pil_image.convert("RGB")
    w, h = img_rgb.size
    data = img_rgb.tobytes("raw", "RGB")
    bytes_per_line = 3 * w
    qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
    # Deep-copy to ensure safety on Windows
    qimg = qimg.copy()
    return QPixmap.fromImage(qimg)


class AnnotateScene(QGraphicsScene):
    """
    QGraphicsScene subclass that forwards mouse events to the parent window.
    """

    def __init__(self, x, y, w, h, parent_window):
        super().__init__(x, y, w, h)
        self.win = parent_window

    def mousePressEvent(self, ev):
        self.win.mouse_press(ev)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        self.win.mouse_move(ev)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.win.mouse_release(ev)
        super().mouseReleaseEvent(ev)


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # Drawing configurations
        self.half_point_size = 5
        self.point_size = self.half_point_size * 2
        self.is_mouse_down = False

        # graphics items for undo
        self.history = []
        # list of (x_min,y_min,x_max,y_max)
        self.boxes = []

        self.filename = None
        self.pil_image = None
        self.original_pixmap = None

        self.start_point = None
        self.end_point = None
        self.rect = None

        # Max view dimensions to prevent over-scaling
        self.max_view_width = 1920
        self.max_view_height = 1080

        # Create buttons
        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Image")
        clear_button = QPushButton("Clear")
        fnet_button = QPushButton("FNet")
        sam_button = QPushButton("SAM")

        # Connect signals
        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_image)
        clear_button.clicked.connect(self.clear_image)
        fnet_button.clicked.connect(self.run_fnet)
        sam_button.clicked.connect(self.run_sam2)

        # Undo shortcut
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.activated.connect(self.undo)

        # Graphics view setup
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        # Layout setup
        top_hbox = QHBoxLayout()
        top_hbox.addWidget(clear_button)
        top_hbox.addStretch()

        bottom_hbox = QHBoxLayout()
        bottom_hbox.addWidget(load_button)
        bottom_hbox.addWidget(save_button)
        bottom_hbox.addWidget(fnet_button)
        bottom_hbox.addWidget(sam_button)
        bottom_hbox.addStretch()

        # Main layout
        vbox = QVBoxLayout(self)
        vbox.addLayout(top_hbox)
        vbox.addWidget(self.view)
        vbox.addLayout(bottom_hbox)
        self.setLayout(vbox)

        # Initial window size
        self.resize(800, 600)

    def clear_image(self):
        """
        Clear the current image and all annotations from the scene.
        """
        if hasattr(self, 'scene'):
            self.scene.clear()
        self.bg_img = self.scene.addPixmap(self.original_pixmap)
        self.bg_img.setPos(0, 0)

        self.history.clear()
        self.boxes.clear()
        self.start_point = None
        self.end_point = None
        self.rect = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Image to Segment",
            ".",
            "Image Files (*.png *.jpg *.bmp)"
        )
        if not file_path:
            return
        self.filename = Path(file_path).stem

        # Load PIL and clear state
        self.pil_image = Image.open(file_path).convert("RGB")
        self.history.clear()
        self.boxes.clear()
        self.start_point = None
        self.end_point = None
        self.rect = None

        pixmap = QPixmap(file_path)
        self.original_pixmap = pixmap
        W, H = pixmap.width(), pixmap.height()
        view_w = min(W, self.max_view_width)
        view_h = min(H, self.max_view_height)
        self.view.setFixedSize(view_w, view_h)

        # Create annotated scene
        self.scene = AnnotateScene(0, 0, W, H, parent_window=self)
        self.bg_img = self.scene.addPixmap(self.original_pixmap)
        self.bg_img.setPos(0, 0)
        self.view.setScene(self.scene)
        if W > view_w or H > view_h:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.adjustSize()

    def run_fnet(self):
        if self.pil_image is None:
            print("No image loaded for FNet inference.")
            return

        pred_mask = fnet_inference(self.pil_image)
        overlaid = overlay_mask_edge(self.pil_image, pred_mask, )
        pixmap = pil2pixmap(overlaid)
        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)

    def run_sam2(self):
        if self.pil_image is None:
            print("No image loaded for SAM2 inference.")
            return
        if not self.boxes:
            print("Draw a box first.")
            return

        pred_mask = sam2_inference(self.pil_image, boxes=self.boxes)
        overlaid = overlay_mask_edge(self.pil_image, pred_mask)
        pixmap = pil2pixmap(overlaid)
        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)

    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = (x, y)
        pen = QPen(QColor(0, 255, 0, 255))
        pen.setWidth(4)
        brush = QBrush(QColor(0, 255, 0, 100))
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=pen,
            brush=brush,
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return
        x, y = ev.scenePos().x(), ev.scenePos().y()
        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        pen = QPen(QColor(0, 255, 0, 200))
        pen.setWidth(4)
        brush = QBrush(QColor(0, 255, 0, 100))
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=pen,
            brush=brush,
        )
        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin, xmax = min(x, sx), max(x, sx)
        ymin, ymax = min(y, sy), max(y, sy)
        rect_pen = QPen(QColor(0, 255, 0, 180))
        rect_pen.setWidth(6)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=rect_pen
        )

    def mouse_release(self, ev):
        self.is_mouse_down = False

        # compute integer box coords
        x1, y1 = map(int, self.start_pos)
        x2, y2 = map(int, (ev.scenePos().x(), ev.scenePos().y()))
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        box = (x_min, y_min, x_max, y_max)

        # Record drawn items for undo
        items = [self.start_point, self.end_point, self.rect]
        self.history.append(items)
        self.boxes.append(box)

    def undo(self):
        if not self.history:
            return
        items = self.history.pop()
        for item in items:
            self.scene.removeItem(item)

    def save_image(self):
        if not hasattr(self, 'view'):
            return
        # file_path, _ = QFileDialog.getSaveFileName(
        #     self,
        #     "Save Current View",
        #     "",
        #     "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        # )
        file_path = output_dir / f"{self.filename}_sam2.png"
        if not file_path:
            return
        print(file_path)
        # Grab the contents of the graphics view
        pixmap = self.view.grab()
        pixmap.save(str(file_path))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())
