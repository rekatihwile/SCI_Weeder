import cv2
import numpy as np


class ManualDetectorLocal:
    def __init__(self, display_scale=1.5, zoom_crop_size=40, zoom_display_size=500):
        self.display_scale = display_scale
        self.zoom_crop_size = zoom_crop_size
        self.zoom_display_size = zoom_display_size

        self.left_points = []
        self.right_points = []

        self.left_frame = None
        self.right_frame = None

        self.h = None
        self.w = None
        self.display_h = None
        self.display_w = None

        self.window_name = "Manual Detector - Stereo Pair"
        self.zoom_window_name = "Manual Detector - Zoom"

        self.zoom_active = False
        self.zoom_panel = None          # "left" or "right"
        self.zoom_frame = None          # original frame for current zoom
        self.zoom_x1 = 0
        self.zoom_y1 = 0
        self.zoom_x2 = 0
        self.zoom_y2 = 0

    def _make_display_canvas(self):
        left_disp = cv2.resize(
            self.left_frame,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_LINEAR,
        )
        right_disp = cv2.resize(
            self.right_frame,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.hstack((left_disp, right_disp))

        cv2.line(
            canvas,
            (self.display_w, 0),
            (self.display_w, self.display_h),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            canvas,
            "LEFT",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "RIGHT",
            (self.display_w + 20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        return canvas

    def _draw_points_on_canvas(self, canvas):
        sx = self.display_w / self.w
        sy = self.display_h / self.h

        for i, (x, y) in enumerate(self.left_points):
            xd = int(x * sx)
            yd = int(y * sy)
            cv2.circle(canvas, (xd, yd), 7, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (xd + 8, yd - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        for i, (x, y) in enumerate(self.right_points):
            xd = int(x * sx) + self.display_w
            yd = int(y * sy)
            cv2.circle(canvas, (xd, yd), 7, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (xd + 8, yd - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return canvas

    def _display_to_original(self, x_disp, y_disp, panel):
        sx = self.w / self.display_w
        sy = self.h / self.display_h

        if panel == "left":
            x_orig = int(x_disp * sx)
            y_orig = int(y_disp * sy)
        else:
            x_local = x_disp - self.display_w
            x_orig = int(x_local * sx)
            y_orig = int(y_disp * sy)

        x_orig = max(0, min(self.w - 1, x_orig))
        y_orig = max(0, min(self.h - 1, y_orig))

        return x_orig, y_orig

    def _begin_zoom(self, panel, x_orig, y_orig):
        self.zoom_active = True
        self.zoom_panel = panel

        self.zoom_frame = self.left_frame.copy() if panel == "left" else self.right_frame.copy()

        self.zoom_x1 = max(0, x_orig - self.zoom_crop_size)
        self.zoom_y1 = max(0, y_orig - self.zoom_crop_size)
        self.zoom_x2 = min(self.w, x_orig + self.zoom_crop_size)
        self.zoom_y2 = min(self.h, y_orig + self.zoom_crop_size)

        cv2.namedWindow(self.zoom_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.zoom_window_name, self._zoom_mouse_callback)

    def _zoom_mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        crop_w = self.zoom_x2 - self.zoom_x1
        crop_h = self.zoom_y2 - self.zoom_y1

        if crop_w <= 0 or crop_h <= 0:
            return

        scale_x = crop_w / self.zoom_display_size
        scale_y = crop_h / self.zoom_display_size

        x_orig = int(self.zoom_x1 + x * scale_x)
        y_orig = int(self.zoom_y1 + y * scale_y)

        x_orig = max(0, min(self.w - 1, x_orig))
        y_orig = max(0, min(self.h - 1, y_orig))

        if self.zoom_panel == "left":
            self.left_points.append((x_orig, y_orig))
        else:
            self.right_points.append((x_orig, y_orig))

        self.zoom_active = False
        self.zoom_panel = None
        self.zoom_frame = None

        try:
            cv2.destroyWindow(self.zoom_window_name)
        except cv2.error:
            pass

    def _main_mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.zoom_active:
            return

        if x < self.display_w:
            panel = "left"
            x_orig, y_orig = self._display_to_original(x, y, "left")
        else:
            panel = "right"
            x_orig, y_orig = self._display_to_original(x, y, "right")

        self._begin_zoom(panel, x_orig, y_orig)

    def _show_zoom_window(self):
        if not self.zoom_active or self.zoom_frame is None:
            return

        crop = self.zoom_frame[self.zoom_y1:self.zoom_y2, self.zoom_x1:self.zoom_x2].copy()
        if crop.size == 0:
            return

        zoom_display = cv2.resize(
            crop,
            (self.zoom_display_size, self.zoom_display_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

        ch, cw = zoom_display.shape[:2]
        cv2.line(zoom_display, (cw // 2, 0), (cw // 2, ch), (255, 100, 0), 1)
        cv2.line(zoom_display, (0, ch // 2), (cw, ch // 2), (255, 100, 0), 1)

        label = f"ZOOM - {self.zoom_panel.upper()} (click precise point)"
        cv2.putText(
            zoom_display,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(self.zoom_window_name, zoom_display)

# Public method to get a single refined target from user input

    def refine_live(self, cameras):
        left_points, right_points = self.detect_live(cameras)

        if not left_points or not right_points:
            return None, None

        return left_points[0], right_points[0]
    
# Public method to get all user-selected targets as lists of points

    def detect_live(self, cameras):
        print("\n=== DETECT ===")
        print("Live stereo view.")
        print("Click a panel to open magnified selection.")
        print("Enter = accept | r = reset | q = quit")

        self.left_points = []
        self.right_points = []

        self.left_frame, self.right_frame = cameras.read_pair()
        self.h, self.w = self.left_frame.shape[:2]
        self.display_w = int(self.w * self.display_scale)
        self.display_h = int(self.h * self.display_scale)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._main_mouse_callback)

        while True:
            self.left_frame, self.right_frame = cameras.read_pair()

            canvas = self._make_display_canvas()
            canvas = self._draw_points_on_canvas(canvas)

            status_1 = f"L points: {len(self.left_points)}    R points: {len(self.right_points)}"
            status_2 = "Click panel -> zoom | Enter = accept | r = reset | q = quit"

            cv2.rectangle(
                canvas,
                (0, self.display_h - 70),
                (canvas.shape[1], self.display_h),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                canvas,
                status_1,
                (20, self.display_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                status_2,
                (20, self.display_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(self.window_name, canvas)
            self._show_zoom_window()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error:
                    pass
                try:
                    cv2.destroyWindow(self.zoom_window_name)
                except cv2.error:
                    pass
                return [], []

            if key == ord("r"):
                self.left_points = []
                self.right_points = []
                self.zoom_active = False
                self.zoom_panel = None
                self.zoom_frame = None
                try:
                    cv2.destroyWindow(self.zoom_window_name)
                except cv2.error:
                    pass

            if key == 13:
                if len(self.left_points) > 0 or len(self.right_points) > 0:
                    break

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        try:
            cv2.destroyWindow(self.zoom_window_name)
        except cv2.error:
            pass

        return self.left_points, self.right_points