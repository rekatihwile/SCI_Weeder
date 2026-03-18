from collections import OrderedDict, deque

import cv2
import numpy as np

from config import UI_MODE


class WeederUI:
    def __init__(self, mode="window"):
        self.mode = mode
        self.state = "INIT"
        self.subtitle = ""
        self.status = OrderedDict()
        self.events = deque(maxlen=8)
        self.panels = OrderedDict()
        self._windows_created = set()

    @property
    def is_headless(self):
        return self.mode != "window"

    def set_state(self, state, subtitle=None):
        self.state = str(state)
        if subtitle is not None:
            self.subtitle = str(subtitle)

    def set_status(self, label, value):
        self.status[str(label)] = str(value)

    def update_status(self, mapping):
        for key, value in mapping.items():
            self.set_status(key, value)

    def clear_status(self):
        self.status.clear()

    def log_event(self, message):
        if message:
            self.events.appendleft(str(message))

    def update_panel(self, name, canvas):
        if canvas is None:
            return
        self.panels[str(name)] = canvas

    def clear_panel(self, name):
        self.panels.pop(str(name), None)
        if self.mode == "window" and name in self._windows_created:
            try:
                cv2.destroyWindow(str(name))
            except Exception:
                pass
            self._windows_created.discard(str(name))

    def _build_status_canvas(self, width=560, row_h=28, top_pad=70, bottom_pad=30):
        rows = max(8, len(self.status) + len(self.events) + 2)
        height = top_pad + rows * row_h + bottom_pad
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv2.putText(canvas, "Laser Weeder", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"State: {self.state}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y = top_pad
        if self.subtitle:
            cv2.putText(canvas, self.subtitle, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y += row_h

        for key, value in self.status.items():
            cv2.putText(canvas, f"{key}: {value}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)
            y += row_h

        if self.events:
            y += 6
            cv2.putText(canvas, "Recent events", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            y += row_h
            for event in self.events:
                cv2.putText(canvas, f"- {event}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                y += row_h

        return canvas

    def _window_xy(self, name):
        positions = {
            "Weeder Status": (30, 30),
            "Triangulation Debug": (30, 440),
            "Workspace Overview": (980, 30),
            "Fine Align": (620, 440),
        }
        return positions.get(name, (60, 60))

    def _show_window(self, name, canvas):
        if name not in self._windows_created:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            x, y = self._window_xy(name)
            cv2.moveWindow(name, x, y)
            self._windows_created.add(name)

        cv2.imshow(name, canvas)

    def render(self):
        self.panels["Weeder Status"] = self._build_status_canvas()

        if self.is_headless:
            return -1

        for name, canvas in self.panels.items():
            self._show_window(name, canvas)

        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self.mode == "window":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


_UI_INSTANCE = None


def get_ui():
    global _UI_INSTANCE
    if _UI_INSTANCE is None:
        _UI_INSTANCE = WeederUI(mode=UI_MODE)
    return _UI_INSTANCE
