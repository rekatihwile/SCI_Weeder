import cv2
import numpy as np

class ManualDetectorLocal:
    def __init__(self, display_scale=1.2, zoom_crop_size=40, zoom_display_size=600):
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

        self.window_name = "Manual Detector - Integrated UI"
        
        self.zoom_active = False
        self.zoom_panel = None
        self.zoom_frame = None
        self.zoom_x1 = 0
        self.zoom_y1 = 0
        self.zoom_x2 = 0
        self.zoom_y2 = 0

        self.accepting_clicks = True

    def _make_combined_canvas(self):
        # 1. Resize the stereo pair
        left_disp = cv2.resize(self.left_frame, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)
        right_disp = cv2.resize(self.right_frame, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Create the stereo strip
        stereo_strip = np.hstack((left_disp, right_disp))
        
        # 3. Create the Zoom Panel
        zoom_panel = np.zeros((self.display_h, self.zoom_display_size, 3), dtype=np.uint8)
        
        if self.zoom_active and self.zoom_frame is not None:
            crop = self.zoom_frame[self.zoom_y1:self.zoom_y2, self.zoom_x1:self.zoom_x2]
            if crop.size > 0:
                zd = cv2.resize(crop, (self.zoom_display_size, self.zoom_display_size), interpolation=cv2.INTER_LANCZOS4)
                # Vertically center the zoom square in the side panel
                y_offset = max(0, (self.display_h - self.zoom_display_size) // 2)
                
                # Ensure we don't go out of bounds if zoom_display_size > display_h
                end_y = min(y_offset + self.zoom_display_size, self.display_h)
                zoom_panel[y_offset:end_y, :self.zoom_display_size] = zd[:(end_y-y_offset), :]
                
                # Crosshair
                cx, cy = self.zoom_display_size // 2, y_offset + (self.zoom_display_size // 2)
                cv2.line(zoom_panel, (cx, y_offset), (cx, end_y), (255, 100, 0), 1)
                cv2.line(zoom_panel, (0, cy), (self.zoom_display_size, cy), (255, 100, 0), 1)
        
        # 4. Final Assemble
        full_canvas = np.hstack((stereo_strip, zoom_panel))
        
        # Visual Dividers
        cv2.line(full_canvas, (self.display_w, 0), (self.display_w, self.display_h), (255, 255, 255), 1)
        cv2.line(full_canvas, (self.display_w * 2, 0), (self.display_w * 2, self.display_h), (0, 255, 0), 2)
        
        return full_canvas

    def _draw_points(self, canvas):
        sx, sy = self.display_w / self.w, self.display_h / self.h
        for x, y in self.left_points:
            cv2.circle(canvas, (int(x * sx), int(y * sy)), 5, (0, 0, 255), -1)
        for x, y in self.right_points:
            cv2.circle(canvas, (int(x * sx) + self.display_w, int(y * sy)), 5, (0, 255, 0), -1)
        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self.accepting_clicks:
            return

        if x < self.display_w * 2:
            self.zoom_active = True
            panel = "left" if x < self.display_w else "right"
            self.zoom_panel = panel
            sx, sy = self.w / self.display_w, self.h / self.display_h
            x_local = x if panel == "left" else x - self.display_w
            x_orig, y_orig = int(x_local * sx), int(y * sy)
            
            self.zoom_frame = self.left_frame.copy() if panel == "left" else self.right_frame.copy()
            self.zoom_x1, self.zoom_y1 = max(0, x_orig - self.zoom_crop_size), max(0, y_orig - self.zoom_crop_size)
            self.zoom_x2, self.zoom_y2 = min(self.w, x_orig + self.zoom_crop_size), min(self.h, y_orig + self.zoom_crop_size)

        elif self.zoom_active and x >= self.display_w * 2:
            z_x = x - (self.display_w * 2)
            y_offset = (self.display_h - self.zoom_display_size) // 2
            z_y = y - y_offset
            
            if 0 <= z_x < self.zoom_display_size and 0 <= z_y < self.zoom_display_size:
                crop_w, crop_h = self.zoom_x2 - self.zoom_x1, self.zoom_y2 - self.zoom_y1
                final_x = int(self.zoom_x1 + z_x * (crop_w / self.zoom_display_size))
                final_y = int(self.zoom_y1 + z_y * (crop_h / self.zoom_display_size))
                
                if self.zoom_panel == "left": self.left_points.append((final_x, final_y))
                else: self.right_points.append((final_x, final_y))
                self.zoom_active = False

    def detect_live(self, cameras):
        self.left_points, self.right_points = [], []
        self.zoom_active = False
        
        # Initialize Window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # --- LINUX MAXIMIZE TRICK ---
        # Toggle fullscreen on then off forces the window manager to expand the parent window
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            if not self.zoom_active:
                fL, fR = cameras.read_pair()
                if fL is not None:
                    self.left_frame, self.right_frame = fL.copy(), fR.copy()
                    if self.h is None:
                        self.h, self.w = self.left_frame.shape[:2]
                        # Set internal resolution based on screen space
                        self.display_w = int(self.w * self.display_scale)
                        self.display_h = int(self.h * self.display_scale)

            canvas = self._make_combined_canvas()
            canvas = self._draw_points(canvas)
            
            # Status bar
            cv2.rectangle(canvas, (0, self.display_h - 40), (canvas.shape[1], self.display_h), (0, 0, 0), -1)
            cv2.putText(canvas, f"L:{len(self.left_points)} R:{len(self.right_points)} | ENTER=Accept | R=Reset | Q=Quit", 
                        (20, self.display_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return [], []
            if key == ord('r'):
                self.left_points, self.right_points = [], []
                self.zoom_active = False
            if key == 13 and len(self.left_points) > 0:
                break

        cv2.destroyWindow(self.window_name)
        return self.left_points, self.right_points

    def refine_live(self, cameras):
        l, r = self.detect_live(cameras)
        return (l[0], r[0]) if (l and r) else (None, None)