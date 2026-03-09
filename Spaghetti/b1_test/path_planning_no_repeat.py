import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import itertools

# ===========================
# Configuration
# ===========================
plt.style.use('dark_background')

WS_W, WS_H = 450, 440
CAM_SPACING, CAM_OFFSET_X = 30, 15.0
FOV_W, FOV_H = 120, 100 

# Simulation Physics
TRANSIT_SPEED, DT, KP = 400, 1/30.0, 4.0 
X_STEP, Y_STEP = (CAM_SPACING + FOV_W) * 0.3, FOV_H * 0.3
WEED_DENSITY = 100

class RayMatchingScanner:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 6))
        self.viewpoints = self.generate_viewpoints()
        self.viewpoint_index = 0
        self.pos = np.copy(self.viewpoints[0])
        self.state = "TRANSIT"
        self.settle_timer = 0
        
        # Ground Truth
        self.weeds = np.random.rand(WEED_DENSITY, 2) * [WS_W, WS_H]
        self.zapped_memory = [] # Absolute G53 coordinates [x, y]
        
        self.target_queue = []
        self.setup_ui()
        self.timer = self.fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self.update_physics)
        self.timer.start()

    def generate_viewpoints(self):
        points = []
        curr_y, direction = WS_H - FOV_H/2, 1 
        while curr_y > 20:
            start_x, end_x = 75, WS_W - 75
            x_range = np.arange(start_x, end_x + 1, X_STEP)
            if direction == -1: x_range = x_range[::-1]
            for x in x_range: points.append([x, curr_y])
            curr_y -= Y_STEP
            direction *= -1
        return np.array(points)

    def filter_with_ray_logic(self, potential_indices):
        """
        Uses the 'Angle/Ray' logic to filter out ghosts of zapped weeds.
        Even if Z is unknown, the vector from camera to weed should match.
        """
        filtered = []
        for idx in potential_indices:
            actual_pos = self.weeds[idx]
            is_ghost = False
            
            for zapped_pos in self.zapped_memory:
                # Calculate the vector from CURRENT camera center to the weed
                # and compare it to the vector to the known zapped position.
                vec_to_weed = actual_pos - self.pos
                vec_to_zap = zapped_pos - self.pos
                
                # If the vectors are nearly identical in direction, it's a re-hit.
                # We use a distance check here as a proxy for angular error in 2D.
                if np.linalg.norm(actual_pos - zapped_pos) < 6.0:
                    is_ghost = True
                    break
            
            if not is_ghost:
                filtered.append(idx)
        return filtered

    def plan_optimized_path(self):
        # 1. Capture what the cameras see right now
        rel_l = self.weeds - (self.pos + np.array([-CAM_OFFSET_X, 0]))
        rel_r = self.weeds - (self.pos + np.array([CAM_OFFSET_X, 0]))
        mask = (np.abs(rel_l[:,0]) < FOV_W/2) & (np.abs(rel_l[:,1]) < FOV_H/2) & \
               (np.abs(rel_r[:,0]) < FOV_W/2) & (np.abs(rel_r[:,1]) < FOV_H/2)
        
        indices = np.where(mask)[0].tolist()
        
        # 2. Apply the Ray-Matching Filter
        valid_indices = self.filter_with_ray_logic(indices)
        
        if not valid_indices: return []

        next_view = self.viewpoints[(self.viewpoint_index + 1) % len(self.viewpoints)]
        
        # 3. Global Optimizer for the local chain
        if len(valid_indices) <= 8:
            best_p, min_d = [], float('inf')
            for p in itertools.permutations(valid_indices):
                d = np.linalg.norm(self.pos - self.weeds[p[0]])
                for i in range(len(p)-1):
                    d += np.linalg.norm(self.weeds[p[i]] - self.weeds[p[i+1]])
                d += np.linalg.norm(self.weeds[p[-1]] - next_view)
                if d < min_d: min_d, best_p = d, p
            return [(self.weeds[i], i) for i in best_p]
        else:
            return [(self.weeds[i], i) for i in valid_indices]

    def update_physics(self):
        if self.state == "TRANSIT":
            target = self.viewpoints[self.viewpoint_index]
            vec = target - self.pos
            if np.linalg.norm(vec) < TRANSIT_SPEED * DT:
                self.pos = np.copy(target); self.state = "INSPECT"; self.settle_timer = 20
            else: self.pos += (vec / np.linalg.norm(vec)) * TRANSIT_SPEED * DT

        elif self.state == "INSPECT":
            if self.settle_timer > 0: self.settle_timer -= 1
            else:
                self.target_queue = self.plan_optimized_path()
                if self.target_queue:
                    next_v = self.viewpoints[(self.viewpoint_index + 1) % len(self.viewpoints)]
                    pts = np.array([self.pos] + [t[0] for t in self.target_queue] + [next_v])
                    self.path_line.set_data(pts[:,0], pts[:,1])
                self.state = "TARGETING" if self.target_queue else "FINISH_VIEW"

        elif self.state == "TARGETING":
            if not self.target_queue:
                self.path_line.set_data([], []); self.state = "FINISH_VIEW"; return
                
            curr_coord, orig_idx = self.target_queue[0]
            err = curr_coord - self.pos
            if np.linalg.norm(err) < 0.8:
                # REGISTER ON ZAP (MPos Ground Truth)
                self.zapped_memory.append(np.copy(self.pos))
                self.target_queue.pop(0)
                self.settle_timer = 5 
            else:
                vel = err * KP
                self.pos += (vel / np.linalg.norm(vel) * min(np.linalg.norm(vel), 200)) * DT

        elif self.state == "FINISH_VIEW":
            self.viewpoint_index = (self.viewpoint_index + 1) % len(self.viewpoints)
            self.state = "TRANSIT"
        self.draw_frame()

    def setup_ui(self):
        self.ax_ws = self.fig.add_subplot(1, 3, 1)
        self.ax_cl = self.fig.add_subplot(1, 3, 2)
        self.ax_cr = self.fig.add_subplot(1, 3, 3)
        self.ax_ws.set_xlim(0, WS_W); self.ax_ws.set_ylim(0, WS_H); self.ax_ws.set_aspect('equal')
        self.ax_ws.set_title("RAY-MATCHING SPATIAL MEMORY")
        self.ax_ws.scatter(self.weeds[:,0], self.weeds[:,1], s=15, c='green', alpha=0.2)
        self.zap_plot = self.ax_ws.scatter([], [], s=60, c='red', marker='x', lw=1.5)
        self.path_line, = self.ax_ws.plot([], [], 'w-', alpha=0.5, lw=1)
        self.rect_l = Rectangle((0,0), FOV_W, FOV_H, fill=True, color='cyan', alpha=0.1)
        self.rect_r = Rectangle((0,0), FOV_W, FOV_H, fill=True, color='magenta', alpha=0.1)
        self.ax_ws.add_patch(self.rect_l); self.ax_ws.add_patch(self.rect_r)
        self.head_marker, = self.ax_ws.plot([], [], 'w+', markersize=15)
        self.format_cam(self.ax_cl, "LEFT FEED", 'cyan'); self.format_cam(self.ax_cr, "RIGHT FEED", 'magenta')
        self.cl_dots, = self.ax_cl.plot([], [], 'ro', mfc='none', ms=12); self.cr_dots, = self.ax_cr.plot([], [], 'ro', mfc='none', ms=12)

    def format_cam(self, ax, title, color):
        ax.set_xlim(-FOV_W/2, FOV_W/2); ax.set_ylim(-FOV_H/2, FOV_H/2); ax.set_aspect('equal')
        ax.axhline(0, color=color, alpha=0.2); ax.axvline(0, color=color, alpha=0.2)

    def draw_frame(self):
        self.head_marker.set_data([self.pos[0]], [self.pos[1]])
        self.rect_l.set_xy((self.pos[0] - 15 - FOV_W/2, self.pos[1] - FOV_H/2))
        self.rect_r.set_xy((self.pos[0] + 15 - FOV_W/2, self.pos[1] - FOV_H/2))
        if self.zapped_memory: self.zap_plot.set_offsets(np.array(self.zapped_memory))
        rel_l, rel_r = self.weeds - (self.pos + [-15, 0]), self.weeds - (self.pos + [15, 0])
        vl = rel_l[(np.abs(rel_l[:,0]) < FOV_W/2) & (np.abs(rel_l[:,1]) < FOV_H/2)]
        vr = rel_r[(np.abs(rel_r[:,0]) < FOV_W/2) & (np.abs(rel_r[:,1]) < FOV_H/2)]
        self.cl_dots.set_data(vl[:,0], vl[:,1]); self.cr_dots.set_data(vr[:,0], vr[:,1])
        self.fig.canvas.draw_idle()

sim = RayMatchingScanner()
plt.show()