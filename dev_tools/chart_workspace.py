#!/usr/bin/env python3
"""
dev_tools/chart_workspace.py
-----------------------------
Generate a full architecture flowchart of every file in LaserWeeder_CleanRuntime.
Outputs: dev_tools/cache/workspace_chart.png

Run with:
    ./run_with_eli_venv.sh dev_tools/chart_workspace.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = REPO / "dev_tools" / "cache" / "workspace_chart.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data: (node_id, label, group, x, y, description)
#   x/y are in figure "units" — we'll place them manually on a large canvas
# =============================================================================

# ---- Color palette per group ----
PALETTE = {
    "entry":       "#1a1a2e",   # deep navy
    "config":      "#16213e",   # dark blue
    "pipeline":    "#0f3460",   # mid blue
    "hardware":    "#1b4332",   # dark green
    "vision":      "#6b2737",   # dark red
    "control":     "#6d4c41",   # brown
    "planning":    "#4a148c",   # purple
    "ui":          "#1a237e",   # indigo
    "dev_tools":   "#212121",   # near-black
    "dashboard":   "#263238",   # dark teal
    "calibration": "#37474f",   # slate
    "data":        "#3e2723",   # dark brown
    "metrics":     "#880e4f",   # dark pink
    "bringup":     "#1b5e20",   # forest green
    "params":      "#424242",   # dark grey
}

TEXT_COLOR = "#e8e8e8"
EDGE_COLOR = "#aaaaaa"
BG_COLOR   = "#0a0a0a"

# =============================================================================
# Node layout  (col, row) — we map to (x, y) linearly
# =============================================================================
# Canvas is 36 wide x 42 tall in abstract units.

COL = {
    "bringup":     0.3,
    "pipeline":    3.5,
    "entry":       3.5,
    "hardware":    7.0,
    "vision":      10.5,
    "control":     14.0,
    "planning":    17.5,
    "ui":          17.5,
    "metrics":     21.0,
    "data":        21.0,
    "dev_tools":   24.5,
    "dashboard":   27.5,
    "calibration": 24.5,
    "params":      3.5,
}

nodes = [
    # ── Entry & Config ──────────────────────────────────────────────────────
    ("main",          "main.py",                        "entry",       3.5,  0.5,
     "Entrypoint\nrun_runtime()"),
    ("hardware_setup","hardware_setup.py",              "entry",       7.0,  0.5,
     "Interactive setup wizard\n(detect cameras, serial port,\nwrite hardware_config.json)"),
    ("config",        "config.py",                      "config",      3.5,  3.5,
     "All constants & flags\n(paths, thresholds, toggles)\nloads params/hardware/*.json"),

    # ── Params (files, not Python) ───────────────────────────────────────────
    ("params_hw",     "params/hardware/\nhardware_config.json\ncamera_config.json",
                                                        "params",      0.3,  3.5,
     "Camera indices, serial port,\nexposure/gain/WB settings"),
    ("params_cv",     "params/cv_weights/\n*.pt  *.engine",
                                                        "params",      0.3,  5.5,
     "YOLO & qpoint model weights"),
    ("params_calib",  "params/calibration/\n*.npz  *.json",
                                                        "params",      0.3,  7.5,
     "Stereo calibration maps\n& pixel error model"),

    # ── Pipeline ────────────────────────────────────────────────────────────
    ("preflight",     "pipeline/preflight.py",          "pipeline",    3.5,  2.0,
     "Print env/dependency info\n(no hardware touched)"),
    ("p_context",     "pipeline/steps/context.py",      "pipeline",    3.5,  5.5,
     "RunContext dataclass\n(shared state across steps)"),
    ("p_cam_setup",   "pipeline/steps/camera_setup.py", "pipeline",    3.5,  7.0,
     "open_cameras()\nrecover on failure"),
    ("p_gantry_setup","pipeline/steps/gantry_setup.py", "pipeline",    3.5,  8.5,
     "open_gantry()\nreal or MockGantry"),
    ("p_det_setup",   "pipeline/steps/detector_setup.py","pipeline",   3.5, 10.0,
     "build_detector()\napply NMS patch\nwarmup YOLO"),
    ("p_survey",      "pipeline/steps/survey.py",       "pipeline",    3.5, 11.5,
     "run_survey_detection()\nburst detect at survey position"),
    ("p_match_plan",  "pipeline/steps/match_plan.py",   "pipeline",    3.5, 13.0,
     "run_match_and_plan()\nstereo match → triangulate\n→ plan_targets()"),
    ("runtime",       "pipeline/steps/runtime.py",      "pipeline",    3.5, 15.5,
     "Main orchestration loop\nsurvey → match → plan\n→ per-target: coarse→fine→strike"),
    ("p_fa_debug",    "pipeline/steps/fine_align_debug.py","pipeline", 3.5, 17.5,
     "save/load cached plan\ncoarse-move to target\nrun Re-ID (no PD loop)"),

    # ── Hardware ────────────────────────────────────────────────────────────
    ("cameras",       "hardware/cameras.py",            "hardware",    7.0,  4.5,
     "StereoCameras\nopen left+right USB cams\nread_pair()  record_frame()"),
    ("gantry",        "hardware/gantry.py",             "hardware",    7.0,  6.5,
     "Gantry (GRBL serial)\nmove_to()  get_status()\nwait_idle()"),
    ("mock_gantry",   "hardware/mock_gantry.py",        "hardware",    7.0,  8.5,
     "MockGantry\nFake gantry for dry-runs\nno serial required"),

    # ── Vision ──────────────────────────────────────────────────────────────
    ("ai_detector",   "vision/detectors/\nai_detector.py",  "vision",  10.5,  5.0,
     "AIDetector  +  _WeedCVCore\n• YOLO segmentation (.pt/.engine)\n• MeristemPredictor heatmap CNN\n• burst-stable detection\n• batch qpoint inference"),
    ("manual_det",    "vision/detectors/\nmanual_detector_local.py","vision",10.5, 9.5,
     "ManualDetectorLocal\nClick-to-mark weeds\nin stereo frame pair"),
    ("matching",      "vision/matching.py",             "vision",      10.5, 12.5,
     "match_points()\nEpipolar stereo matching\nleft↔right pixel pairs"),

    # ── Control ─────────────────────────────────────────────────────────────
    ("coarse_move",   "control/coarse_move.py",         "control",     14.0,  6.0,
     "TriangulationCoarseMover\npixel→mm stereo triangulation\ngantry coarse XY move\ndetect_stable_points()"),
    ("fine_align_motion","control/fine_align_motion.py","control",     14.0,  9.5,
     "fine_align_target()\nLK optical flow tracking\nsettle + snap meristem\nper-frame gantry correction"),
    ("fine_align_reid","control/fine_align_reid.py",    "control",     14.0, 12.5,
     "run_fine_align_reid()\nRe-detect after settle\nepipolar stereo re-ID\nrefine with qpoint"),
    ("strike",        "control/strike.py",              "control",     14.0, 15.5,
     "fire_target()\nArm + pulse laser\n(FIRE flag gates actual fire)\nrecord frames during strike"),
    ("calib_corr",    "control/calibration_correction.py","control",   14.0, 18.0,
     "AffineXYCorrection\nfit_affine_xy_correction()\nPost-triangulation\nlinear XY offset correction"),
    ("pixel_err",     "control/pixel_error_model.py",   "control",     14.0, 20.0,
     "StereoPixelErrorModel\nLinear regression on\nstereo pixel features\n→ predict XY error"),

    # ── Planning ────────────────────────────────────────────────────────────
    ("target_planner","planning/target_planner.py",     "planning",    17.5,  7.5,
     "plan_targets()\nFilter out-of-workspace targets\nSort by nearest-neighbour\ntour order"),

    # ── UI ──────────────────────────────────────────────────────────────────
    ("ui_terminal",   "ui/terminal.py",                 "ui",          17.5, 11.0,
     "ANSI terminal UI\nIn-place fine-align status\nTarget result / skip lines\nSSH-safe (no curses)"),
    ("ui_tri_debug",  "ui/triangulation_debug.py",      "ui",          17.5, 13.5,
     "show_match_debug_view()\nDraw epipolar lines\n& matched stereo points"),
    ("ui_ws_plot",    "ui/workspace_plot.py",            "ui",          17.5, 16.0,
     "show_workspace_triangulation_map()\nMatplotlib 3D scatter\nof triangulated targets"),

    # ── Metrics ─────────────────────────────────────────────────────────────
    ("exp_logger",    "metrics/experiment_logger.py",   "metrics",     21.0,  5.0,
     "ExperimentLogger\nCSV + JSON per-run logs\ntiming, hit/miss, counts\nfor analysis & experiments"),

    # ── Data Collection ─────────────────────────────────────────────────────
    ("grid_capture",  "data_collection/grid_capture.py","data",        21.0,  9.0,
     "grid_capture.py\nDrive gantry grid pattern\nCapture stereo images\nfor training dataset"),
    ("photo_dash",    "data_collection/photo_dashboard.py","data",     21.0, 11.5,
     "photo_dashboard.py\nFlask web UI\nBrowse training photos\n(data labelling helper)"),

    # ── Dev Tools (standalone) ──────────────────────────────────────────────
    ("benchmark",     "dev_tools/\nbenchmark_yolo_backend.py","dev_tools",24.5,3.5,
     "Benchmark .pt vs .engine\nFPS / latency comparison\non saved images"),
    ("export_trt",    "dev_tools/\nexport_tensorrt_engine.py","dev_tools",24.5,6.0,
     "Export YOLO .pt → TensorRT .engine\nusing ultralytics export API"),
    ("render_video",  "dev_tools/\nrender_trial_video.py",   "dev_tools",24.5,8.5,
     "Render annotated MP4\nfrom trial_recordings/\nfolder (YOLO overlay)"),

    # ── Dashboard ────────────────────────────────────────────────────────────
    ("dash_main",     "dev_tools/remote/\ndashboard.py",    "dashboard",27.5,  3.5,
     "Flask app entry\nSSH port-forward on :5000\nregisters all blueprints"),
    ("dash_state",    "dev_tools/remote/\ndashboard_state.py","dashboard",27.5, 5.5,
     "Shared state singleton\ncamera/gantry locks\nconfig constants"),
    ("dash_routes",   "dev_tools/remote/\ndashboard_routes.py","dashboard",27.5, 7.5,
     "URL registration\nAll Flask endpoints wired here"),
    ("dash_cam",      "dev_tools/remote/\ndashboard_camera.py","dashboard",27.5, 9.5,
     "Camera open/close/recover\nMJPEG stream  live preview\ncrop helpers"),
    ("dash_yolo",     "dev_tools/remote/\ndashboard_yolo.py","dashboard",27.5,11.5,
     "YOLO detection helpers\nburst scan debug\ncached match result"),
    ("dash_gantry",   "dev_tools/remote/\ndashboard_gantry.py","dashboard",27.5,13.5,
     "Gantry open/close\nstatus / move / jog\nraw GRBL commands"),
    ("dash_tri",      "dev_tools/remote/\ndashboard_triangulate.py","dashboard",27.5,15.5,
     "Stereo click→triangulate\nrectified pair capture\n3D matplotlib plot"),
    ("dash_fa",       "dev_tools/remote/\ndashboard_fine_align.py","dashboard",27.5,17.5,
     "Fine-align debug page\nLoad cached plan → coarse move\n→ Re-ID → show result"),
    ("dash_imgs",     "dev_tools/remote/\ndashboard_images.py","dashboard",27.5,19.5,
     "Image drawing helpers\nbase64 encode  JPEG bytes\n(no hardware, no Flask)"),
    ("dash_rect",     "dev_tools/remote/\ndashboard_rectify.py","dashboard",27.5,21.5,
     "Load stereo rectification maps\nApply remap to frame pair"),
    ("dash_settings", "dev_tools/remote/\ndashboard_settings.py","dashboard",27.5,23.5,
     "Persistent dashboard UI state\nSave/load settings JSON\nwrite config.py survey params"),
    ("dash_ws3d",     "dev_tools/remote/\ndashboard_workspace3d.py","dashboard",27.5,25.5,
     "3D workspace reprojection\nGantry XY → stereo pixels\n(inverse of triangulation)"),

    # ── Calibration tools ───────────────────────────────────────────────────
    ("cap_stereo",    "dev_tools/calibration/\ncapture_stereo_pairs.py","calibration",24.5,13.0,
     "Capture checkerboard\nstereo pairs to disk\nfor intrinsic calibration"),
    ("cap_and_calib", "dev_tools/calibration/\ncapture_and_calibrate.py","calibration",24.5,15.5,
     "Capture + run fisheye\nstereo calibration in one script"),
    ("fisheye_calib", "dev_tools/calibration/\nstereo_checkerboard\n_fisheye_calibrate.py","calibration",24.5,18.0,
     "Fisheye stereo calibration\nfrom saved checkerboard images\n→ stereo_calib.npz"),
    ("manual_tri",    "dev_tools/calibration/\nmanual_triangulation\n_calibration.py","calibration",24.5,20.5,
     "Drive gantry to known XY\nclick weed in stereo → record\ntriangulation sample"),
    ("build_err_model","dev_tools/calibration/\nbuild_stereo_pixel\n_error_model.py","calibration",24.5,23.0,
     "Fit linear error model\nfrom triangulation samples\n→ pixel_error_model.json"),

    # ── Bringup scripts ──────────────────────────────────────────────────────
    ("nms_patch",     "bringup/_nms_patch.py",          "bringup",     0.3,  0.5,
     "Monkey-patch torchvision NMS\nwith pure-PyTorch fallback\n(Jetson C++ op broken)"),
    ("b00",           "bringup/00_env_check.py",         "bringup",     0.3,  9.5,
     "Verify Python env\ncheck imports & CUDA\nno hardware"),
    ("b01",           "bringup/01_camera_open.py",       "bringup",     0.3, 11.5,
     "Open stereo cameras\nprint caps & grab frame"),
    ("b02",           "bringup/02_yolo_detection.py",    "bringup",     0.3, 13.5,
     "Load YOLO, run detection\non one camera frame"),
    ("b03",           "bringup/03_gantry_status.py",     "bringup",     0.3, 15.5,
     "Open GRBL, print\ncurrent gantry status"),
    ("b04",           "bringup/04_gantry_home.py",       "bringup",     0.3, 17.5,
     "Home the gantry\n(moves to limit switches)"),
    ("b05",           "bringup/05_gantry_move_survey.py","bringup",     0.3, 19.5,
     "Move gantry to\nsurvey position"),
    ("b06",           "bringup/06_survey_detect_only.py","bringup",     0.3, 21.5,
     "At current position:\nburst detect only\nno match/plan/move"),
    ("b07",           "bringup/07_match_plan_only.py",   "bringup",     0.3, 23.5,
     "Survey + match + plan\nMockGantry, no movement"),
    ("b08_fa",        "bringup/08_fine_align_debug.py",  "bringup",     0.3, 25.5,
     "Load cached plan\ncoarse-move + Re-ID\nno PD loop, no laser"),
    ("b08_rs",        "bringup/08_runtime_step_machine.py","bringup",   0.3, 27.5,
     "Full pipeline dry-run\nMockGantry, no laser\nno fine-align/strike"),
]

# =============================================================================
# Edges: (from_id, to_id, style)   style = "call" | "uses" | "data"
# =============================================================================

edges = [
    # main → pipeline
    ("main",        "runtime",          "call"),
    ("main",        "preflight",        "call"),
    ("main",        "config",           "uses"),

    # config ↔ params
    ("config",      "params_hw",        "data"),
    ("config",      "params_cv",        "data"),
    ("config",      "params_calib",     "data"),

    # runtime → pipeline steps
    ("runtime",     "p_context",        "call"),
    ("runtime",     "p_cam_setup",      "call"),
    ("runtime",     "p_gantry_setup",   "call"),
    ("runtime",     "p_det_setup",      "call"),
    ("runtime",     "p_survey",         "call"),
    ("runtime",     "p_match_plan",     "call"),

    # runtime → control
    ("runtime",     "coarse_move",      "call"),
    ("runtime",     "fine_align_motion","call"),
    ("runtime",     "strike",           "call"),

    # runtime → ui
    ("runtime",     "ui_terminal",      "call"),
    ("runtime",     "ui_ws_plot",       "call"),
    ("runtime",     "ui_tri_debug",     "call"),

    # runtime → metrics
    ("runtime",     "exp_logger",       "uses"),

    # pipeline setup steps → hardware
    ("p_cam_setup",  "cameras",         "call"),
    ("p_gantry_setup","gantry",         "call"),
    ("p_gantry_setup","mock_gantry",    "call"),
    ("p_det_setup",  "ai_detector",     "call"),

    # survey → coarse_move + cameras + detector
    ("p_survey",     "coarse_move",     "call"),
    ("p_survey",     "cameras",         "uses"),

    # match_plan → matching + planner
    ("p_match_plan", "matching",        "call"),
    ("p_match_plan", "target_planner",  "call"),

    # control flow
    ("coarse_move",  "cameras",         "uses"),
    ("coarse_move",  "ai_detector",     "call"),
    ("coarse_move",  "matching",        "call"),
    ("coarse_move",  "calib_corr",      "uses"),
    ("coarse_move",  "pixel_err",       "uses"),
    ("coarse_move",  "gantry",          "call"),

    ("fine_align_motion","cameras",     "uses"),
    ("fine_align_motion","gantry",      "call"),
    ("fine_align_motion","ai_detector", "call"),
    ("fine_align_motion","fine_align_reid","call"),
    ("fine_align_motion","ui_terminal", "call"),

    ("fine_align_reid","matching",      "call"),
    ("fine_align_reid","ai_detector",   "call"),

    ("strike",       "gantry",          "call"),
    ("strike",       "cameras",         "uses"),

    # vision
    ("ai_detector",  "params_cv",       "data"),

    # matching
    ("matching",     "params_calib",    "data"),
    ("matching",     "ui_terminal",     "call"),

    # bringup NMS patch
    ("nms_patch",    "ai_detector",     "uses"),
    ("b02",          "nms_patch",       "uses"),
    ("b02",          "cameras",         "call"),
    ("b06",          "nms_patch",       "uses"),
    ("b06",          "cameras",         "call"),
    ("b06",          "ai_detector",     "call"),
    ("b07",          "nms_patch",       "uses"),
    ("b07",          "matching",        "call"),
    ("b07",          "target_planner",  "call"),
    ("b08_fa",       "nms_patch",       "uses"),
    ("b08_fa",       "p_fa_debug",      "call"),
    ("b08_rs",       "runtime",         "call"),
    ("b01",          "cameras",         "call"),
    ("b03",          "gantry",          "call"),
    ("b04",          "gantry",          "call"),
    ("b05",          "gantry",          "call"),
    ("p_fa_debug",   "coarse_move",     "call"),
    ("p_fa_debug",   "fine_align_reid", "call"),

    # dev_tools
    ("benchmark",    "ai_detector",     "call"),
    ("export_trt",   "params_cv",       "data"),
    ("render_video", "ai_detector",     "call"),
    ("render_video", "cameras",         "uses"),

    # calibration tools
    ("cap_stereo",   "cameras",         "call"),
    ("cap_and_calib","cameras",         "call"),
    ("fisheye_calib","params_calib",    "data"),
    ("manual_tri",   "gantry",          "call"),
    ("manual_tri",   "cameras",         "call"),
    ("manual_tri",   "coarse_move",     "call"),
    ("build_err_model","pixel_err",     "uses"),
    ("build_err_model","params_calib",  "data"),

    # data collection
    ("grid_capture", "cameras",         "call"),
    ("grid_capture", "gantry",          "call"),
    ("photo_dash",   "config",          "uses"),

    # dashboard
    ("dash_main",    "dash_state",      "uses"),
    ("dash_main",    "dash_routes",     "call"),
    ("dash_routes",  "dash_cam",        "call"),
    ("dash_routes",  "dash_yolo",       "call"),
    ("dash_routes",  "dash_gantry",     "call"),
    ("dash_routes",  "dash_tri",        "call"),
    ("dash_routes",  "dash_fa",         "call"),
    ("dash_routes",  "dash_settings",   "call"),
    ("dash_routes",  "dash_ws3d",       "call"),
    ("dash_cam",     "cameras",         "call"),
    ("dash_cam",     "dash_state",      "uses"),
    ("dash_cam",     "dash_imgs",       "uses"),
    ("dash_cam",     "dash_rect",       "uses"),
    ("dash_yolo",    "ai_detector",     "call"),
    ("dash_yolo",    "dash_state",      "uses"),
    ("dash_gantry",  "gantry",          "call"),
    ("dash_gantry",  "dash_state",      "uses"),
    ("dash_tri",     "matching",        "call"),
    ("dash_tri",     "dash_rect",       "uses"),
    ("dash_tri",     "dash_imgs",       "uses"),
    ("dash_fa",      "nms_patch",       "uses"),
    ("dash_fa",      "coarse_move",     "call"),
    ("dash_fa",      "fine_align_reid", "call"),
    ("dash_rect",    "params_calib",    "data"),
    ("dash_settings","config",          "uses"),
    ("dash_ws3d",    "params_calib",    "data"),
    ("hardware_setup","cameras",        "call"),
    ("hardware_setup","gantry",         "call"),
    ("hardware_setup","params_hw",      "data"),
]

# =============================================================================
# Draw
# =============================================================================

SCALE_X = 1.05
SCALE_Y = 1.1
FIG_W    = 38
FIG_H    = 32

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-0.5, 31)
ax.set_ylim(-1.5, 30)
ax.axis("off")
ax.invert_yaxis()

# Build lookup
node_map = {n[0]: n for n in nodes}
pos_map  = {n[0]: (n[3], n[4]) for n in nodes}   # x, y

BOX_W = 2.55
BOX_H = 1.05

EDGE_STYLES = {
    "call":  dict(color="#4dd0e1", lw=0.9, alpha=0.55, style="solid"),
    "uses":  dict(color="#a5d6a7", lw=0.65, alpha=0.45, style="dashed"),
    "data":  dict(color="#ffcc80", lw=0.65, alpha=0.45, style="dotted"),
}

# ── Draw edges first (behind boxes) ──
for src, dst, style in edges:
    if src not in pos_map or dst not in pos_map:
        continue
    x0, y0 = pos_map[src]
    x1, y1 = pos_map[dst]
    es = EDGE_STYLES[style]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=es["color"],
            lw=es["lw"],
            alpha=es["alpha"],
            linestyle=es["style"],
            connectionstyle="arc3,rad=0.08",
        ),
        zorder=1,
    )

# ── Group label banners ──
GROUP_LABELS = {
    "bringup":     (0.3,  -0.8, "BRINGUP  (sequential debug scripts)"),
    "entry":       (3.5,  -0.8, "ENTRY"),
    "pipeline":    (3.5,   4.7, "PIPELINE  /steps"),
    "hardware":    (7.0,   3.8, "HARDWARE"),
    "vision":      (10.5,  3.8, "VISION"),
    "control":     (14.0,  4.7, "CONTROL"),
    "planning":    (17.5,  6.3, "PLANNING"),
    "ui":          (17.5,  9.7, "UI"),
    "metrics":     (21.0,  3.8, "METRICS"),
    "data":        (21.0,  7.6, "DATA COLLECTION"),
    "dev_tools":   (24.5,  2.3, "DEV TOOLS"),
    "dashboard":   (27.5,  2.3, "REMOTE DASHBOARD  (Flask)"),
    "calibration": (24.5, 11.5, "CALIBRATION TOOLS"),
    "params":      (0.3,   2.3, "params/  (files)"),
    "config":      (3.5,   2.5, "CONFIG"),
}

for grp, (gx, gy, label) in GROUP_LABELS.items():
    color = PALETTE[grp]
    ax.text(
        gx, gy, label,
        color=TEXT_COLOR, fontsize=6.5, fontweight="bold",
        ha="center", va="center", alpha=0.55,
        zorder=2,
    )

# ── Draw boxes ──
for nid, label, group, nx, ny, desc in nodes:
    color = PALETTE.get(group, "#333333")

    # box shadow
    shadow = FancyBboxPatch(
        (nx - BOX_W/2 + 0.05, ny - BOX_H/2 + 0.05),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.08",
        facecolor="#000000", edgecolor="none",
        alpha=0.5, zorder=2,
    )
    ax.add_patch(shadow)

    # main box
    box = FancyBboxPatch(
        (nx - BOX_W/2, ny - BOX_H/2),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#555555",
        linewidth=0.6, alpha=0.93, zorder=3,
    )
    ax.add_patch(box)

    # file label (top, bold)
    ax.text(
        nx, ny - 0.22,
        label,
        color=TEXT_COLOR, fontsize=5.2, fontweight="bold",
        ha="center", va="center", zorder=4,
        linespacing=1.3,
    )

    # description (bottom, lighter)
    ax.text(
        nx, ny + 0.3,
        desc,
        color="#cccccc", fontsize=3.8,
        ha="center", va="center", zorder=4,
        linespacing=1.25,
        style="italic",
    )

# ── Title ──
ax.text(
    15.0, -1.15,
    "LaserWeeder  —  Full Workspace Architecture",
    color="#e0e0e0", fontsize=13, fontweight="bold",
    ha="center", va="center", zorder=5,
)

# ── Legend ──
legend_x, legend_y = 0.3, 29.0
for i, (style_key, label) in enumerate([
    ("call", "→ function call / instantiate"),
    ("uses", "- - uses / imports"),
    ("data", "··· reads/writes file data"),
]):
    es = EDGE_STYLES[style_key]
    ax.annotate(
        "",
        xy=(legend_x + 0.7, legend_y + i * 0.55),
        xytext=(legend_x, legend_y + i * 0.55),
        arrowprops=dict(
            arrowstyle="-|>",
            color=es["color"],
            lw=1.5,
            alpha=0.9,
            linestyle=es["style"],
        ),
        zorder=6,
    )
    ax.text(
        legend_x + 0.85, legend_y + i * 0.55,
        label,
        color=TEXT_COLOR, fontsize=6.5, va="center", zorder=6,
    )

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"[chart] Saved → {OUT}")
print(f"[chart] Image size: {OUT.stat().st_size // 1024} KB")
