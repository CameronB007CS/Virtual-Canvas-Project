"""
Virtual Canvas — Computer Vision Drawing Application
Author: Portfolio Project
Stack:  OpenCV + MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────

@dataclass
class BrushConfig:
    color: Tuple[int, int, int] = (0, 0, 255)   # BGR
    thickness: int = 8
    name: str = "Red"


@dataclass
class UIConfig:
    toolbar_height: int = 80
    padding: int = 14
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    colors: list = field(default_factory=lambda: [
        {"name": "Red",    "bgr": (50,  50,  230)},
        {"name": "Green",  "bgr": (50,  200, 80)},
        {"name": "Blue",   "bgr": (230, 100, 50)},
        {"name": "Yellow", "bgr": (30,  220, 220)},
        {"name": "White",  "bgr": (240, 240, 240)},
    ])
    thickness_levels: list = field(default_factory=lambda: [4, 8, 14, 20])


# ──────────────────────────────────────────────
#  Hand detector (MediaPipe wrapper)
# ──────────────────────────────────────────────

class HandDetector:
    """Thin wrapper around MediaPipe Hands for landmark extraction."""

    # Landmark indices
    WRIST        = 0
    THUMB_TIP    = 4
    INDEX_TIP    = 8
    MIDDLE_TIP   = 12
    RING_TIP     = 16
    PINKY_TIP    = 20
    INDEX_MCP    = 5
    MIDDLE_MCP   = 9
    RING_MCP     = 13
    PINKY_MCP    = 17

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.75,
                 tracking_conf: float = 0.65):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._draw_spec_dot = self._mp_draw.DrawingSpec(
            color=(200, 200, 200), thickness=1, circle_radius=2)
        self._draw_spec_line = self._mp_draw.DrawingSpec(
            color=(120, 120, 120), thickness=1)

    def process(self, bgr_frame: np.ndarray):
        """Return (landmarks_px, results) for the first detected hand."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        h, w = bgr_frame.shape[:2]
        landmarks_px = []
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            for lm in hand.landmark:
                landmarks_px.append((int(lm.x * w), int(lm.y * h)))
        return landmarks_px, results

    def draw_skeleton(self, frame: np.ndarray, results) -> None:
        if results.multi_hand_landmarks:
            self._mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                self._mp_hands.HAND_CONNECTIONS,
                self._draw_spec_dot,
                self._draw_spec_line,
            )

    # ── Gesture recognition ──────────────────

    @staticmethod
    def _finger_up(lm, tip_idx: int, mcp_idx: int) -> bool:
        """True when fingertip is above its MCP joint (camera y-axis inverted)."""
        return lm[tip_idx][1] < lm[mcp_idx][1]

    @classmethod
    def get_gesture(cls, lm: list) -> str:
        """
        Returns one of:
          'draw'     — only index finger raised
          'hover'    — index + middle raised (selection / eraser mode)
          'clear'    — all five fingers open (open palm)
          'fist'     — no fingers raised
          'other'    — anything else
        """
        if not lm:
            return "none"

        thumb_up  = lm[cls.THUMB_TIP][0] < lm[cls.THUMB_TIP - 1][0]  # rough thumb check
        index_up  = cls._finger_up(lm, cls.INDEX_TIP,  cls.INDEX_MCP)
        middle_up = cls._finger_up(lm, cls.MIDDLE_TIP, cls.MIDDLE_MCP)
        ring_up   = cls._finger_up(lm, cls.RING_TIP,   cls.RING_MCP)
        pinky_up  = cls._finger_up(lm, cls.PINKY_TIP,  cls.PINKY_MCP)

        fingers = [index_up, middle_up, ring_up, pinky_up]
        count   = sum(fingers)

        if count == 0:
            return "fist"
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "draw"
        if index_up and middle_up and not ring_up and not pinky_up:
            return "hover"
        if count == 4 and thumb_up:
            return "clear"
        if count == 4:
            return "clear"
        return "other"


# ──────────────────────────────────────────────
#  FPS counter (rolling average)
# ──────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ──────────────────────────────────────────────
#  Toolbar renderer
# ──────────────────────────────────────────────

class Toolbar:
    """Renders the top HUD bar and handles hit-testing for clicks/hover."""

    SWATCH_W    = 54
    SWATCH_H    = 40
    THICK_W     = 46
    THICK_H     = 40

    def __init__(self, frame_w: int, ui: UIConfig):
        self._w   = frame_w
        self._ui  = ui
        self._color_rects: list[Tuple[int,int,int,int]] = []
        self._thick_rects: list[Tuple[int,int,int,int]] = []
        self._build_layout()

    def _build_layout(self):
        x = self._ui.padding
        y = (self._ui.toolbar_height - self.SWATCH_H) // 2
        for _ in self._ui.colors:
            self._color_rects.append((x, y, x + self.SWATCH_W, y + self.SWATCH_H))
            x += self.SWATCH_W + 8

        # thickness controls — right-aligned
        x = self._w - self._ui.padding
        for _ in reversed(self._ui.thickness_levels):
            x -= self.THICK_W
            self._thick_rects.insert(0, (x, y, x + self.THICK_W, y + self.THICK_H))
            x -= 8

    def draw(self, frame: np.ndarray, brush: BrushConfig,
             gesture: str, index_tip: Optional[Tuple[int,int]]) -> None:
        h, w = self._ui.toolbar_height, self._w

        # Semi-transparent dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Bottom separator line
        cv2.line(frame, (0, h), (w, h), (60, 60, 70), 1)

        # ── Color swatches ────────────────────
        for i, (entry, rect) in enumerate(zip(self._ui.colors, self._color_rects)):
            x1, y1, x2, y2 = rect
            selected = (entry["bgr"] == brush.color)

            # Glow / selection ring
            if selected:
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 2)
            else:
                cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (60,60,70), 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), entry["bgr"], -1)

            # Hover highlight
            if index_tip and self._in_rect(index_tip, rect, toolbar_only=True):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

            # Label
            label_y = y2 + 15
            cv2.putText(frame, entry["name"], (x1+2, label_y),
                        self._ui.font, 0.35, (180,180,190), 1, cv2.LINE_AA)

        # ── Thickness buttons ─────────────────
        label_x = self._thick_rects[0][0] - 2
        cv2.putText(frame, "SIZE", (label_x - 36, 28),
                    self._ui.font, 0.38, (140,140,155), 1, cv2.LINE_AA)

        for i, (t, rect) in enumerate(zip(self._ui.thickness_levels, self._thick_rects)):
            x1, y1, x2, y2 = rect
            selected = (t == brush.thickness)
            bg_col = (50, 50, 60) if not selected else (80, 80, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_col, -1)
            border = (200,200,210) if selected else (60,60,70)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1 if not selected else 2)

            # Draw a representative dot
            cx, cy = (x1+x2)//2, (y1+y2)//2
            radius = max(2, min(t//2, 10))
            dot_col = brush.color if selected else (160,160,170)
            cv2.circle(frame, (cx, cy), radius, dot_col, -1)

            if index_tip and self._in_rect(index_tip, rect, toolbar_only=True):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

        # ── Mode badge ────────────────────────
        mode_map = {
            "draw":  ("● DRAW",   (60,190,60)),
            "hover": ("◎ HOVER",  (60,180,230)),
            "clear": ("✕ CLEAR",  (50,50,220)),
            "fist":  ("● IDLE",   (120,120,130)),
            "other": ("● IDLE",   (120,120,130)),
            "none":  ("◌ NO HAND",(90,90,100)),
        }
        txt, col = mode_map.get(gesture, ("◌", (90,90,100)))
        tx = w // 2
        (tw, _), _ = cv2.getTextSize(txt, self._ui.font, 0.55, 2)
        cv2.putText(frame, txt, (tx - tw//2, 32),
                    self._ui.font, 0.55, col, 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (tx - tw//2, 32),
                    self._ui.font, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # ── Hit-testing helpers ───────────────────

    @staticmethod
    def _in_rect(pt: Tuple[int,int], rect: Tuple[int,int,int,int],
                 toolbar_only: bool = False) -> bool:
        x, y = pt
        x1, y1, x2, y2 = rect
        if toolbar_only and y > 80:
            return False
        return x1 <= x <= x2 and y1 <= y <= y2

    def hit_color(self, pt: Tuple[int,int]) -> Optional[dict]:
        for entry, rect in zip(self._ui.colors, self._color_rects):
            if self._in_rect(pt, rect, toolbar_only=True):
                return entry
        return None

    def hit_thickness(self, pt: Tuple[int,int]) -> Optional[int]:
        for t, rect in zip(self._ui.thickness_levels, self._thick_rects):
            if self._in_rect(pt, rect, toolbar_only=True):
                return t
        return None


# ──────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────

class VirtualCanvas:
    """
    Computer-vision drawing app.

    Gestures
    --------
    Index only      → draw mode  (line follows fingertip)
    Index + middle  → hover mode (lifts pen, can select toolbar)
    Open palm (4+)  → clear canvas
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)

        # Read back actual resolution (camera may negotiate different values)
        self._W = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._H = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._detector = HandDetector()
        self._fps      = FPSCounter()
        self._ui_cfg   = UIConfig()
        self._brush    = BrushConfig()
        self._toolbar  = Toolbar(self._W, self._ui_cfg)

        # Persistent canvas (BGRA — alpha channel for erasing)
        self._canvas: np.ndarray = np.zeros((self._H, self._W, 3), dtype=np.uint8)

        # Drawing state
        self._prev_point: Optional[Tuple[int,int]] = None
        self._gesture_prev: str = "none"

        # Toolbar hover — dwell-click: hover for N frames to select
        self._hover_pt: Optional[Tuple[int,int]] = None
        self._dwell_counter: int = 0
        self._DWELL_FRAMES: int = 18       # ~0.6 s at 30 fps
        self._dwell_target: Optional[str] = None   # "color_N" or "thick_N"

    # ── Main loop ────────────────────────────

    def run(self) -> None:
        cv2.namedWindow("Virtual Canvas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Virtual Canvas", self._W, self._H)

        while True:
            ret, raw_frame = self._cap.read()
            if not ret:
                print("[ERROR] Cannot read from camera.")
                break

            frame = cv2.flip(raw_frame, 1)           # mirror view
            lm, results = self._detector.process(frame)
            fps_val      = self._fps.tick()
            gesture      = HandDetector.get_gesture(lm)
            index_tip    = lm[HandDetector.INDEX_TIP] if lm else None

            # ── Canvas update ─────────────────
            self._update_canvas(gesture, index_tip, frame)

            # ── Blend canvas onto camera frame ─
            canvas_mask = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_mask, 1, 255, cv2.THRESH_BINARY)
            frame[mask > 0] = self._canvas[mask > 0]

            # ── Draw hand skeleton (subtle) ───
            self._detector.draw_skeleton(frame, results)

            # ── Fingertip cursor ──────────────
            if index_tip:
                self._draw_cursor(frame, index_tip, gesture)

            # ── Toolbar ───────────────────────
            self._toolbar.draw(frame, self._brush, gesture, index_tip)

            # ── Toolbar dwell-click logic ─────
            if gesture == "hover" and index_tip:
                self._handle_toolbar_hover(index_tip, frame)
            else:
                self._dwell_counter = 0
                self._dwell_target  = None

            # ── HUD overlays ──────────────────
            self._draw_hud(frame, fps_val, gesture)

            cv2.imshow("Virtual Canvas", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                self._clear_canvas()

        self._cleanup()

    # ── Canvas operations ────────────────────

    def _update_canvas(self, gesture: str, index_tip: Optional[Tuple[int,int]],
                       frame: np.ndarray) -> None:
        tb_h = self._ui_cfg.toolbar_height

        if gesture == "clear":
            if self._gesture_prev != "clear":   # trigger once on transition
                self._clear_canvas()
            self._prev_point = None

        elif gesture == "draw":
            if index_tip and index_tip[1] > tb_h:
                if self._prev_point and self._prev_point[1] > tb_h:
                    cv2.line(self._canvas,
                             self._prev_point, index_tip,
                             self._brush.color, self._brush.thickness,
                             lineType=cv2.LINE_AA)
                self._prev_point = index_tip
            else:
                self._prev_point = None   # finger in toolbar — don't draw

        else:
            self._prev_point = None       # pen lifted in hover/fist/etc.

        self._gesture_prev = gesture

    def _clear_canvas(self) -> None:
        self._canvas[:] = 0

    # ── Toolbar dwell-click ───────────────────

    def _handle_toolbar_hover(self, pt: Tuple[int,int], frame: np.ndarray) -> None:
        color_entry = self._toolbar.hit_color(pt)
        thick_val   = self._toolbar.hit_thickness(pt)

        target = None
        if color_entry:
            target = f"color_{color_entry['name']}"
        elif thick_val is not None:
            target = f"thick_{thick_val}"

        if target and target == self._dwell_target:
            self._dwell_counter += 1
        else:
            self._dwell_counter = 1
            self._dwell_target  = target

        if target and self._dwell_counter >= self._DWELL_FRAMES:
            # Commit selection
            if color_entry:
                self._brush.color = color_entry["bgr"]
                self._brush.name  = color_entry["name"]
            elif thick_val is not None:
                self._brush.thickness = thick_val
            self._dwell_counter = 0

        # Draw dwell progress arc
        if target and self._dwell_counter > 0:
            pct   = self._dwell_counter / self._DWELL_FRAMES
            angle = int(360 * pct)
            cv2.ellipse(frame, pt, (18, 18), -90, 0, angle,
                        (255, 255, 255), 2, cv2.LINE_AA)

    # ── Visual helpers ───────────────────────

    def _draw_cursor(self, frame: np.ndarray, pt: Tuple[int,int], gesture: str) -> None:
        x, y = pt
        if gesture == "draw":
            r = self._brush.thickness // 2 + 2
            cv2.circle(frame, (x, y), r, self._brush.color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r + 2, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 10, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y),  2, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, fps: float, gesture: str) -> None:
        h, w = frame.shape[:2]

        # FPS — bottom-right
        fps_str = f"FPS  {fps:5.1f}"
        (tw, th), _ = cv2.getTextSize(fps_str, self._ui_cfg.font, 0.55, 1)
        x, y = w - tw - 14, h - 12
        cv2.rectangle(frame, (x-6, y-th-4), (x+tw+6, y+4), (15,15,20), -1)
        cv2.putText(frame, fps_str, (x, y), self._ui_cfg.font,
                    0.55, (100, 230, 100), 1, cv2.LINE_AA)

        # Gesture hint — bottom-left
        hints = {
            "draw":  "☞  Drawing",
            "hover": "✋  Hover / Select",
            "clear": "✕  Canvas cleared",
            "fist":  "✊  Idle",
            "other": "    —",
            "none":  "    No hand detected",
        }
        hint = hints.get(gesture, "")
        cv2.putText(frame, hint, (14, h - 14), self._ui_cfg.font,
                    0.50, (160, 160, 170), 1, cv2.LINE_AA)

        # Keyboard shortcut reminder — small
        cv2.putText(frame, "Q/Esc: quit   C: clear", (14, h - 34),
                    self._ui_cfg.font, 0.36, (80, 80, 90), 1, cv2.LINE_AA)

    # ── Cleanup ───────────────────────────────

    def _cleanup(self) -> None:
        self._cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Computer Vision Virtual Canvas")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    app = VirtualCanvas(camera_id=args.camera,
                        width=args.width,
                        height=args.height)
    app.run()