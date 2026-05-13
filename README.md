# Virtual Canvas — Computer Vision Drawing App
### Built with OpenCV + MediaPipe | Python

---

## What Is This?

A real-time drawing application that turns your webcam and bare hands into a canvas and brush — no mouse, no touchscreen, no hardware.

Point your index finger at the camera and draw. Raise two fingers to hover and switch colours. Open your palm to wipe the canvas clean. That's it.

I built this because I wanted a project that *felt* like real software — something I could show someone in 30 seconds and they'd immediately understand what it does. No explaining required.

---

## Demo

>  *Record some sort of tutorial demo showing my app in motion*
>  `Demo Link will go here soon`

---

## How It Works

The application runs a continuous loop at 30+ FPS. Every frame goes through three stages:

**1. Hand Detection**
MediaPipe processes the webcam feed and returns 21 3D landmarks mapped across the hand — every knuckle, fingertip, and joint. These come back as normalised coordinates between 0 and 1.

**2. Coordinate Mapping**
Those normalised coordinates get multiplied by the frame dimensions to produce exact pixel positions. The fingertip becomes a point on a 2D canvas that mirrors your physical movement in real time.

**3. Gesture Recognition**
The app compares each fingertip's y-position against its MCP (knuckle) joint. If the tip is above the knuckle, that finger is raised. This lets the app distinguish between:

| Gesture | Action |
|---|---|
| ☞ Index finger only | Draw mode — line follows fingertip |
|  Index + Middle | Hover mode — pen lifts, toolbar active |
|  Open palm | Clear canvas |
|  Fist | Idle |

---

## Features

- **Real-time drawing** at 30+ FPS with zero perceptible lag
- **Colour picker** — Red, Green, Blue, Yellow, White
- **Brush size selector** — 4 thickness levels
- **Dwell-click toolbar** — hover your finger over a colour for ~0.6 seconds to select it (no physical click needed)
- **FPS counter** — live performance readout in the corner
- **Hand skeleton overlay** — subtle landmark visualisation while you draw
- **Class-based architecture** — five focused classes, each with a single responsibility

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| OpenCV | Webcam capture, canvas rendering, drawing primitives |
| MediaPipe | Hand landmark detection and tracking |
| NumPy | Frame manipulation as arrays |

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/CameronB007CS/Virtual-Canvas-Project.git
cd Virtual-Canvas-Project
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run**
```bash
python virtual_canvas.py
```

**Optional flags**
```bash
python virtual_canvas.py --camera 1        # use a different camera
python virtual_canvas.py --width 1920 --height 1080
```

Press **Q** or **Esc** to quit. Press **C** to clear the canvas from keyboard.

---

## Code Architecture

The codebase is structured around five classes with clear separation of concerns — the kind of structure you'd find in a production codebase, not a script.

```
HandDetector     — MediaPipe wrapper, landmark extraction, gesture logic
FPSCounter       — Rolling average FPS using a deque
Toolbar          — HUD rendering and dwell-click hit detection
VirtualCanvas    — Main loop, canvas state, orchestration
BrushConfig      — Brush state as a clean dataclass
UIConfig         — UI constants and colour definitions
```

The canvas itself is a persistent NumPy array that only gets written to when a new line segment is drawn. No re-rendering history each frame — just blend the canvas array onto the live camera feed using a binary mask. This is what keeps the frame rate high.

---

## What I Learned

This project taught me things a classroom assignment wouldn't:

- **Real-time constraints are unforgiving.** Every millisecond of processing delay shows up as lag between your finger and the line. You learn quickly what's expensive and what isn't.
- **Coordinate systems matter.** Camera space, normalised space, pixel space — getting these wrong means your drawing appears somewhere completely different to your finger.
- **Gesture design is a UX problem, not just a technical one.** The gestures have to feel natural enough that someone picks it up in 10 seconds without reading any instructions.

---

## Where This Goes Next

This project is a foundation. The same hand tracking and coordinate mapping skills apply directly to:

- **Sign Language Translator** — classify gestures against a trained ML model to translate ASL letters in real time
- **Gesture-Controlled Interfaces** — map hand position to control music, navigate presentations, or interact with any application without a mouse
- **Rehabilitation Tracking Tool** — measure joint angles and range of motion using landmark distances, with real clinical applications in physiotherapy
- **AR Collaborative Whiteboard** — multi-hand drawing over a network connection, combining CV with sockets and real-time data sync

The building block is the same: track a hand, extract meaning from its shape, map that meaning to an action. What changes is what the action does.

---

## About

First year Computer Science student interested in the intersection of computer vision, real-time systems, and human-computer interaction. This project started as a curiosity — *can I build something that reacts to the physical world using nothing but a webcam and Python?* — and turned into one of the most technically interesting things I've built so far.

Always open to feedback, collaboration, or just a conversation about CV and ML projects.

**GitHub:** [CameronB007CS](https://github.com/CameronB007CS)
