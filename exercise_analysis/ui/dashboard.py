"""Fixed 1120 x 630 canvas with a letterboxed view and a separate instrument panel."""
import cv2
import numpy as np

from exercise_analysis.ui.skeleton import draw_skeleton

BG = (28, 25, 22)
PANEL = (44, 39, 34)
INK = (241, 244, 241)
MUTED = (179, 176, 164)
MINT = (161, 229, 129)
AMBER = (92, 201, 255)
CORAL = (130, 137, 255)
BLUE = (246, 193, 144)
STATUS = {"GOOD": MINT, "BAD": CORAL, "UNASSESSABLE": BLUE}
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(canvas, value, x, y, size=.5, color=INK, width=None, weight=1):
    value = str(value)
    if width:
        measured = cv2.getTextSize(value, FONT, size, weight)[0][0]
        size *= min(1., width / max(1, measured))
    cv2.putText(canvas, value, (x, y), FONT, size, color, weight, cv2.LINE_AA)


def render(bgr, pose, state, visibility, source, help_open=False):
    canvas = np.full((630, 1120, 3), BG, np.uint8)
    # Header
    cv2.rectangle(canvas, (20, 18), (26, 40), MINT, -1)
    text(canvas, "Exercise analysis", 40, 39, .83)
    text(canvas, source.upper(), 796, 36, .48, MUTED)
    minutes, seconds = divmod(int(state.elapsed), 60)
    text(canvas, f"{minutes:02d}:{seconds:02d}", 982, 39, .78, width=116)

    # Resize before drawing landmarks, keeping inference geometry independent of the HUD.
    frame_h, frame_w = bgr.shape[:2]
    scale = min(692 / frame_w, 386 / frame_h)
    w, h = max(1, round(frame_w * scale)), max(1, round(frame_h * scale))
    preview = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    draw_skeleton(preview, pose, visibility, w, h)
    cv2.rectangle(canvas, (20, 60), (711, 445), (16, 15, 14), -1)
    x, y = 20 + (692-w)//2, 60 + (386-h)//2
    canvas[y:y+h, x:x+w] = preview

    # Recognition and cumulative session scores
    cv2.rectangle(canvas, (726, 60), (1099, 503), PANEL, -1)
    mode_color = BLUE if state.decision.mode == "Manual" else MINT
    text(canvas, state.decision.mode.upper(), 746, 91, .5, mode_color)
    active = state.decision.active
    text(canvas, active, 746, 132, 1.05, AMBER if active == "Unknown" else INK, 331)
    pred = state.prediction
    score = "--" if pred.missing_input else f"{pred.top_score:.2f}"
    text(canvas, f"Frame: {pred.exercise or 'Unknown'}   Score: {score}", 746, 162, .49, MUTED, 332)
    cv2.line(canvas, (746, 180), (1077, 180), (77, 72, 65))
    if state.highlight:
        cv2.rectangle(canvas, (733, 187), (1090, 354), STATUS[state.last.status], 1)
    text(canvas, "SESSION TOTALS", 746, 207, .45, MUTED)
    text(canvas, state.quality["GOOD"], 746, 274, 2., MINT, 141, 2)
    text(canvas, "GOOD REPS", 746, 300, .51, MINT)
    text(canvas, state.completed, 923, 274, 1.38, INK, 148)
    text(canvas, "TOTAL CYCLES", 923, 300, .49, MUTED)
    text(canvas, f"BAD   {state.quality['BAD']}", 746, 342, .56, CORAL, 125)
    text(canvas, f"UNASSESSABLE   {state.quality['UNASSESSABLE']}", 888, 342, .47, BLUE, 190)
    cv2.line(canvas, (746, 362), (1077, 362), (77, 72, 65))
    motion_label = f"MOTION  /  {state.count.exercise}" if state.count.exercise and state.count.exercise != active else "MOTION"
    text(canvas, motion_label, 746, 390, .45, MUTED)
    text(canvas, state.count.phase, 746, 427, .84, INK, 246)
    progress = state.count.progress
    text(canvas, f"{progress:.0%}", 1001, 425, .65, MUTED, 76)
    cv2.rectangle(canvas, (746, 447), (1077, 457), BG, -1)
    if progress > 0:
        cv2.rectangle(canvas, (746, 447), (746 + round(331*progress), 457), MINT, -1)

    for i, (name, value) in enumerate(state.angles._asdict().items()):
        x = 22 + 138*i
        text(canvas, name.split("_")[0].upper(), x, 469, .39, MUTED)
        text(canvas, "--" if value is None else f"{value:.0f} deg", x, 495, .57)

    cv2.rectangle(canvas, (20, 517), (1099, 590), PANEL, -1)
    accent = CORAL if not state.tracking else AMBER if state.count.warning else MINT
    cv2.rectangle(canvas, (20, 517), (25, 590), accent, -1)
    text(canvas, "FEEDBACK", 38, 541, .4, MUTED)
    text(canvas, state.feedback, 38, 574, .75, INK, 645)
    last = state.last
    last_color = STATUS[last.status] if last else MUTED
    if state.highlight:
        cv2.rectangle(canvas, (724, 518), (1098, 589), last_color, 2)
    text(canvas, "LAST COMPLETED", 743, 536, .4, MUTED)
    text(canvas, f"{last.cycle.exercise}   {last.status}" if last else "--",
         743, 561, .60, last_color, 337)
    if last:
        text(canvas, last.feedback, 743, 581, .44, MUTED, 337)
    text(canvas, "H   Controls", 22, 617, .45, MUTED)
    text(canvas, "Q / Esc   Finish session", 890, 617, .45, MUTED, 208)
    if help_open:
        cv2.rectangle(canvas, (726, 60), (1099, 503), PANEL, -1)
        text(canvas, "Session controls", 748, 108, .84)
        for i, row in enumerate(("1    Squat", "2    Push-up", "3    Pull-up", "4    Jumping Jack",
                                 "5    Russian Twist", "A    Automatic recognition",
                                 "H    Close controls", "Q / Esc    Finish and save")):
            text(canvas, row, 750, 157+i*43, .6, INK)
    return canvas
