"""Live ASV four-camera SHORELINE + OBSTACLE detector (single self-contained file).

Run from the project root in a separate Command Prompt BEFORE starting MATLAB:
    python live_obstacle_shoreline_detector.py

Watches live_detection_io/<cam>_input.png and writes <cam>_overlay.png.

What was wrong with the previous combined script
  It embedded the OLD V4 boundary detector, which locks onto the sky/water
  horizon. The obstacle search band was therefore built around the horizon,
  and anything larger than MAX_WIDTH/MAX_AREA (all the boxes here) was rejected.

How this version works
  Shoreline : land mask -> lowest land pixel per column -> water-below check
              -> gap bridging + smoothing (the version that worked for you).
  Horizon   : strongest bright->dark vertical edge (sky -> water/land).
  Obstacles : anything below the horizon that is NOT saturated blue water and
              NOT green/olive land (grey/white boxes, dark slate boxes, cubes).
              The own-boat hull and the wake glow are removed automatically:
              first by a size rule, then by a static mask learned from pixels
              that stay "non-water" for many frames.
              Each obstacle is tagged WATER or SHORE using the shoreline.
Experimental, tuned for the synthetic scene, not safety validated.
"""
from pathlib import Path
import os
import time
import uuid

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
IO_DIR = PROJECT_ROOT / "live_detection_io"
CAMERAS = ("front", "right", "rear", "left")
POLL_SECONDS = 0.025
SAVE_DEBUG_MASKS = False          # True -> also writes <cam>_mask.png

# ---------------- HSV thresholds (OpenCV H 0-179) ---------------------------
LAND_LOWER = np.array([18, 60, 10], dtype=np.uint8)     # olive beach + dark green
LAND_UPPER = np.array([82, 255, 235], dtype=np.uint8)
WATER_LOWER = np.array([85, 60, 8], dtype=np.uint8)     # blue / cyan-blue water
WATER_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# ---------------- Shoreline tuning ------------------------------------------
MIN_LAND_AREA_RATIO = 0.0015
WATER_CHECK_OFFSET = 4
WATER_CHECK_DEPTH = 14
WATER_CHECK_MIN_FRAC = 0.5
BOTTOM_MARGIN = 3
MAX_GAP = 12
MIN_SEGMENT_LEN = 25
MEDIAN_WIN = 9
MEAN_WIN = 15
TEMPORAL_ALPHA = 0.6
MIN_CONFIDENCE = 0.20

# ---------------- Obstacle tuning -------------------------------------------
WATER_SAT_MIN = 205               # real water is very saturated; boxes are below this
MIN_VALUE = 15                    # ignore near-black pixels (unreliable hue)
HORIZON_MARGIN = 3                # ignore this many rows under the horizon line
OBS_MIN_AREA = 30
OBS_MIN_SIDE = 4                  # min width and height (px)
OBS_MAX_ASPECT = 6.0              # drops thin horizon-haze strips (w/h)
OBS_MAX_W_RATIO = 0.60            # max box width  as fraction of frame width
OBS_MAX_H_RATIO = 0.60            # max box height as fraction of frame height
OBS_CLOSE_KERNEL = 5              # joins the lit/shaded faces of one box
HULL_MIN_AREA_RATIO = 0.015       # bottom-touching blobs bigger than this = own hull
HULL_LEARN_FRAMES = 60            # frames before the static hull mask is trusted
HULL_PERSIST_ALPHA = 0.05
HULL_PERSIST_THRESH = 0.85
SHORE_TAG_MARGIN = 3              # px above shoreline => obstacle is on land
ONLY_WATER_OBSTACLES = True       # True: report only obstacles standing in the water
BASE_STRIP_ROWS = (2, 10)         # rows just under a box base used to tell water from land
BASE_LAND_FRAC = 0.5              # >= this land share under the base => obstacle is on shore
RING_PX = 12                      # surrounding ring used as a second land/water check
RING_LAND_FRAC = 0.25             # >= this land share around the box => obstacle is on shore

_state = {}                       # per-camera memory


# =============================================================================
# helpers
# =============================================================================
def _sliding_median(a, win):
    p = np.pad(a, win // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, win), axis=1)


def _moving_mean(a, win):
    p = np.pad(a, win // 2, mode="edge")
    return np.convolve(p, np.ones(win) / win, mode="valid")


def _finite_runs(arr, min_len=1):
    idx = np.flatnonzero(np.isfinite(arr))
    if idx.size == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            yield int(s), int(e)


def segment_masks(hsv):
    land = cv2.inRange(hsv, LAND_LOWER, LAND_UPPER)
    water = cv2.inRange(hsv, WATER_LOWER, WATER_UPPER)
    land = cv2.morphologyEx(land, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(land, 8)
    if n > 1:
        keep = np.concatenate(([False], stats[1:, cv2.CC_STAT_AREA]
                               >= MIN_LAND_AREA_RATIO * land.size))
        land = (keep[labels] * 255).astype(np.uint8)
    return land > 0, water > 0


def estimate_horizon(frame):
    """Row of the strongest bright->dark edge (sky above, water/land below)."""
    h = frame.shape[0]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 2)
    g = g.astype(np.float32)
    y0, y1 = int(h * 0.20), int(h * 0.80)
    d = g[y0 + 5:y1, :] - g[y0 - 5:y1 - 10, :]
    return int(np.median(np.argmin(d, axis=0) + y0))


# =============================================================================
# shoreline
# =============================================================================
def compute_shoreline(land, water, cam):
    h, w = land.shape
    has_land = land.any(axis=0)
    bottom = (h - 1 - np.argmax(land[::-1, :], axis=0)).astype(np.int32)

    cs = np.zeros((h + 1, w), dtype=np.int32)
    np.cumsum(water, axis=0, out=cs[1:])
    lo = np.minimum(h, bottom + WATER_CHECK_OFFSET)
    hi = np.minimum(h, bottom + WATER_CHECK_OFFSET + WATER_CHECK_DEPTH)
    cols = np.arange(w)
    depth = hi - lo
    frac = (cs[hi, cols] - cs[lo, cols]) / np.maximum(depth, 1)
    valid = (has_land & (bottom < h - BOTTOM_MARGIN)
             & (depth >= 6) & (frac >= WATER_CHECK_MIN_FRAC))

    y = np.full(w, np.nan, dtype=np.float32)
    y[valid] = bottom[valid]
    shore = np.full(w, np.nan, dtype=np.float32)
    idx = np.flatnonzero(valid)
    if idx.size:
        splits = np.flatnonzero(np.diff(idx) > MAX_GAP + 1)
        starts = np.r_[idx[0], idx[splits + 1]]
        ends = np.r_[idx[splits], idx[-1]]
        for s, e in zip(starts, ends):
            if valid[s:e + 1].sum() < MIN_SEGMENT_LEN:
                continue
            xs = np.arange(s, e + 1)
            vx = np.flatnonzero(valid[s:e + 1]) + s
            seg = np.interp(xs, vx, y[vx])
            shore[s:e + 1] = _moving_mean(_sliding_median(seg, MEDIAN_WIN), MEAN_WIN)

    st = _state.setdefault(cam, {})
    prev = st.get("shore")
    if prev is not None and prev.shape == shore.shape:
        both = np.isfinite(shore) & np.isfinite(prev)
        shore[both] = TEMPORAL_ALPHA * shore[both] + (1 - TEMPORAL_ALPHA) * prev[both]
    st["shore"] = shore.copy()

    n_shore = int(np.isfinite(shore).sum())
    n_land = int(has_land.sum())
    conf = n_shore / n_land if n_land else 0.0
    found = n_shore >= MIN_SEGMENT_LEN and conf >= MIN_CONFIDENCE
    return shore, found, conf


# =============================================================================
# obstacles
# =============================================================================
def _touching_bottom(mask_u8):
    """Keep only connected components that touch the bottom image rows."""
    n, labels, _, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)
    touch = np.unique(labels[-2:, :])
    touch = touch[touch > 0]
    return np.isin(labels, touch)


def hull_mask(raw, cam):
    """Own-boat hull / wake glow: static, large, connected to the bottom edge."""
    h, w = raw.shape
    st = _state.setdefault(cam, {})
    persist = st.get("persist")
    if persist is None or persist.shape != raw.shape:
        persist = np.zeros(raw.shape, dtype=np.float32)
        st["frames"] = 0
    persist = (1 - HULL_PERSIST_ALPHA) * persist + HULL_PERSIST_ALPHA * raw
    st["persist"] = persist
    st["frames"] = st.get("frames", 0) + 1

    if st["frames"] >= HULL_LEARN_FRAMES:
        static = (persist > HULL_PERSIST_THRESH).astype(np.uint8) * 255
        hull = _touching_bottom(static)
        return cv2.dilate(hull.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    # warm-up: size rule (big blobs touching the bottom edge)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    hull = np.zeros(raw.shape, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if y + bh >= h - 2 and area >= HULL_MIN_AREA_RATIO * h * w:
            hull |= labels == i
    return hull


def _shore_y_near(shore, cx, reach=80):
    """Shoreline row at column cx (or at the nearest column within `reach`)."""
    if shore is None:
        return None
    if np.isfinite(shore[cx]):
        return float(shore[cx])
    lo, hi = max(0, cx - reach), min(len(shore), cx + reach + 1)
    seg = shore[lo:hi]
    ok = np.flatnonzero(np.isfinite(seg))
    if ok.size == 0:
        return None
    j = ok[np.argmin(np.abs(ok + lo - cx))]
    return float(seg[j])


def detect_obstacles(frame, hsv, land, water, horizon, shore, cam):
    h, w = land.shape
    cand = (hsv[:, :, 1] < WATER_SAT_MIN) & (hsv[:, :, 2] >= MIN_VALUE)
    cand &= ~cv2.dilate(land.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    cand[:min(h, horizon + HORIZON_MARGIN), :] = False

    raw = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)).astype(bool)
    hull = hull_mask(raw, cam)
    mask = (raw & ~hull).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((OBS_CLOSE_KERNEL, OBS_CLOSE_KERNEL), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    obstacles = []
    span = max(1, h - horizon)
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < OBS_MIN_AREA or bw < OBS_MIN_SIDE or bh < OBS_MIN_SIDE:
            continue
        if bw > OBS_MAX_W_RATIO * w or bh > OBS_MAX_H_RATIO * h:
            continue
        if bw / float(bh) > OBS_MAX_ASPECT:
            continue
        cx = int(np.clip(x + bw // 2, 0, w - 1))
        bottom = y + bh
        where = "water"
        r0, r1 = BASE_STRIP_ROWS
        ya, yb = min(h, bottom + r0), min(h, bottom + r1)
        if yb > ya:
            lstrip = land[ya:yb, x:x + bw]
            wstrip = water[ya:yb, x:x + bw]
            nl, nw = int(lstrip.sum()), int(wstrip.sum())
            if nl + nw > 0 and nl / float(nl + nw) >= BASE_LAND_FRAC:
                where = "shore"
        if where == "water":
            x0r, y0r = max(0, x - RING_PX), max(0, y - RING_PX)
            x1r, y1r = min(w, x + bw + RING_PX), min(h, y + bh + RING_PX)
            lr, wr = land[y0r:y1r, x0r:x1r], water[y0r:y1r, x0r:x1r]
            box = np.zeros(lr.shape, dtype=bool)
            box[y - y0r:y - y0r + bh, x - x0r:x - x0r + bw] = True
            nl, nw = int(lr[~box].sum()), int(wr[~box].sum())
            if nl + nw >= 20 and nl / float(nl + nw) >= RING_LAND_FRAC:
                where = "shore"
        ref = _shore_y_near(shore, cx)
        if where == "water" and ref is not None and bottom < ref - SHORE_TAG_MARGIN:
            where = "shore"
        near = float(np.clip((bottom - horizon) / span, 0.0, 1.0))
        obstacles.append({"x": x, "y": y, "w": bw, "h": bh, "area": area,
                          "near": near, "where": where})
    if ONLY_WATER_OBSTACLES:
        obstacles = [o for o in obstacles if o["where"] == "water"]
    obstacles.sort(key=lambda o: o["near"], reverse=True)    # closest first
    return obstacles, mask


# =============================================================================
# per-frame pipeline + drawing
# =============================================================================
def annotate_frame(frame, cam):
    h, w = frame.shape[:2]
    out = frame.copy()
    if h < 40 or w < 40:
        return out, 0, False, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    land, water = segment_masks(hsv)
    horizon = estimate_horizon(frame)

    shore, shore_ok, shore_conf = compute_shoreline(land, water, cam)
    obstacles, mask = detect_obstacles(frame, hsv, land, water, horizon,
                                       shore if shore_ok else None, cam)

    if shore_ok:
        for s, e in _finite_runs(shore, MIN_SEGMENT_LEN):
            xs = np.arange(s, e, 3)
            if xs[-1] != e - 1:
                xs = np.r_[xs, e - 1]
            pts = np.column_stack((xs, np.clip(shore[xs], 0, h - 1))).round().astype(np.int32)
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 3, cv2.LINE_AA)
        xn = int(np.nanargmax(shore))
        cv2.circle(out, (xn, int(round(float(shore[xn])))), 7, (0, 255, 255), 2, cv2.LINE_AA)

    for i, o in enumerate(obstacles[:20], 1):
        color = (0, 0, 255) if o["where"] == "water" else (0, 140, 255)
        x, y, bw, bh = o["x"], o["y"], o["w"], o["h"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(out, f"OBS {i} {o['where']} {o['near']:.2f}", (x, max(58, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    s_label = (f"SHORELINE | conf {shore_conf:.2f}" if shore_ok
               else f"NO SHORE VISIBLE | conf {shore_conf:.2f}")
    s_color = (0, 255, 0) if shore_ok else (0, 165, 255)
    cv2.rectangle(out, (8, 8), (min(w - 8, 460), 42), (0, 0, 0), -1)
    cv2.putText(out, s_label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (8, 46), (min(w - 8, 460), 76), (0, 0, 0), -1)
    cv2.putText(out, f"OBSTACLES: {len(obstacles)} | horizon y={horizon}", (16, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out, len(obstacles), shore_ok, mask, shore_conf


def atomic_write_image(path: Path, image, retries=15):
    temp = path.with_name(f"{path.stem}_{uuid.uuid4().hex}_tmp.png")
    try:
        if not cv2.imwrite(str(temp), image):
            return False
        for attempt in range(retries):
            try:
                os.replace(str(temp), str(path))
                return True
            except PermissionError:
                time.sleep(0.02 * (attempt + 1))
        return False
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass


def main():
    IO_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("ASV LIVE SHORELINE + OBSTACLE DETECTOR - FOUR CAMERAS")
    print(f"Exchange folder: {IO_DIR}")
    print(f"Learning own-hull mask for the first {HULL_LEARN_FRAMES} frames per camera.")
    print("Experimental visual detections only; not safety validated.")
    print("Waiting for MATLAB frames. Press Ctrl+C to stop.")
    print("=" * 66)

    seen = {name: None for name in CAMERAS}
    while True:
        for name in CAMERAS:
            inp = IO_DIR / f"{name}_input.png"
            out = IO_DIR / f"{name}_overlay.png"
            try:
                if not inp.is_file():
                    continue
                stamp = inp.stat().st_mtime_ns
                if stamp == seen[name]:
                    continue
                frame = cv2.imread(str(inp), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                annotated, count, shore_ok, mask, conf = annotate_frame(frame, name)
                if SAVE_DEBUG_MASKS and mask is not None:
                    atomic_write_image(IO_DIR / f"{name}_mask.png", mask)
                if atomic_write_image(out, annotated):
                    seen[name] = stamp
                    print(f"{name.upper():5s}: shore={'yes' if shore_ok else 'no ':3s} "
                          f"conf {conf:.2f} | {count} obstacles")
                else:
                    print(f"{name.upper():5s}: output locked; retrying")
            except (OSError, cv2.error, ValueError) as exc:
                print(f"{name.upper():5s}: temporary frame issue: {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetector stopped.")