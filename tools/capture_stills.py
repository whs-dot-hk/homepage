#!/usr/bin/env python3
"""Capture meaningful video stills and keep a gallery page in timestamp order.

Auto-gen finds the important moments first. Debug is three steps. Step 1 shows
one frame per moment so the beats can be kept or dropped. Step 2, --frames with
those timestamps, shows the whole shot each kept moment sits in: one frame every
fifth of a second, earlier first, the same way a single-still picker does. A few
seconds before and after the detected cut are shown too, dimmer, so a camera
move is still a choice. The time on a card is that frame. A tenth that would point at a different picture
is written with a hundredth, so the saved frame is the one on the card. Step 3,
--final with the times copied from that textbox, saves that frame and drops
every still that was not checked. The chosen frame is the frame. A sharper
neighbor is a different picture, so it is not swapped in. A moment has to show a person.
On a page about one person, that person has to be the subject (Soyeon, when
the page is about her). A shot of someone else, or of an empty set, is not a
moment. A trailer is different: many characters share the video, so a K-drama
page keeps every held shot that has a person, and the choice page splits those
moments into a tab per character. That does not change a one-person page.
How many moments there are still comes from the video: a new shot, including
a short cut when someone is clearly in it, a pose that jumps, a closer face,
or a hand doing something, but only where a person is actually in the frame. Step 2 does not rank or drop
frames inside that shot. Auto-gen still offers a few frames, and a closed-mouth smile ranks ahead of an open mouth
or a flat mouth. It does not aim for a fixed count. Step 3 is the gallery:
each checked time is one still, and a still that was not checked is removed.
An existing jpeg of a different beat is never overwritten. The filename is the timestamp, never
a sequence number.

  tools/capture_stills.py VIDEO --at 39 --page content/kpop/toesa-halgeyo.md
  tools/capture_stills.py VIDEO --page content/kpop/toesa-halgeyo.md
  tools/capture_stills.py VIDEO --page content/kpop/toesa-halgeyo.md --dry-run
  tools/capture_stills.py VIDEO --page content/kpop/toesa-halgeyo.md --debug
  tools/capture_stills.py VIDEO --page content/kpop/toesa-halgeyo.md --frames 6,23.5,62.8
  tools/capture_stills.py VIDEO --page content/kpop/toesa-halgeyo.md --final 6,23.5,62.8

Models live in tools/models/. If one is missing, capture still works: it keeps
shot changes and the sharpest nearby frame.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YOLOX = MODEL_DIR / "yolox.onnx"
HANDS = MODEL_DIR / "hand_landmarker.task"
LIPS = MODEL_DIR / "face_landmarker.task"
# A face matches the subject when the median cosine to her gallery seeds clears
# this, and at least three seeds agree. Tuned on this video: other people land
# well below 0.40, including the narrator's close-ups once a mustache is rejected.
SUBJECT_MEDIAN = 0.40
SUBJECT_AGREE = 0.38
SUBJECT_AGREE_N = 3
SUBJECT_MIN_AREA = 0.002
PERSON_MIN_AREA = 0.02
CLOSEUP_AREA = 0.05
CLOSEUP_MEDIAN = 0.62
STACHE_REJECT = 6.0


@dataclass
class Still:
    src: str
    alt: str
    at: str

    def seconds(self) -> float:
        minutes, seconds = self.at.split(":")
        return int(minutes) * 60 + float(seconds)

    def block(self) -> str:
        return f'[[extra.stills]]\nsrc = "{self.src}"\nalt = "{self.alt}"\nat = "{self.at}"\n'


@dataclass
class Event:
    t: float
    sharp: float
    shot: int
    face_area: float = 0.0
    gesture: float = 0.0
    pose: float = 0.0
    cut: float = 0.0
    people: float = 0.0
    subject: float = 0.0
    subject_area: float = 0.0
    smile: float = 0.0
    focus: float = 0.0
    reasons: list[str] = field(default_factory=list)
    # Trailer only. (character id, face area). 0 is a face that did not join a character.
    faces: list[tuple[int, float]] = field(default_factory=list)
    cast_ids: list[int] = field(default_factory=list)
    # Step 2. The shot this strip covers, and the kept times inside it.
    shot_lo: float = 0.0
    shot_hi: float = 0.0
    kept_times: list[float] = field(default_factory=list)
    # Frames offered past the detected cut, so a camera move can still be chosen.
    nearby: bool = False


def _on_tenth(t: float) -> bool:
    """True when the time is already a tenth, so the old label still round-trips."""
    return abs(t * 10 - round(t * 10)) < 0.005


def stamp(t: float) -> str:
    """Filename-safe timestamp. A tenth stays a tenth. A frame between tenths keeps the hundredth."""
    t = max(0.0, t)
    minutes, seconds = divmod(t, 60.0)
    if _on_tenth(t):
        if abs(seconds - round(seconds)) < 0.05:
            return f"{int(minutes):02d}m{int(round(seconds)):02d}s"
        tenths = int(round(seconds * 10))
        return f"{int(minutes):02d}m{tenths // 10:02d}.{tenths % 10}s"
    hundredths = int(round(seconds * 100))
    return f"{int(minutes):02d}m{hundredths // 100:02d}.{hundredths % 100:02d}s"


def label(t: float) -> str:
    """Timestamp a person can paste. Two decimals only when one decimal would be a different frame."""
    t = max(0.0, t)
    minutes, seconds = divmod(t, 60.0)
    if not _on_tenth(t):
        return f"{int(minutes)}:{seconds:05.2f}"
    if abs(seconds - round(seconds)) < 0.05:
        return f"{int(minutes)}:{int(round(seconds)):02d}.0"
    return f"{int(minutes)}:{seconds:04.1f}"


def frame_index(fps: float, t: float) -> int:
    return int(round(max(0.0, t) * fps))


def canonical_time(fps: float, t: float) -> float:
    """Least precision that still seeks back to this same frame.

    A tenth is two or three frames. Rounding both of them to 1:23.3 saved the
    neighbor of the picture on the card. A hundredth is used only then.
    """
    index = frame_index(fps, t)
    tenth = round(index / fps, 1)
    if frame_index(fps, tenth) == index:
        return tenth
    hundredth = round(index / fps, 2)
    if frame_index(fps, hundredth) == index:
        return hundredth
    return round(index / fps, 3)


def content_box(bgr: np.ndarray, thresh: int = 8) -> tuple[int, int, int, int]:
    """Drop letterbox bars only. A dark room is the picture, not a side border."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows = gray.mean(axis=1)
    dark = rows < thresh
    top = int(dark[: h // 3].sum()) if dark[0] else 0
    bottom = int(dark[-(h // 3) :].sum()) if dark[-1] else 0
    if top < 8:
        top = 0
    if bottom < 8:
        bottom = 0
    y1 = h - bottom
    if y1 - top < h * 0.5:
        return 0, 0, w, h
    return 0, top, w, y1


def crop_bars(bgr: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = content_box(bgr)
    return bgr[y0:y1, x0:x1]


def histogram(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist / (float(np.linalg.norm(hist)) + 1e-8)


def sharpness(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    roi = gray[int(h * 0.12) : int(h * 0.88), int(w * 0.08) : int(w * 0.92)]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def open_face():
    if not YUNET.exists():
        print("no face model, shot changes only", file=sys.stderr)
        return None
    return cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)


def open_recog():
    if not SFACE.exists() or not hasattr(cv2, "FaceRecognizerSF"):
        print("no face recognizer, cannot tell the subject from other people", file=sys.stderr)
        return None
    return cv2.FaceRecognizerSF.create(str(SFACE), "")


def open_people():
    if not YOLOX.exists():
        print("no person model, a detected face still counts as a person", file=sys.stderr)
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("yolox_people", Path(__file__).resolve().parent / "yolox.py")
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YoloX(str(YOLOX), confThreshold=0.4, nmsThreshold=0.45, objThreshold=0.4)


def open_hands():
    if not HANDS.exists():
        return None
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    except ImportError:
        print("no mediapipe, gesture pass skipped", file=sys.stderr)
        return None
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HANDS)),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
        )
    )


def face_area(det, bgr: np.ndarray) -> float:
    if det is None:
        return 0.0
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return 0.0
    return max(float(f[2]) * float(f[3]) for f in faces) / float(w * h)


def gesture_score(det, bgr: np.ndarray) -> float:
    if det is None:
        return 0.0
    from mediapipe import Image, ImageFormat

    small = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res = det.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return 0.0

    def dist(a, b) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    score = 0.0
    for hand in res.hand_landmarks:
        wrist = hand[0]
        index = dist(hand[8], wrist)
        others = (dist(hand[12], wrist) + dist(hand[16], wrist) + dist(hand[20], wrist)) / 3
        score = max(score, index - others, index * 0.35)
    return float(score)


def _norm_feat(feat: np.ndarray) -> np.ndarray:
    flat = np.asarray(feat, dtype=np.float32).reshape(-1)
    return flat / (float(np.linalg.norm(flat)) + 1e-8)


def _largest_face(det, bgr: np.ndarray, score: float = 0.55):
    if det is None:
        return None
    height, width = bgr.shape[:2]
    det.setInputSize((width, height))
    det.setScoreThreshold(score)
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda face: float(face[2]) * float(face[3]))


def _face_feat(recog, bgr: np.ndarray, face) -> np.ndarray | None:
    if recog is None:
        return None
    aligned = recog.alignCrop(bgr, face)
    if aligned is None or aligned.size == 0:
        return None
    return _norm_feat(recog.feature(aligned))


def open_lips():
    """Lip landmarks. YuNet's mouth corners sit on the lip line and miss a closed smile."""
    if not LIPS.exists():
        return None
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    except ImportError:
        return None
    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LIPS)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
        )
    )


def smile_score(bgr: np.ndarray, face, lips=None) -> float:
    """How much this is a closed-mouth smile. 0 is a flat mouth, an open mouth, or no face.

    The preferred smile is lips together with the corners lifted. An open mouth,
    teeth showing, or a singing shape scores 0 even if the corners are up.
    ``face`` is unused; the lip model finds the mouth itself. It stays in the
    signature so older call sites keep working.
    """
    del face
    if lips is None:
        return 0.0
    from mediapipe import Image, ImageFormat

    small = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    found = lips.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
    if not found.face_landmarks:
        return 0.0
    marks = found.face_landmarks[0]

    def point(index: int) -> np.ndarray:
        return np.array([marks[index].x, marks[index].y], dtype=np.float32)

    width = float(np.linalg.norm(point(61) - point(291)))
    if width < 0.02:
        return 0.0
    # Gap between the inner lips, relative to how wide the mouth is.
    open_ratio = abs(float(point(13)[1] - point(14)[1])) / width
    mid_y = (float(point(13)[1]) + float(point(14)[1])) / 2.0
    # Positive when both corners sit above the lip center.
    lift = ((mid_y - float(point(61)[1])) + (mid_y - float(point(291)[1]))) / 2.0 / width
    if open_ratio > 0.16 or lift < 0.04:
        return 0.0
    closed = max(0.0, 1.0 - open_ratio / 0.16)
    raised = min(1.0, (lift - 0.04) / 0.16)
    return float(closed * (0.35 + 0.65 * raised))


def _subject_smile(recog, bgr: np.ndarray, det, seeds: np.ndarray, lips=None) -> float:
    """Smile of the subject's face in this frame. 0 if she is not the face found."""
    if det is None or recog is None or seeds.size == 0 or lips is None:
        return 0.0
    height, width = bgr.shape[:2]
    det.setInputSize((width, height))
    det.setScoreThreshold(0.45)
    _, faces = det.detect(bgr)
    if faces is None:
        return 0.0
    best = 0.0
    best_smile = 0.0
    for face in faces:
        area = float(face[2]) * float(face[3]) / float(width * height)
        if area < SUBJECT_MIN_AREA or float(face[-1]) < 0.5:
            continue
        feat = _face_feat(recog, bgr, face)
        if feat is None:
            continue
        median = float(np.median(seeds @ feat))
        if int(np.sum((seeds @ feat) >= SUBJECT_AGREE)) < SUBJECT_AGREE_N or median < SUBJECT_MEDIAN:
            continue
        if median > best:
            best = median
            best_smile = smile_score(bgr, face, lips)
    return best_smile


def _mustache_dark(bgr: np.ndarray, face) -> float:
    """How much darker the lip shadow is than the cheek. A mustache reads positive."""
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    nose_y = float(face[9])
    mouth_x = (float(face[10]) + float(face[12])) / 2
    mouth_y = (float(face[11]) + float(face[13])) / 2
    mouth_w = abs(float(face[10]) - float(face[12]))
    if mouth_w < 4 or mouth_y <= nose_y:
        return 0.0
    x0 = int(np.clip(mouth_x - mouth_w * 0.28, 0, width - 1))
    x1 = int(np.clip(mouth_x + mouth_w * 0.28, 0, width))
    y0 = int(np.clip(nose_y + (mouth_y - nose_y) * 0.15, 0, height - 1))
    y1 = int(np.clip(nose_y + (mouth_y - nose_y) * 0.72, 0, height))
    box_x, box_w = float(face[0]), float(face[2])
    cx0 = int(np.clip(box_x + box_w * 0.08, 0, width - 1))
    cx1 = int(np.clip(box_x + box_w * 0.22, 0, width))
    cy0 = int(np.clip(mouth_y - 8, 0, height - 1))
    cy1 = int(np.clip(mouth_y + 8, 0, height))
    if x1 <= x0 or y1 <= y0 or cx1 <= cx0 or cy1 <= cy0:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    cheek = gray[cy0:cy1, cx0:cx1]
    if patch.size == 0 or cheek.size == 0:
        return 0.0
    return float(cheek.mean()) - float(patch.mean())


def _people_area(det, bgr: np.ndarray) -> float:
    """Largest person box, as a fraction of the frame. 0 when nobody is detected."""
    if det is None:
        return 0.0
    small = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA)
    preds = det.infer(small)
    if preds is None or len(preds) == 0:
        return 0.0
    best = 0.0
    for row in preds:
        if int(row[5]) != 0 or float(row[4]) < 0.4:
            continue
        area = float(row[2]) * float(row[3]) / float(640 * 640)
        best = max(best, area)
    return best


# Trailer face clusters. Filled by score_cast, restored when a cast scan is loaded.
CAST_CENTROIDS: list[tuple[int, np.ndarray]] = []
CAST_BEST: dict[int, tuple[float, np.ndarray]] = {}


def page_is_trailer(page: Path) -> bool:
    """Trailers have many characters, so the one-lead filter stays off.

    K-drama, movie, and lyric-video pages are multi-character. Any page marked as
    an official trailer is too. A one-person music video is not.
    """
    text = page.read_text()
    # A lyric video has a cast, same as a trailer. A one-person music video does not.
    return (
        'category_slug = "kdrama"' in text
        or 'category_slug = "movies"' in text
        or 'video_label = "Official trailer"' in text
        or 'video_label = "Official lyric video"' in text
    )


def _has_person(event: Event) -> bool:
    """Someone is in the frame. Not limited to the page's one subject."""
    if any(area >= 0.004 for _who, area in event.faces):
        return True
    if event.face_area >= 0.008:
        return True
    return event.people >= PERSON_MIN_AREA


def _detect_faces(det, bgr: np.ndarray, score: float = 0.5, min_area: float = 0.0035) -> list[tuple[float, np.ndarray]]:
    if det is None:
        return []
    height, width = bgr.shape[:2]
    det.setInputSize((width, height))
    det.setScoreThreshold(score)
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    found = []
    for face in faces:
        area = float(face[2]) * float(face[3]) / float(width * height)
        if area < min_area:
            continue
        found.append((area, np.asarray(face, dtype=np.float32).copy()))
    found.sort(key=lambda item: -item[0])
    return found[:6]


def _box15(box) -> np.ndarray:
    row = np.zeros(15, dtype=np.float32)
    flat = np.asarray(box, dtype=np.float32).reshape(-1)
    n = min(15, int(flat.size))
    row[:n] = flat[:n]
    return row


def _crop_face(bgr: np.ndarray, box, pad: float = 0.45) -> np.ndarray:
    height, width = bgr.shape[:2]
    flat = np.asarray(box, dtype=np.float32).reshape(-1)
    if flat.size < 4:
        return bgr
    x, y, fw, fh = [float(v) for v in flat[:4]]
    if fw < 2 or fh < 2:
        return bgr
    x0 = max(0, int(x - fw * pad))
    y0 = max(0, int(y - fh * pad))
    x1 = min(width, int(x + fw * (1 + pad)))
    y1 = min(height, int(y + fh * (1 + pad)))
    if x1 <= x0 or y1 <= y0:
        return bgr
    return bgr[y0:y1, x0:x1]


def _cluster_faces(feats: list[np.ndarray], areas: list[float], match: float = 0.42) -> tuple[list[list[int]], list[np.ndarray]]:
    """Greedy clusters. Clear faces seed first, so a blurry match does not start a person."""
    clusters: list[list[int]] = []
    cents: list[np.ndarray] = []
    for i in sorted(range(len(feats)), key=lambda n: -areas[n]):
        feat = feats[i]
        best_i = -1
        best_c = match
        for ci, cent in enumerate(cents):
            score = float(np.dot(feat, cent))
            if score >= best_c:
                best_c = score
                best_i = ci
        if best_i >= 0:
            clusters[best_i].append(i)
            cents[best_i] = _norm_feat(np.mean([feats[j] for j in clusters[best_i]], axis=0))
        else:
            clusters.append([i])
            cents.append(feat)
    return clusters, cents


def _tab_ids(event: Event) -> list[int]:
    if event.cast_ids:
        return list(event.cast_ids)
    ids: list[int] = []
    for who, area in event.faces:
        if area < 0.006 or who in ids:
            continue
        ids.append(who)
    return ids


def _who_labels(event: Event) -> list[str]:
    ids = _tab_ids(event)
    if not ids:
        return []
    return ["others" if who == 0 else f"character {who}" for who in ids]


def _cast_ids_near(events: list[Event], index: int, step: float) -> list[int]:
    """Characters in this shot, not only the one face on the representative frame."""
    shot = events[index].shot
    t = events[index].t
    counts: dict[int, int] = {}
    best: dict[int, float] = {}
    for event in events:
        if event.shot != shot or abs(event.t - t) > 1.2:
            continue
        for who, area in event.faces:
            if area < 0.008:
                continue
            counts[who] = counts.get(who, 0) + 1
            best[who] = max(best.get(who, 0.0), area)
    ids = [who for who in counts if who != 0 and (counts[who] >= 2 or best[who] >= 0.018)]
    ids.sort(key=lambda who: -best[who])
    if ids:
        return ids
    if any(who == 0 and area >= 0.012 for who, area in events[index].faces):
        return [0]
    if _has_person(events[index]):
        return [0]
    return []


def _subject_seeds(page: Path, det, recog) -> np.ndarray:
    """The page's main face. Gallery stills of other people stay in the smaller cluster."""
    media, _who = media_of(page)
    feats: list[np.ndarray] = []
    for path in sorted(media.glob("*.jpg")):
        if path.name.startswith("poster"):
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        face = _largest_face(det, bgr, 0.65)
        if face is None:
            continue
        height, width = bgr.shape[:2]
        area = float(face[2]) * float(face[3]) / float(width * height)
        if area < 0.008 or float(face[-1]) < 0.7:
            continue
        feat = _face_feat(recog, bgr, face)
        if feat is not None:
            feats.append(feat)
    if len(feats) < 4:
        raise SystemExit(f"need a few clear faces of the subject in {media} to know who to keep")
    clusters: list[list[np.ndarray]] = []
    for feat in feats:
        best_i = None
        best_c = -1.0
        for index, members in enumerate(clusters):
            cent = _norm_feat(np.mean(members, axis=0))
            score = float(np.dot(feat, cent))
            if score > best_c:
                best_c = score
                best_i = index
        if best_i is not None and best_c >= 0.42:
            clusters[best_i].append(feat)
        else:
            clusters.append([feat])
    clusters.sort(key=len, reverse=True)
    seeds = clusters[0]
    if len(seeds) < 4:
        raise SystemExit("could not find a stable face for the subject in the gallery")
    # Spread the seeds so one close-up does not define her.
    step = max(1, len(seeds) // 10)
    picked = seeds[::step][:10]
    print(f"subject seeds {len(picked)} from {len(seeds)} gallery faces ({len(feats)} clear faces)", file=sys.stderr)
    return np.stack(picked)


def _match_subject(recog, bgr: np.ndarray, det, seeds: np.ndarray) -> tuple[float, float, object]:
    """Best subject face in this frame: (median cosine, face area, face). Zeros if it is not her."""
    if det is None or recog is None or seeds.size == 0:
        return 0.0, 0.0, None
    height, width = bgr.shape[:2]
    det.setInputSize((width, height))
    det.setScoreThreshold(0.45)
    _, faces = det.detect(bgr)
    if faces is None:
        return 0.0, 0.0, None
    best_median = 0.0
    best_area = 0.0
    best_face = None
    for face in faces:
        area = float(face[2]) * float(face[3]) / float(width * height)
        if area < SUBJECT_MIN_AREA or float(face[-1]) < 0.5:
            continue
        feat = _face_feat(recog, bgr, face)
        if feat is None:
            continue
        sims = seeds @ feat
        median = float(np.median(sims))
        agree = int(np.sum(sims >= SUBJECT_AGREE))
        if agree < SUBJECT_AGREE_N or median < SUBJECT_MEDIAN:
            continue
        # A big face that also has a mustache is the narrator, not her. Her own
        # close-ups are clean-shaven, so this does not throw those away.
        if area >= 0.04 and _mustache_dark(bgr, face) > STACHE_REJECT:
            continue
        if median > best_median:
            best_median = median
            best_area = area
            best_face = face
    return best_median, best_area, best_face


def is_subject(event: Event) -> bool:
    """A person is in the frame, and that person is the subject."""
    if event.subject < SUBJECT_MEDIAN or event.subject_area < SUBJECT_MIN_AREA:
        return False
    # A clear face is a person even if the body box missed. A small face still
    # needs a body, so a poster or a speck does not qualify.
    if event.subject_area >= 0.008:
        return True
    return event.people >= PERSON_MIN_AREA


def _step(events: list[Event]) -> float:
    if len(events) < 2:
        return 0.25
    return float(np.median(np.diff([event.t for event in events]))) or 0.25


def _known_times(page: Path) -> list[float]:
    """Timestamps already on the page. A first pass must not drop these beats."""
    if not page.exists():
        return []
    _, stills, _ = parse_stills(page.read_text())
    return [row.seconds() for row in stills]


def _beats(times: list[float], gap: float = 2.5) -> list[list[float]]:
    """Nearby timestamps are one beat. A pause starts the next one."""
    ordered = sorted(times)
    beats: list[list[float]] = []
    for t in ordered:
        if not beats or t - beats[-1][-1] > gap:
            beats.append([t])
        else:
            beats[-1].append(t)
    return beats


def _smooth(values: np.ndarray, width: int = 3) -> np.ndarray:
    if values.size < width or width < 2:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _rolling_median(values: np.ndarray, radius: int) -> np.ndarray:
    out = np.empty(values.shape, dtype=np.float32)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def _high_threshold(values: np.ndarray, k: float, floor: float) -> float:
    """A jump has to clear the noise of this video, and an absolute floor."""
    if values.size == 0:
        return floor
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(floor, med + k * 1.4826 * mad)


def _local_peaks(values: np.ndarray, thr: float) -> list[int]:
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] < thr:
            continue
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    return peaks


def _shot_spans(events: list[Event]) -> list[tuple[int, int]]:
    """Spans from the histogram shot id. A new id is a new picture, not a quota."""
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(events)):
        if events[i].shot != events[start].shot:
            spans.append((start, i))
            start = i
    spans.append((start, len(events)))
    return spans


def _subject_runs(events: list[Event], start: int, end: int, step: float, visible=is_subject) -> list[tuple[int, int]]:
    """Stretches where a person stays visible. A gap is a different beat, not one moment."""
    hits = [i for i in range(start, end) if visible(events[i])]
    if not hits:
        return []
    gap = max(1, int(round(2.5 / step)))
    runs: list[tuple[int, int]] = []
    run_start = hits[0]
    prev = hits[0]
    for index in hits[1:]:
        if index - prev > gap:
            runs.append((run_start, prev + 1))
            run_start = index
        prev = index
    runs.append((run_start, prev + 1))
    return runs


def _shot_frame(events: list[Event], start: int, end: int, step: float, visible=is_subject, rank=None) -> int | None:
    """The clearest frame of a person in this shot, not the blurred cut."""
    good = [i for i in range(start, end) if visible(events[i])]
    if not good:
        return None
    settle = max(1, int(round(0.3 / step)))
    settled = [i for i in good if i >= start + settle] or good
    key = rank or (lambda i: (events[i].subject_area >= 0.006, events[i].subject, events[i].subject_area, events[i].sharp))
    return max(settled, key=key)


def moments_of(events: list[Event], limit: int | None = None, known: list[float] | None = None, cast: bool = False) -> list[Event]:
    """How many important moments this video has.

    A moment is a shot that shows a person and can be a picture. On a one-person
    page that person is the subject. On a trailer, any held shot with a person
    counts, and the choice page tabs those moments by character. A cut under a
    second still counts when someone is in it. One blurry sample is a flash.
    A jump, a closer face, or a gesture later in that shot can be another moment.
    Shots with nobody in them are not moments. Nearby hits of the same beat fold
    into one. A gallery timestamp is its own beat unless the scan already has
    that same beat. Nothing is added to reach a count. `limit` is only an optional ceiling.
    """
    if not events:
        return []
    step = _step(events)

    def visible(event: Event) -> bool:
        return _has_person(event) if cast else is_subject(event)

    def face_size(event: Event) -> float:
        if cast:
            if event.faces:
                return max(area for _who, area in event.faces)
            return event.face_area
        return event.subject_area

    rank = None
    if cast:
        rank = lambda i: (face_size(events[i]) >= 0.006, face_size(events[i]), events[i].sharp)
    hold_for = 1.0
    spans = _shot_spans(events)
    # index, strength, reasons. Strength is only for folding and an optional ceiling.
    cands: list[tuple[int, float, set[str]]] = []
    skipped = 0
    for start, end in spans:
        hold = (end - start) * step
        runs = _subject_runs(events, start, end, step, visible)
        if not runs:
            if hold >= hold_for:
                skipped += 1
            continue
        # One sample is a flash. A shorter cut is still a moment when a person is
        # in the picture: 1:22 is 0.75s, 1:33 is 0.5s. Both were dropped by a 1s hold.
        if hold < 0.45:
            continue
        # Each stretch where a person stays visible is its own beat. A wide shot
        # or a turn away still counts, as long as someone is in that stretch.
        for run_start, run_end in runs:
            rep = _shot_frame(events, run_start, run_end, step, visible, rank)
            if rep is None:
                continue
            hits = run_end - run_start
            face_max = max(face_size(events[i]) for i in range(run_start, run_end))
            if hold < hold_for:
                event = events[rep]
                if cast:
                    # A one-sample blur at the cut is the transition, not the beat.
                    if hits < 2 and event.cut >= 0.04 and event.sharp < 12 and face_size(event) < 0.02:
                        continue
                else:
                    # The cut frame itself, one soft sample, is the transition, not the beat.
                    cut_flash = hits < 2 and event.cut >= 0.04 and event.sharp < 12
                    stays = hits >= 2 and face_max >= 0.01
                    picture = face_max >= 0.04 and not cut_flash
                    if not stays and not picture:
                        continue
            elif hits < 2 and hold > 2.0 and (face_size(events[rep]) < 0.01 if cast else events[rep].subject < 0.50):
                continue
            cands.append((rep, 2.0 + face_max * 10 + (face_size(events[rep]) if cast else events[rep].subject), {"shot"}))

    taken = {index for index, _, _ in cands}
    radius = max(2, int(round(1.5 / step)))
    pose = np.array([event.pose for event in events], dtype=np.float32)
    pose_jump = pose - _rolling_median(pose, radius)
    pose_thr = _high_threshold(pose_jump[1:], 3.5, 20.0)
    span_of = {i: (start, end) for start, end in spans for i in range(start, end)}
    for i in _local_peaks(pose_jump, pose_thr):
        if pose[i] < 24 or i not in span_of or not visible(events[i]):
            continue
        start, end = span_of[i]
        if (end - start) * step < 3.0:
            continue
        if any(abs(events[i].t - events[have].t) < 2.0 for have in taken):
            continue
        cands.append((i, float(pose_jump[i]) / 18.0, {"pose"}))
        taken.add(i)

    face = _smooth(np.array([face_size(event) for event in events], dtype=np.float32))
    face_jump = face - _rolling_median(face, radius)
    face_thr = _high_threshold(face_jump[1:], 3.5, 0.04)
    for i in _local_peaks(face_jump, face_thr):
        if face[i] < 0.02 or not visible(events[i]):
            continue
        if any(abs(events[i].t - events[have].t) < 2.0 for have in taken):
            continue
        cands.append((i, float(face_jump[i]) * 20, {"face"}))
        taken.add(i)

    hand = np.array([event.gesture for event in events], dtype=np.float32)
    hand_thr = 0.0
    if float(hand.max()) > 0:
        hand_s = _smooth(hand)
        hand_jump = hand_s - _rolling_median(hand_s, radius)
        hand_thr = _high_threshold(hand_jump[1:], 3.5, 0.04)
        for i in _local_peaks(hand_jump, hand_thr):
            if not visible(events[i]):
                continue
            if any(abs(events[i].t - events[have].t) < 2.0 for have in taken):
                continue
            cands.append((i, float(hand_jump[i]) * 8, {"gesture"}))
            taken.add(i)

    folded = _fold_beats(cands, events)
    # Gallery times the shot pass did not already keep. Each timestamp is its
    # own beat: 1:22 is not 1:23, and 1:33 is not 1:32. The old 2.5s window
    # folded those into the neighbor and the first pass never showed them.
    covered = [events[index].t for index, _, _ in folded]
    for asked in known or []:
        if any(abs(asked - have) < 0.9 for have in covered):
            continue
        index = min(range(len(events)), key=lambda i: abs(events[i].t - asked))
        near = [i for i, event in enumerate(events) if abs(event.t - asked) <= 0.5 and visible(event)]
        if near:
            her = min(near, key=lambda i: abs(events[i].t - asked))
            # Only move onto her frame when that does not land on a beat already kept.
            if not any(abs(events[her].t - have) < 0.9 for have in covered):
                index = her
        folded.append((index, 3.0, {"marked"}))
        covered.append(events[index].t)
    folded = _fold_beats([(index, strength, reasons) for index, strength, reasons in folded], events)
    identified = len(folded)
    if limit is not None and identified > limit:
        folded = sorted(folded, key=lambda item: -item[1])[:limit]
        print(f"moment ceiling {limit}, pipeline identified {identified}", file=sys.stderr)
    folded.sort(key=lambda item: events[item[0]].t)
    n_shot = sum("shot" in reasons for _, _, reasons in folded)
    n_pose = sum("pose" in reasons for _, _, reasons in folded)
    n_face = sum("face" in reasons for _, _, reasons in folded)
    n_hand = sum("gesture" in reasons for _, _, reasons in folded)

    found: list[Event] = []
    for index, _strength, reasons in folded:
        event = events[index]
        event.reasons = ["moment"] + [name for name in ("shot", "pose", "face", "gesture", "marked") if name in reasons]
        if cast:
            event.cast_ids = _cast_ids_near(events, index, step)
            event.reasons.extend(_who_labels(event))
        elif is_subject(event):
            event.reasons.append("soyeon")
        found.append(event)
    dropped = "nobody is in them" if cast else "Soyeon is not the person in them"
    print(
        f"moment pipeline: {len(found)} from the video "
        f"({n_shot} held shots, {n_pose} jumps, {n_face} close-ups, {n_hand} gestures; "
        f"{skipped} held shots dropped because {dropped}; "
        f"pose>={pose_thr:.1f} face>={face_thr:.3f} hand>={hand_thr:.3f})",
        file=sys.stderr,
    )
    return found


def _fold_beats(cands: list[tuple[int, float, set[str]]], events: list[Event]) -> list[tuple[int, float, set[str]]]:
    """One moment per beat. A jump on a cut stays with the shot frame."""
    if not cands:
        return []
    ordered = sorted(cands, key=lambda item: (events[item[0]].t, -item[1]))
    kept: list[tuple[int, float, set[str]]] = []
    for index, strength, reasons in ordered:
        if not kept:
            kept.append((index, strength, set(reasons)))
            continue
        prev_i, prev_s, prev_r = kept[-1]
        gap = events[index].t - events[prev_i].t
        if gap < 0.9:
            merged = prev_r | reasons
            prefer_prev = "shot" in prev_r and "shot" not in reasons
            prefer_new = "shot" in reasons and "shot" not in prev_r
            if prefer_prev or (not prefer_new and strength <= prev_s):
                kept[-1] = (prev_i, max(prev_s, strength), merged)
            else:
                kept[-1] = (index, max(prev_s, strength), merged)
            continue
        kept.append((index, strength, set(reasons)))
    return kept


def choices_around(events: list[Event], moments: list[Event], per: int, video: Path | None = None, page: Path | None = None, cast: bool = False) -> list[Event]:
    """A few different frames around each moment, for a person to choose from.

    The moment itself is one choice. The others come from a fine pass around that
    beat, not only the coarse scan. A closed-mouth smile ranks ahead of an open
    mouth or a flat mouth, then a closer face, a jump, or a hand. On a one-person
    page, a frame of someone else is not offered. A trailer offers any person.
    """
    per = max(1, per)
    picks: list[Event] = []
    seen: set[int] = set()
    fine = _fine_around(video, page, moments, cast) if video is not None and page is not None else {}
    tag = "cast" if cast else "soyeon"

    def on_screen(event: Event) -> bool:
        return _has_person(event) if cast else is_subject(event)

    for nth, moment in enumerate(moments):
        left = moments[nth - 1].t if nth else events[0].t
        right = moments[nth + 1].t if nth + 1 < len(moments) else events[-1].t
        # Wide enough to catch the smile just before a blur, not only the peak.
        lo = moment.t - min(2.4, max(1.2, (moment.t - left) * 0.45))
        hi = moment.t + min(2.4, max(1.2, (right - moment.t) * 0.45))
        window = [event for event in events if lo <= event.t <= hi and on_screen(event)]
        window.extend(fine.get(id(moment), []))
        if not window:
            # The beat was kept even though this sample missed a face. Offer the
            # moment itself, not a later shot where the match happened to succeed.
            window = [moment]
        ranked = [moment]
        # A choice has to be in focus. A soft face is not offered when the beat
        # has a sharper one. The moment sample is replaced when it is the blur.
        peers = [event for event in window if event.focus > 0 or event.sharp > 0]
        best_focus = max((_focus_of(event) for event in peers), default=0.0)

        def usable(event: Event) -> bool:
            score = _focus_of(event)
            if best_focus >= 28 and score < best_focus * 0.55:
                return False
            close = event.face_area if cast else event.subject_area
            if best_focus >= 28 and score < 22 and close >= 0.01:
                return False
            return event.sharp >= 8 or score >= 22

        if not usable(moment):
            stand_ins = [event for event in window if usable(event) and abs(event.t - moment.t) <= 0.8]
            if stand_ins:
                stand = min(stand_ins, key=lambda event: (abs(event.t - moment.t), -_focus_of(event)))
                stand.reasons = ["choice", tag, "focus"]
                ranked = [stand]
        floor = max(40.0, ranked[0].sharp * 0.55)
        rest = []
        for event in window:
            if abs(event.t - ranked[0].t) < 0.05 or event.sharp < floor or not usable(event):
                continue
            if cast:
                if not (event.faces or event.face_area >= 0.004 or event.people >= PERSON_MIN_AREA):
                    continue
            elif event.subject_area < SUBJECT_MIN_AREA:
                continue
            rest.append(event)
        rest.sort(key=_choice_rank)
        for event in rest:
            if len(ranked) >= per:
                break
            if any(abs(event.t - have.t) < 0.35 for have in ranked):
                continue
            event.reasons = ["choice", tag]
            if event.smile >= 0.12:
                event.reasons.append("smile")
            if event.focus >= 28:
                event.reasons.append("focus")
            if event.pose >= 20:
                event.reasons.append("pose")
            if (event.face_area if cast else event.subject_area) >= 0.02:
                event.reasons.append("face")
            if event.gesture >= 0.04:
                event.reasons.append("gesture")
            ranked.append(event)
        for event in sorted(ranked, key=lambda item: item.t):
            key = int(round(event.t * 10))
            if key in seen:
                continue
            seen.add(key)
            if cast:
                for label in _who_labels(event):
                    if label not in event.reasons:
                        event.reasons.append(label)
            elif tag not in event.reasons:
                event.reasons.append(tag)
            picks.append(event)
    return picks


def _match_cast(feat: np.ndarray) -> int:
    """Nearest character tab for this face. 0 when it does not match one."""
    best_who = 0
    best_c = 0.42
    for who, cent in CAST_CENTROIDS:
        score = float(np.dot(feat, cent))
        if score >= best_c:
            best_c = score
            best_who = who
    return best_who


def _fine_cast(video: Path, moments: list[Event]) -> dict[int, list[Event]]:
    """Frames around a trailer moment, tagged with whichever character is in them."""
    face = open_face()
    recog = open_recog()
    if face is None or recog is None or not moments:
        return {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(0.12 * fps)))
    found: dict[int, list[Event]] = {}
    for moment in moments:
        lo = max(0.0, moment.t - 2.2)
        hi = moment.t + 2.2
        start = int(round(lo * fps))
        end = int(round(hi * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        bucket: list[Event] = []
        index = start
        while index <= end:
            ok, bgr = cap.read()
            if not ok:
                break
            if (index - start) % stride == 0:
                hits = _detect_faces(face, bgr)
                faces: list[tuple[int, float]] = []
                for area, box in hits:
                    feat = _face_feat(recog, bgr, box)
                    who = _match_cast(feat) if feat is not None else 0
                    if who or area >= 0.008:
                        faces.append((who, area))
                event = Event(
                    t=index / fps,
                    sharp=sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                    shot=moment.shot,
                    face_area=hits[0][0] if hits else 0.0,
                    faces=faces,
                    focus=person_focus(bgr, face)[0] if hits else 0.0,
                )
                if _has_person(event):
                    bucket.append(event)
            index += 1
        found[id(moment)] = bucket
        print(f"fine {label(moment.t)} {len(bucket)} frames", file=sys.stderr, flush=True)
    cap.release()
    return found


def _fine_around(video: Path, page: Path, moments: list[Event], cast: bool = False) -> dict[int, list[Event]]:
    """Every few frames around each kept moment, scored for who and for a smile.

    The coarse scan steps by a quarter second and skips the smile. This pass
    only covers the beats already kept. A trailer matches faces to the character
    tabs instead of the one subject.
    """
    if cast:
        return _fine_cast(video, moments)
    face = open_face()
    recog = open_recog()
    lips = open_lips()
    if face is None or recog is None or not moments:
        return {}
    seeds = _subject_seeds(page, face, recog)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(0.12 * fps)))
    found: dict[int, list[Event]] = {}
    for moment in moments:
        lo = max(0.0, moment.t - 2.2)
        hi = moment.t + 2.2
        start = int(round(lo * fps))
        end = int(round(hi * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        bucket: list[Event] = []
        index = start
        while index <= end:
            ok, bgr = cap.read()
            if not ok:
                break
            if (index - start) % stride == 0:
                subject, area, face_box = _match_subject(recog, bgr, face, seeds)
                event = Event(
                    t=index / fps,
                    sharp=sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                    shot=moment.shot,
                    subject=subject,
                    subject_area=area,
                    smile=smile_score(bgr, face_box, lips) if face_box is not None else 0.0,
                    focus=person_focus(bgr, face)[0] if face_box is not None else 0.0,
                )
                if is_subject(event):
                    bucket.append(event)
            index += 1
        found[id(moment)] = bucket
        print(f"fine {label(moment.t)} {len(bucket)} frames of her", file=sys.stderr, flush=True)
    cap.release()
    return found


def _focus_of(event: Event) -> float:
    """Face focus when the fine pass measured it, otherwise the whole-frame score."""
    return event.focus if event.focus > 0 else event.sharp


def _choice_rank(event: Event) -> float:
    """Lower is better. A closed-mouth smile outranks an open mouth or a flat one.

    Soft frames are already dropped before this rank. A sharp face still beats a soft smile.
    """
    smile = 0.0 if event.smile >= 0.22 else 1.0 if event.smile >= 0.08 else 2.0
    return smile - min(event.smile, 0.6) * 4.0 - event.subject_area * 2.0 - _focus_of(event) / 4000.0


def scene_vec(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    scene = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return scene / (float(np.linalg.norm(scene)) + 1e-8)


def person_patch(bgr: np.ndarray) -> np.ndarray:
    """The center of the frame, where the dancer usually is."""
    small = cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    patch = gray[int(height * 0.08) : int(height * 0.92), int(width * 0.30) : int(width * 0.70)]
    return patch.astype(np.float32)


def scan_cache_path(video: Path, step: float, cast: bool = False) -> Path:
    stat = video.stat()
    tag = "cast2" if cast else "who6"
    key = f"{video.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{step:.3f}|{tag}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return Path("/tmp/stills-scan") / f"{digest}.npz"


def events_from_cache(path: Path) -> list[Event] | None:
    if not path.exists():
        return None
    data = np.load(path)
    if "subject" not in data.files or "people" not in data.files:
        return None
    events = []
    rows = zip(
        data["t"], data["sharp"], data["shot"], data["face"], data["gesture"],
        data["pose"], data["cut"], data["people"], data["subject"], data["subject_area"],
        data["smile"] if "smile" in data.files else np.zeros(len(data["t"])),
    )
    for row in rows:
        events.append(Event(
            t=float(row[0]),
            sharp=float(row[1]),
            shot=int(row[2]),
            face_area=float(row[3]),
            gesture=float(row[4]),
            pose=float(row[5]),
            cut=float(row[6]),
            people=float(row[7]),
            subject=float(row[8]),
            subject_area=float(row[9]),
            smile=float(row[10]),
        ))
    if not events:
        return None
    if "face_who" in data.files and "face_area_tab" in data.files:
        for i, event in enumerate(events):
            faces = []
            for who, area in zip(data["face_who"][i], data["face_area_tab"][i]):
                if int(who) >= 0 and float(area) > 0:
                    faces.append((int(who), float(area)))
            event.faces = faces
    if "cast_cent" in data.files and "cast_id" in data.files:
        global CAST_CENTROIDS, CAST_BEST
        CAST_CENTROIDS = [
            (int(who), np.asarray(cent, dtype=np.float32))
            for who, cent in zip(data["cast_id"], data["cast_cent"])
        ]
        CAST_BEST = {}
        if "cast_best_t" in data.files and "cast_best_box" in data.files:
            for who, t, box in zip(data["cast_id"], data["cast_best_t"], data["cast_best_box"]):
                CAST_BEST[int(who)] = (float(t), np.asarray(box, dtype=np.float32))
    return events


def load_base_scan(video: Path, step: float) -> list[Event] | None:
    """Shot scan from before identity was scored. Same samples, no subject yet."""
    stat = video.stat()
    base = f"{video.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{step:.3f}"
    names = [
        hashlib.sha1(base.encode()).hexdigest()[:16],
        hashlib.sha1(f"{base}|who6".encode()).hexdigest()[:16],
    ]
    for name in names:
        path = Path("/tmp/stills-scan") / f"{name}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if "t" not in data.files or "shot" not in data.files:
            continue
        events = []
        rows = zip(
            data["t"], data["sharp"], data["shot"], data["face"], data["gesture"], data["pose"], data["cut"],
        )
        for row in rows:
            events.append(Event(
                t=float(row[0]),
                sharp=float(row[1]),
                shot=int(row[2]),
                face_area=float(row[3]),
                gesture=float(row[4]),
                pose=float(row[5]),
                cut=float(row[6]),
            ))
        if events:
            return events
    return None


def save_scan(path: Path, events: list[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    who = np.full((len(events), 6), -1, dtype=np.int16)
    tab_area = np.zeros((len(events), 6), dtype=np.float32)
    for i, event in enumerate(events):
        for j, (face_who, area) in enumerate(event.faces[:6]):
            who[i, j] = face_who
            tab_area[i, j] = area
    extra = {
        "face_who": who,
        "face_area_tab": tab_area,
    }
    if CAST_CENTROIDS:
        extra["cast_id"] = np.array([face_who for face_who, _cent in CAST_CENTROIDS], dtype=np.int16)
        extra["cast_cent"] = np.stack([cent for _who, cent in CAST_CENTROIDS]).astype(np.float32)
        extra["cast_best_t"] = np.array([CAST_BEST[face_who][0] for face_who, _cent in CAST_CENTROIDS], dtype=np.float32)
        extra["cast_best_box"] = np.stack([CAST_BEST[face_who][1] for face_who, _cent in CAST_CENTROIDS]).astype(np.float32)
    np.savez(
        path,
        t=np.array([event.t for event in events], dtype=np.float32),
        sharp=np.array([event.sharp for event in events], dtype=np.float32),
        shot=np.array([event.shot for event in events], dtype=np.int32),
        face=np.array([event.face_area for event in events], dtype=np.float32),
        gesture=np.array([event.gesture for event in events], dtype=np.float32),
        pose=np.array([event.pose for event in events], dtype=np.float32),
        cut=np.array([event.cut for event in events], dtype=np.float32),
        people=np.array([event.people for event in events], dtype=np.float32),
        subject=np.array([event.subject for event in events], dtype=np.float32),
        subject_area=np.array([event.subject_area for event in events], dtype=np.float32),
        smile=np.array([event.smile for event in events], dtype=np.float32),
        **extra,
    )


def score_cast(video: Path, events: list[Event], step: float) -> None:
    """Cluster faces in a trailer. Each recurring person gets a tab id, starting at 1."""
    global CAST_CENTROIDS, CAST_BEST
    CAST_CENTROIDS = []
    CAST_BEST = {}
    face = open_face()
    recog = open_recog()
    people = open_people()
    if face is None or recog is None:
        print("cast check skipped, moments are not split by character", file=sys.stderr)
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(step * fps)))
    samples: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []
    index = 0
    sample = 0
    while sample < len(events):
        ok, bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            event = events[sample]
            event.people = _people_area(people, bgr)
            for area, box in _detect_faces(face, bgr):
                feat = _face_feat(recog, bgr, box)
                if feat is None:
                    continue
                samples.append((sample, event.t, area, feat, box))
            sample += 1
            if sample % 40 == 0:
                print(f"cast {event.t:.0f}s", file=sys.stderr, flush=True)
        index += 1
    cap.release()
    if sample != len(events):
        raise SystemExit(f"cast pass saw {sample} frames, scan had {len(events)}")
    if not samples:
        print("cast: no faces", file=sys.stderr)
        return
    clusters, cents = _cluster_faces([row[3] for row in samples], [row[2] for row in samples])
    ranked = []
    for members, cent in zip(clusters, cents):
        count = len(members)
        total = sum(samples[i][2] for i in members)
        # A couple of frames is a flash, not a character tab. Those faces stay in Others.
        if count >= 4:
            ranked.append((total, members, cent))
    ranked.sort(key=lambda item: -item[0])
    ranked = ranked[:8]
    who_of: dict[int, int] = {}
    for nth, (_total, members, cent) in enumerate(ranked, start=1):
        CAST_CENTROIDS.append((nth, cent))
        best = max(members, key=lambda i: samples[i][2])
        CAST_BEST[nth] = (samples[best][1], _box15(samples[best][4]))
        for i in members:
            who_of[i] = nth
    by_event: dict[int, dict[int, float]] = {}
    for i, row in enumerate(samples):
        who = who_of.get(i, 0)
        if who == 0 and row[2] < 0.008:
            continue
        bucket = by_event.setdefault(row[0], {})
        bucket[who] = max(bucket.get(who, 0.0), row[2])
    for event_i, bucket in by_event.items():
        events[event_i].faces = sorted(bucket.items(), key=lambda item: -item[1])
    counts = ", ".join(str(len(members)) for _total, members, _cent in ranked)
    print(f"cast: {len(CAST_CENTROIDS)} characters ({counts})", file=sys.stderr)


def score_subject(video: Path, page: Path, events: list[Event], step: float) -> None:
    """Mark each sample that shows a person, and whether that person is the subject."""
    face = open_face()
    recog = open_recog()
    people = open_people()
    if face is None or recog is None:
        print("subject check skipped, frames are not filtered to Soyeon", file=sys.stderr)
        return
    seeds = _subject_seeds(page, face, recog)
    lips = open_lips()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(step * fps)))
    index = 0
    sample = 0
    while sample < len(events):
        ok, bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            event = events[sample]
            event.people = _people_area(people, bgr)
            matched = _match_subject(recog, bgr, face, seeds)
            event.subject, event.subject_area = matched[0], matched[1]
            event.smile = smile_score(bgr, matched[2], lips) if matched[2] is not None else 0.0
            sample += 1
            if sample % 40 == 0:
                print(f"subject {event.t:.0f}s", file=sys.stderr, flush=True)
        index += 1
    cap.release()
    if sample != len(events):
        raise SystemExit(f"subject pass saw {sample} frames, scan had {len(events)}")
    kept = sum(1 for event in events if is_subject(event))
    print(f"subject visible in {kept}/{len(events)} samples", file=sys.stderr)


def scan(video: Path, page: Path, step: float) -> list[Event]:
    cast = page_is_trailer(page)
    cached = scan_cache_path(video, step, cast)
    loaded = events_from_cache(cached)
    if loaded:
        print(f"scan cache {cached} ({len(loaded)} samples)", file=sys.stderr)
        return loaded
    events = load_base_scan(video, step)
    if events:
        print(f"shot cache reused ({len(events)} samples), scoring who is in frame", file=sys.stderr)
    else:
        events = _scan_shots(video, step)
    if cast:
        score_cast(video, events, step)
    else:
        score_subject(video, page, events, step)
    save_scan(cached, events)
    return events


def _scan_shots(video: Path, step: float) -> list[Event]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(step * fps)))
    face = open_face()
    hands = open_hands()
    events: list[Event] = []
    index = 0
    shot = 0
    prev = None
    prev_scene = None
    prev_person = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            t = index / fps
            current = histogram(bgr)
            scene = scene_vec(bgr)
            person = person_patch(bgr)
            cut = 0.0
            pose = 0.0
            if prev is not None and float(np.dot(prev, current)) < 0.82:
                shot += 1
            if prev_scene is not None:
                cut = 1.0 - float(np.dot(prev_scene, scene))
            if prev_person is not None:
                pose = float(np.mean(np.abs(person - prev_person)))
            prev = current
            prev_scene = scene
            prev_person = person
            events.append(Event(
                t=t,
                sharp=sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                shot=shot,
                face_area=face_area(face, bgr),
                gesture=gesture_score(hands, bgr),
                pose=pose,
                cut=cut,
            ))
            if len(events) % 40 == 0:
                print(f"scanned {t:.0f}s", file=sys.stderr, flush=True)
        index += 1
        if total and index >= total:
            break
    cap.release()
    if not events:
        raise SystemExit(f"no frames in {video}")
    return events


def grab(video: Path, t: float, window: float) -> tuple[float, np.ndarray]:
    """Sharpest decoded frame within +/- window, staying inside the asked second."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    lo = max(0.0, t - window, float(int(t)))
    hi = min(t + window, float(int(t) + 1) - 1.0 / fps)
    start = int(round(lo * fps))
    end = int(round(hi * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    best_t = t
    best_img = None
    best_sharp = -1.0
    index = start
    while index <= end:
        ok, bgr = cap.read()
        if not ok:
            break
        score = sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if score > best_sharp:
            best_sharp = score
            best_t = index / fps
            best_img = bgr
        index += 1
    cap.release()
    if best_img is None:
        raise SystemExit(f"no frame near {t:.1f}s")
    return best_t, best_img


def _fps_of(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return fps


def grab_at(video: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame at {t:.2f}s")
    return bgr


def _same_shot(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two frames are the same cut, not a nearby cut that happens to be sharp."""
    return float(np.dot(histogram(a), histogram(b))) >= 0.82 and float(np.dot(scene_vec(a), scene_vec(b))) >= 0.90


def _lap(gray: np.ndarray) -> float:
    if gray.size < 16:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def person_focus(bgr: np.ndarray, det=None) -> tuple[float, float]:
    """How sharp the person is, and how much of the frame the face covers.

    A smooth wall makes the whole frame look soft even when the face is sharp.
    Motion blur does the opposite: the face score collapses. Focus is the face
    when one is found, otherwise the head region.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    focus = 0.0
    area = 0.0
    if det is not None:
        det.setInputSize((width, height))
        det.setScoreThreshold(0.45)
        _, faces = det.detect(bgr)
        if faces is not None:
            for face in faces:
                fw, fh = float(face[2]), float(face[3])
                found = fw * fh / float(width * height)
                if float(face[-1]) < 0.5 or found < 0.003 or found <= area:
                    continue
                x, y = float(face[0]), float(face[1])
                x0 = max(0, int(x - fw * 0.08))
                y0 = max(0, int(y - fh * 0.12))
                x1 = min(width, int(x + fw * 1.08))
                y1 = min(height, int(y + fh * 1.12))
                area = found
                focus = _lap(gray[y0:y1, x0:x1])
    if area >= 0.003:
        return focus, area
    head = gray[int(height * 0.08) : int(height * 0.62), int(width * 0.28) : int(width * 0.72)]
    return _lap(head), 0.0


def _focus_ok(focus: float, area: float) -> bool:
    """A face large enough to judge, and sharp enough to keep as a still."""
    if area >= 0.03:
        return focus >= 36.0
    if area >= 0.003:
        return focus >= 28.0
    return focus >= 40.0


def _shot_run(frames: list[tuple[int, np.ndarray, float, float]], asked_i: int) -> list[int]:
    """Indexes in ``frames`` that are the same cut as the asked frame.

    A cut is a jump between consecutive frames. Comparing a blur to a sharp
    frame half a second away is not a cut test — blur washes the histogram.
    """
    if not frames:
        return []
    pos = next(i for i, row in enumerate(frames) if row[0] == asked_i)
    keep = {pos}

    def blocked(left: int, right: int) -> bool:
        a = frames[left]
        b = frames[right]
        if a[3] >= 0.02 and b[3] < 0.004:
            return True
        if b[3] >= 0.02 and a[3] < 0.004:
            return True
        hist = float(np.dot(histogram(a[1]), histogram(b[1])))
        scene = float(np.dot(scene_vec(a[1]), scene_vec(b[1])))
        return hist < 0.62 or scene < 0.84

    for i in range(pos, 0, -1):
        if blocked(i - 1, i):
            break
        keep.add(i - 1)
    for i in range(pos, len(frames) - 1):
        if blocked(i, i + 1):
            break
        keep.add(i + 1)
    return sorted(keep)


def _clearer(row: tuple[int, np.ndarray, float, float], asked_focus: float, asked_whole: float) -> bool:
    """True only when her face and the whole frame are both clearly sharper.

    A white wall makes a sharp face score soft, and a center crop with no face
    can score a blur as the sharp frame. Both have to improve, or the pick stays.
    """
    _index, bgr, focus, area = row
    if area < 0.004 or focus < 24 or focus < asked_focus * 1.8:
        return False
    whole = float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return whole >= asked_whole * 1.25


def grab_final(video: Path, t: float, window: float = 0.5) -> tuple[float, np.ndarray]:
    """The chosen timestamp, or the nearest in-focus frame of that same shot.

    Step 3 does not hunt a new moment. It starts at the time that was picked.
    A sharp pick stays put. A soft pick is replaced only when her face is found
    and a nearby frame of the same shot is clearly sharper on both her face and
    the whole frame. No face means the chosen frame stays — a center crop is not
    proof of blur. A different cut is never used. If the whole shot is motion
    blur, the chosen frame is what gets saved.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    asked = int(round(t * fps))
    radius = max(3, int(round(window * fps)))
    start = max(0, asked - radius)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    det = open_face()
    frames: list[tuple[int, np.ndarray, float, float]] = []
    index = start
    while index <= asked + radius:
        ok, bgr = cap.read()
        if not ok:
            break
        focus, area = person_focus(bgr, det)
        frames.append((index, bgr, focus, area))
        index += 1
    cap.release()
    if not frames:
        raise SystemExit(f"no frame at {t:.2f}s")
    asked_row = min(frames, key=lambda row: abs(row[0] - asked))
    asked_t = canonical_time(fps, asked_row[0] / fps)
    if asked_row[3] < 0.004 or _focus_ok(asked_row[2], asked_row[3]):
        return asked_t, asked_row[1]
    run = [frames[i] for i in _shot_run(frames, asked_row[0])]
    asked_whole = float(cv2.Laplacian(cv2.cvtColor(asked_row[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    # A chosen close-up must stay a close-up. The next cut can be sharp and
    # faceless; histogram sometimes misses that jump when the blur is washed out.
    faced = [row for row in run if row[3] >= 0.004 and row[0] != asked_row[0]]
    clearer = [row for row in faced if _clearer(row, asked_row[2], asked_whole)]
    if not clearer:
        return asked_t, asked_row[1]
    clearer.sort(key=lambda row: (abs(row[0] - asked), -row[2]))
    return canonical_time(fps, clearer[0][0] / fps), clearer[0][1]


def media_of(page: Path) -> tuple[Path, str]:
    text = page.read_text()
    found = re.search(r'poster = "/media/([^/]+)/', text)
    if not found:
        raise SystemExit(f"{page} has no poster under /media/")
    slug = found.group(1)
    korean = re.search(r'korean = "([^"]*)"', text)
    who = korean.group(1) if korean else page.stem
    return ROOT / "themes" / "whs" / "static" / "media" / slug, who


def parse_stills(text: str) -> tuple[str, list[Still], str]:
    """Split a page into the lead, the stills, and the closing front matter."""
    marker = "[[extra.stills]]"
    if marker not in text:
        close = text.rfind("\n+++")
        if close == -1:
            raise SystemExit(f"page has no front matter to attach stills to")
        return text[: close + 1], [], text[close + 1 :]
    parts = text.split(marker)
    head = parts[0]
    stills: list[Still] = []
    tail = ""
    last = len(parts) - 2
    for i, chunk in enumerate(parts[1:]):
        body = chunk
        if i == last:
            close = re.search(r"\n\+\+\+", chunk)
            if close:
                body = chunk[: close.start()]
                tail = chunk[close.start() + 1 :]
        src = re.search(r'src = "([^"]+)"', body)
        alt = re.search(r'alt = "([^"]*)"', body)
        at = re.search(r'at = "([^"]+)"', body)
        if not (src and alt and at):
            raise SystemExit("a still is missing src, alt, or at")
        stills.append(Still(src.group(1), alt.group(1), at.group(1)))
    if not tail.startswith("+++"):
        raise SystemExit("refusing to write a page that would lose its closing front matter")
    return head, stills, tail


def alt_for(stills: list[Still], who: str, at: str) -> str:
    if stills and " at " in stills[0].alt:
        return stills[0].alt.rsplit(" at ", 1)[0] + f" at {at}"
    return f"{who} at {at}"


def write_page(page: Path, head: str, stills: list[Still], tail: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = re.sub(r"^updated = .*$", f"updated = {now}", head, count=1, flags=re.M)
    body = "\n".join(still.block() for still in stills)
    page.write_text(head + body + tail)


def add_new(stills: list[Still], still: Still) -> tuple[list[Still], bool]:
    """Append a still. An existing src stays; nothing is removed."""
    if any(row.src == still.src for row in stills):
        return stills, False
    kept = stills + [still]
    kept.sort(key=lambda row: (row.seconds(), row.src))
    return kept, True


def save_jpeg(path: Path, bgr: np.ndarray) -> None:
    """Write a new jpeg. Never replace a file that is already on disk."""
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = crop_bars(bgr)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise SystemExit(f"could not encode {path}")
    path.write_bytes(buf.tobytes())


def fresh_name(media: Path, t: float) -> str:
    """A timestamp filename that does not collide with a file already saved."""
    name = stamp(t) + ".jpg"
    if not (media / name).exists():
        return name
    seconds = t % 60.0
    hundredths = int(round(seconds * 100))
    fine = f"{int(t // 60):02d}m{hundredths // 100:02d}.{hundredths % 100:02d}s.jpg"
    if not (media / fine).exists():
        return fine
    stem = fine[:-4]
    n = 2
    while (media / f"{stem}-{n}.jpg").exists():
        n += 1
    return f"{stem}-{n}.jpg"


def add_still(page: Path, t: float, bgr: np.ndarray) -> str:
    media, who = media_of(page)
    head, stills, tail = parse_stills(page.read_text())
    name = fresh_name(media, t)
    at = label(t)
    src = f"/media/{media.name}/{name}"
    still = Still(src, alt_for(stills, who, at), at)
    updated, added = add_new(stills, still)
    if not added:
        print(f"already in the gallery: {src}")
        return src
    save_jpeg(media / name, bgr)
    write_page(page, head, updated, tail)
    return src


def group_by_moment(moments: list[Event], choices: list[Event]) -> list[tuple[Event, list[Event]]]:
    """Put each choice under the moment it was offered for."""
    groups: list[tuple[Event, list[Event]]] = [(moment, []) for moment in moments]
    if not groups:
        return groups
    for event in choices:
        nearest = min(range(len(moments)), key=lambda i: abs(moments[i].t - event.t))
        groups[nearest][1].append(event)
    for moment, bucket in groups:
        if not any(abs(event.t - moment.t) < 0.05 for event in bucket):
            bucket.append(moment)
        bucket.sort(key=lambda event: event.t)
    return groups


def debug_jpeg(path: Path, bgr: np.ndarray) -> None:
    """A preview frame. Safe to replace; this folder is not the gallery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = crop_bars(bgr)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise SystemExit(f"could not encode {path}")
    path.write_bytes(buf.tobytes())


def in_gallery(event: Event, stills: list[Still]) -> bool:
    # A hundredth is a different frame of the same tenth. Do not call that one already saved.
    return any(abs(row.seconds() - event.t) < 0.02 for row in stills)


def _cast_tab_bar(moments: list[Event], thumbs: dict[int, str]) -> str:
    """Tabs for a trailer. Empty when there are not two characters to split."""
    counts: dict[int, int] = {}
    for moment in moments:
        for who in _tab_ids(moment):
            counts[who] = counts.get(who, 0) + 1
    named = [who for who in sorted(counts) if who]
    if len(named) < 2:
        return ""
    buttons = [f'<button type="button" class="on" data-tab="all">All <span>{len(moments)}</span></button>']
    for who in named:
        img = ""
        if who in thumbs:
            img = f'<img src="{html.escape(thumbs[who], quote=True)}" alt="">'
        buttons.append(
            f'<button type="button" data-tab="{who}">{img}Character {who} <span>{counts[who]}</span></button>'
        )
    if 0 in counts:
        buttons.append(f'<button type="button" data-tab="0">Others <span>{counts[0]}</span></button>')
    return f'<div class="tabs" id="tabs">{"".join(buttons)}</div>'


def _write_cast_thumbs(video: Path, frames: Path) -> dict[int, str]:
    """A face crop for each character tab. Not a gallery still."""
    thumbs: dict[int, str] = {}
    frames.mkdir(parents=True, exist_ok=True)
    for who, (t, box) in sorted(CAST_BEST.items()):
        crop = _crop_face(grab_at(video, t), box)
        if crop.size == 0:
            continue
        name = f"cast-{who}.jpg"
        path = frames / name
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        path.write_bytes(buf.tobytes())
        thumbs[who] = f"frames/{name}"
    return thumbs


def render_debug(title: str, groups: list[tuple[Event, list[tuple[Event, str, bool]]]], step: str, tabs: str = "") -> str:
    """Step 1 confirms moments. Step 2 picks a frame, and only for moments already kept."""
    choosing_frames = step == "frames"
    sections: list[str] = []
    for nth, (moment, frames) in enumerate(groups, start=1):
        why = ", ".join(moment.reasons) or "moment"
        cards: list[str] = []
        for event, src, have in frames:
            tags = ", ".join(event.reasons) or "choice"
            if event.subject >= SUBJECT_MEDIAN:
                tags += f" {event.subject:.2f}"
            if event.smile >= 0.08:
                tags += f", smile {event.smile:.2f}"
            if event.focus >= 8:
                tags += f", focus {event.focus:.0f}"
            at = html.escape(label(event.t))
            badge = '<em class="have">already in gallery</em>' if have else ""
            where = " nearby" if event.nearby else ""
            checked = " checked" if "kept moment" in event.reasons else ""
            cards.append(
                f'<label class="pick{where}">'
                f'<input type="checkbox" data-at="{at}" data-note="{html.escape(tags)}" data-have="{1 if have else 0}"{checked}>'
                f'<img src="{html.escape(src, quote=True)}" alt="{at}">'
                f"<span><time>{at}</time> · {html.escape(tags)}{badge}</span>"
                f"</label>"
            )
        if choosing_frames and moment.kept_times:
            kept = html.escape(_join_times(moment.kept_times))
            plural = "s" if len(moment.kept_times) > 1 else ""
            note = (
                f"Same shot as the kept moment{plural} at {kept}. "
                f"The shot runs {html.escape(label(moment.shot_lo))}–{html.escape(label(moment.shot_hi))}. "
                f"{len(frames)} frames, one every fifth of a second. "
                "Earlier is on the left. Dimmer frames are a few seconds before and after that shot, "
                "so a camera move is still a choice. Check the frame you want. This page does not change the gallery."
            )
            heading = f"Around {html.escape(label(moment.kept_times[0]))}"
        elif choosing_frames:
            heading = f"Moment {nth}"
            note = (
                f"Kept moment. {html.escape(why)}. "
                f"{len(frames)} frame{'s' if len(frames) != 1 else ''} around {html.escape(label(moment.t))}. "
                "Earlier is on the left. Check the frame, not the beat."
            )
        else:
            heading = f"Moment {nth}"
            note = (
                f"One frame stands in for this beat. {html.escape(why)}. "
                "Check it to keep the moment. Leave it unchecked to drop the moment. "
                "The frame itself is chosen in the next step."
            )
        ids = _tab_ids(moment) or ([0] if tabs else [])
        who = " ".join(str(i) for i in ids)
        attr = f' data-who="{who}"' if who else ""
        sections.append(
            f'<section data-step="{step}"{attr}>'
            f"<h2>{heading} <time>{html.escape(label(moment.t))}</time></h2>"
            f"<p>{note}</p>"
            f'<div class="grid">{"".join(cards)}</div>'
            f"</section>"
        )
    if choosing_frames:
        lead = (
            "Step 2 of 2. Each section is one shot you kept. "
            "Every fifth of a second is shown, earlier on the left, so you can pick the frame. "
            "Dimmer frames sit a few seconds before and after the detected shot, so a camera move is still a choice. "
            "A time with two decimals is one frame. Saving it saves that picture, not the other frame in the same tenth. "
            "A softer kept time also shows the sharper frame of that shot, with its own time. "
            "Check the frames you want. Leave a shot unchecked to skip it. "
            "Copy the box when the frames are confirmed. This page does not change the gallery."
        )
        box = "# step 2: frames\n# 0 chosen\n"
        hint = "step 2: frames"
    else:
        lead = (
            "Step 1 of 2. Choose the moments first. Each card is one beat, shown by a single frame. "
            "Check a moment to keep it, and leave it unchecked to drop it. "
            "Do not pick the frame yet. Copy the box, then run the frames step with those timestamps. "
            "This page does not change the gallery."
        )
        box = "# step 1: moments\n# 0 chosen\n"
        hint = "step 1: moments"
    if tabs:
        lead += " Characters are on their own tabs. A moment with more than one face is listed under each of them."
    body = "\n".join(sections) or "<p>No moments found.</p>"
    page = (
        DEBUG_PAGE.replace("__TITLE__", html.escape(title))
        .replace("__LEAD__", lead)
        .replace("__BOX__", box)
        .replace("__HINT__", hint)
        .replace("__TABS__", tabs)
        .replace("__BODY__", body)
    )
    return page


DEBUG_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__ — __HINT__</title>
  <style>
    :root { color-scheme: light; --ink: #17151f; --muted: #5e596b; --line: #e6e3ee; --accent: #5b45f5; --soft: #f6f5f8; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #fff; color: var(--ink); font: 1.02rem/1.5 Inter, "Segoe UI", sans-serif; }
    header, main { width: min(76rem, calc(100% - 1.5rem)); margin-inline: auto; }
    header { position: sticky; top: 0; z-index: 5; background: #fffc; backdrop-filter: blur(10px); padding: 1rem 0 0.8rem; border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: 1.45rem; letter-spacing: -0.03em; }
    h2 { margin: 0; font-size: 1.15rem; letter-spacing: -0.02em; }
    h2 time, .pick time { font-variant-numeric: tabular-nums; }
    p { margin: 0.35rem 0 0; color: var(--muted); }
    .bar { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; margin-top: 0.7rem; }
    button { cursor: pointer; border: 0; border-radius: 999px; background: var(--accent); color: #fff; font: inherit; font-weight: 700; padding: 0.4rem 0.9rem; }
    button.ghost { background: var(--soft); color: var(--ink); }
    #count { font-weight: 700; }
    textarea { width: 100%; min-height: 8.5rem; margin-top: 0.7rem; padding: 0.75rem 0.85rem; border: 1px solid var(--line); border-radius: 12px; background: #fff; color: var(--ink); font: 0.95rem/1.45 ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; resize: vertical; }
    section { padding: 1.4rem 0 0.4rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(16rem, 1fr)); gap: 0.8rem; margin-top: 0.7rem; }
    .pick { display: block; cursor: pointer; overflow: hidden; border: 3px solid transparent; border-radius: 14px; background: var(--soft); }
    .pick:has(input:checked) { border-color: var(--accent); background: #f3f0ff; }
    .pick.nearby { opacity: 0.72; }
    .pick.nearby:has(input:checked) { opacity: 1; }
    .pick input { position: absolute; opacity: 0; }
    .pick img { display: block; width: 100%; height: auto; background: #fff; }
    .pick span { display: block; padding: 0.4rem 0.55rem 0.5rem; font-size: 0.86rem; color: var(--muted); }
    .have { margin-left: 0.35rem; font-style: normal; font-weight: 700; color: var(--accent); }
    .tabs { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.8rem; }
    .tabs button { display: inline-flex; align-items: center; gap: 0.4rem; background: var(--soft); color: var(--ink); }
    .tabs button.on { background: var(--accent); color: #fff; }
    .tabs img { width: 1.8rem; height: 1.8rem; border-radius: 999px; object-fit: cover; background: #fff; }
    section.is-hidden { display: none; }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <p>__LEAD__</p>
    <div class="bar">
      <strong id="count">0 chosen</strong>
      <button type="button" id="copy">Copy choices</button>
      <button type="button" class="ghost" id="clear">Clear</button>
    </div>
    <textarea id="choices" spellcheck="false">__BOX__</textarea>
    __TABS__
  </header>
  <main>
__BODY__
  </main>
  <script>
    const box = document.getElementById("choices");
    const count = document.getElementById("count");
    const checks = [...document.querySelectorAll("input[data-at]")];
    const seconds = (at) => {
      const [minutes, rest] = at.split(":");
      return Number(minutes) * 60 + Number(rest);
    };
    const sync = () => {
      const picked = checks.filter((item) => item.checked).sort((a, b) => seconds(a.dataset.at) - seconds(b.dataset.at));
      const lines = picked.map((item) => {
        let line = item.dataset.at;
        if (item.dataset.note) line += "  " + item.dataset.note;
        if (item.dataset.have === "1") line += "  already in gallery";
        return line;
      });
      const step = document.querySelector("section")?.dataset.step === "frames" ? "step 2: frames" : "step 1: moments";
      count.textContent = lines.length + " chosen";
      box.value = ["# " + step, "# " + lines.length + " chosen", ...lines].join("\\n");
    };
    checks.forEach((item) => item.addEventListener("change", sync));
    document.getElementById("clear").addEventListener("click", () => {
      checks.forEach((item) => { item.checked = false; });
      sync();
    });
    document.getElementById("copy").addEventListener("click", async () => {
      const button = document.getElementById("copy");
      try {
        await navigator.clipboard.writeText(box.value);
        button.textContent = "Copied";
      } catch (err) {
        box.focus();
        box.select();
        button.textContent = "Select the box and copy";
      }
    });
    const tabs = [...document.querySelectorAll("#tabs [data-tab]")];
    const showTab = (id) => {
      document.querySelectorAll("section[data-who]").forEach((sec) => {
        const whos = (sec.dataset.who || "").split(" ");
        sec.classList.toggle("is-hidden", id !== "all" && !whos.includes(id));
      });
      tabs.forEach((btn) => btn.classList.toggle("on", btn.dataset.tab === id));
    };
    tabs.forEach((btn) => btn.addEventListener("click", () => showTab(btn.dataset.tab)));
    sync();
  </script>
</body>
</html>
"""


def _kept_time(fps: float, t: float) -> float:
    """The pasted time when it already seeks to this frame.

    Canonical time can be a hundredth earlier, so 0:32.67 became 0:32.66 and the
    gallery still no longer looked like the kept frame. Keep the time that was
    pasted when both land on the same picture.
    """
    canon = canonical_time(fps, t)
    if frame_index(fps, t) == frame_index(fps, canon):
        return round(t, 2)
    return canon


def _join_times(times: list[float]) -> str:
    """A short list of timestamps for a shot note."""
    labels = [label(t) for t in times]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _sample_shots(video: Path, step: float = 0.2) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Every fifth of a second: time, color histogram, scene vector.

    The time is on the fifth-of-a-second grid, so a strip lines up with the
    single-still picker. The vectors are only for finding the cut.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    index = 0
    nxt = 0.0
    last_print = -10.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = index / fps
        if t + 0.5 / fps >= nxt:
            samples.append((round(nxt, 1), histogram(bgr), scene_vec(bgr)))
            nxt += step
            if t - last_print >= 8:
                print(f"shot sample {t:.0f}s", file=sys.stderr, flush=True)
                last_print = t
        index += 1
        if total and index >= total:
            break
    cap.release()
    return samples


def _same_sample(a: tuple[float, np.ndarray, np.ndarray], b: tuple[float, np.ndarray, np.ndarray]) -> bool:
    """True when two nearby samples are the same cut.

    A blink can drop the color match to about 0.80 without leaving the shot.
    A real cut drops further, or the scene itself changes.
    """
    return float(np.dot(a[1], b[1])) >= 0.75 and float(np.dot(a[2], b[2])) >= 0.90


def _shot_continues(samples: list[tuple[float, np.ndarray, np.ndarray]], i: int, j: int) -> bool:
    """True when sample j is still the shot at i.

    A head turn can make one fifth of a second look like a cut. Look further in
    that same direction, not back at i. A sample always matches itself, and
    treating that as a recovery glued the whole video into one shot. A real cut
    does not match again within the next half-second.
    """
    if _same_sample(samples[i], samples[j]):
        return True
    step = 1 if j > i else -1
    k = j + step
    while 0 <= k < len(samples) and abs(samples[k][0] - samples[i][0]) <= 0.55:
        if _same_sample(samples[i], samples[k]):
            return True
        k += step
    return False


def _shot_span(
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    t: float,
    hint: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[int, int]:
    """Indexes of the shot that contains ``t``. A cut stops the walk.

    The anchor is the sample that matches the frame at ``t``. A halfway time is
    0.1s from two samples, and a float tie was landing on the next cut.
    """
    near = [i for i in range(len(samples)) if abs(samples[i][0] - t) <= 0.12]
    if not near:
        near = [min(range(len(samples)), key=lambda i: abs(samples[i][0] - t))]

    def matches(i: int) -> bool:
        if hint is None:
            return False
        hist = float(np.dot(samples[i][1], hint[0]))
        scene = float(np.dot(samples[i][2], hint[1]))
        return hist >= 0.75 and scene >= 0.88

    matched = [i for i in near if matches(i)]
    pool = matched or near
    anchor = min(pool, key=lambda i: (round(abs(samples[i][0] - t), 2), i))
    lo = anchor
    while lo > 0 and _shot_continues(samples, lo, lo - 1):
        lo -= 1
    hi = anchor
    while hi + 1 < len(samples) and _shot_continues(samples, hi, hi + 1):
        hi += 1
    return lo, hi


def _same_picture(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two frames are the same cut, including a small move inside it."""
    return float(np.dot(histogram(a), histogram(b))) >= 0.75 and float(np.dot(scene_vec(a), scene_vec(b))) >= 0.88


def _read_at(cap: cv2.VideoCapture, fps: float, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index(fps, t))
    ok, bgr = cap.read()
    return bgr if ok else None


def _clearer_time(cap: cv2.VideoCapture, fps: float, t: float) -> float | None:
    """A clearly sharper frame of the same shot, within a fifth of a second.

    The tenth that was pasted can be the soft frame. The next frame can be the
    picture. It is offered beside the pasted time, with its own timestamp, so
    the two are not saved as the same tenth.
    """
    center = frame_index(fps, t)
    start = max(0, center - 8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows: list[tuple[int, np.ndarray, float]] = []
    index = start
    while index <= center + 8:
        ok, bgr = cap.read()
        if not ok:
            break
        cropped = crop_bars(bgr)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        rows.append((index, bgr, float(cv2.Laplacian(gray, cv2.CV_64F).var())))
        index += 1
    asked = next((row for row in rows if row[0] == center), None)
    if asked is None:
        return None
    pos = next(i for i, row in enumerate(rows) if row[0] == center)
    keep = {pos}
    for i in range(pos, 0, -1):
        if not _same_picture(rows[i - 1][1], rows[i][1]):
            break
        keep.add(i - 1)
    for i in range(pos, len(rows) - 1):
        if not _same_picture(rows[i][1], rows[i + 1][1]):
            break
        keep.add(i + 1)
    best = max((rows[i] for i in keep), key=lambda row: row[2])
    if best[0] == center or best[2] < asked[2] * 1.6 or best[2] < asked[2] + 25:
        return None
    return canonical_time(fps, best[0] / fps)


def shot_strips(video: Path, moments: list[Event], stills: list[Still]) -> list[tuple[Event, list[Event]]]:
    """One strip per shot. Every fifth of a second, earlier first.

    A kept moment is a time inside a shot, not a short list of ranked frames.
    The shot is the one the frame at that time belongs to, not the sample a
    float tie lands on. Two kept times in the same shot share one strip. A
    gallery still inside the shot is included so it can be marked. A clearly
    sharper frame of that same shot is added with its own time. A few seconds
    before and after the detected cut are included too, marked as nearby, so a
    camera move is still a choice. This does not change the gallery.
    """
    if not moments:
        return []
    print("sampling shots every 0.2s", file=sys.stderr, flush=True)
    samples = _sample_shots(video)
    if not samples:
        raise SystemExit(f"no frames in {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    rows: list[tuple[int, int, Event]] = []
    clearer: dict[int, float] = {}
    for moment in moments:
        bgr = _read_at(cap, fps, moment.t)
        hint = (histogram(bgr), scene_vec(bgr)) if bgr is not None else None
        lo, hi = _shot_span(samples, moment.t, hint)
        alt = _clearer_time(cap, fps, moment.t)
        if alt is not None and abs(alt - moment.t) >= 0.015:
            clearer[id(moment)] = alt
            print(f"  {label(moment.t)} is soft; sharper frame {label(alt)}", file=sys.stderr)
        rows.append((lo, hi, moment))
    cap.release()
    rows.sort(key=lambda row: (row[0], row[2].t))
    merged: list[tuple[int, int, list[Event]]] = []
    for lo, hi, moment in rows:
        if merged and lo <= merged[-1][1]:
            prev_lo, prev_hi, group = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi), group + [moment])
        else:
            merged.append((lo, hi, [moment]))
    groups: list[tuple[Event, list[Event]]] = []
    for lo, hi, group in merged:
        group.sort(key=lambda event: event.t)
        ids: list[int] = []
        for moment in group:
            for who in _tab_ids(moment):
                if who not in ids:
                    ids.append(who)
        who_notes = ["others" if who == 0 else f"character {who}" for who in ids]
        if not who_notes and any("soyeon" in moment.reasons for moment in group):
            who_notes = ["soyeon"]
        kept = [_kept_time(fps, moment.t) for moment in group]
        rep = replace(
            group[0],
            cast_ids=ids,
            shot_lo=samples[lo][0],
            shot_hi=samples[hi][0],
            kept_times=kept,
            reasons=["kept", *who_notes],
        )
        times = [canonical_time(fps, samples[i][0]) for i in range(lo, hi + 1)]
        # A pan or a zoom can look like a cut. Offer a few seconds past the
        # detected edges too, so a frame before or after the shot can be chosen.
        # Those stay marked as nearby and do not change where the shot was found.
        pad = 3.0
        edge = {t: False for t in times}
        for i in range(len(samples)):
            t = samples[i][0]
            if t < samples[lo][0] - pad or t > samples[hi][0] + pad:
                continue
            key = canonical_time(fps, t)
            if any(abs(key - have) < 0.015 for have in edge):
                continue
            edge[key] = not (samples[lo][0] - 0.05 <= t <= samples[hi][0] + 0.05)
        extras = [*kept, *[clearer.get(id(moment)) for moment in group], *[still.seconds() for still in stills]]
        for extra in extras:
            if extra is None:
                continue
            forced = any(abs(extra - k) < 0.015 for k in kept) or any(
                alt is not None and abs(extra - alt) < 0.015 for alt in (clearer.get(id(moment)) for moment in group)
            )
            inside = samples[lo][0] - 0.12 <= extra <= samples[hi][0] + 0.12
            key = canonical_time(fps, extra)
            if (forced or inside) and not any(abs(key - have) < 0.015 for have in edge):
                edge[key] = False
        bucket: list[Event] = []
        seen_shown: set[int] = set()
        for t in sorted(edge):
            match = [k for k in kept if abs(t - k) < 0.015]
            shown = match[0] if match else t
            key = int(round(shown * 100))
            if key in seen_shown:
                continue
            seen_shown.add(key)
            bucket.append(Event(
                t=shown,
                sharp=group[0].sharp,
                shot=group[0].shot,
                reasons=(["kept moment"] if match else ["choice"]) + who_notes,
                faces=list(group[0].faces),
                cast_ids=list(ids),
                nearby=edge[t] and not match,
            ))
        groups.append((rep, bucket))
    return groups


def _read_frames(video: Path, events: list[Event]) -> dict[float, np.ndarray]:
    """Decode each asked timestamp once, walking forward through the video."""
    if not events:
        return {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    wanted: dict[int, float] = {}
    for event in events:
        wanted[frame_index(fps, event.t)] = round(event.t, 3)
    start = min(wanted)
    last = max(wanted)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    found: dict[float, np.ndarray] = {}
    index = start
    while index <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        key = wanted.get(index)
        if key is not None:
            found[key] = bgr
        index += 1
    cap.release()
    return found


def write_debug(
    video: Path,
    page: Path,
    moments: list[Event],
    choices: list[Event],
    dest: Path | None,
    step: str = "moments",
    strips: list[tuple[Event, list[Event]]] | None = None,
) -> Path:
    """Write one debug step. Pictures go in a debug folder, never into the gallery.

    Step 1 writes one frame per moment, so the beat can be kept or dropped.
    Step 2 writes every fifth of a second of each kept shot.
    """
    media, who = media_of(page)
    _, stills, _ = parse_stills(page.read_text())
    out = dest or (ROOT / "tools" / "debug" / media.name)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    frames = out / "frames"
    groups = strips if strips is not None else group_by_moment(moments, choices if step == "frames" else moments)
    images = _read_frames(video, [event for _moment, bucket in groups for event in bucket]) if strips is not None else {}
    rendered: list[tuple[Event, list[tuple[Event, str, bool]]]] = []
    total = sum(len(bucket) for _, bucket in groups)
    done = 0
    for moment, bucket in groups:
        cards: list[tuple[Event, str, bool]] = []
        for event in bucket:
            name = stamp(event.t) + ".jpg"
            bgr = images.get(round(event.t, 3))
            if bgr is None:
                bgr = grab_at(video, event.t)
            debug_jpeg(frames / name, bgr)
            cards.append((event, f"frames/{name}", in_gallery(event, stills)))
            done += 1
            if done % 8 == 0 or done == total:
                print(f"debug frames {done}/{total}", file=sys.stderr, flush=True)
        rendered.append((moment, cards))
    found = re.search(r'^title = "([^"]*)"', page.read_text(), re.M)
    title = found.group(1) if found else who
    thumbs = _write_cast_thumbs(video, frames) if page_is_trailer(page) else {}
    tabs = _cast_tab_bar([moment for moment, _cards in rendered], thumbs)
    name = "frames.html" if step == "frames" else "index.html"
    path = out / name
    path.write_text(render_debug(title, rendered, step, tabs))
    return path


def parse_frame_times(text: str) -> list[float]:
    """Timestamps from the step-1 textbox. Comments and notes after the time are ignored."""
    times: list[float] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        token = line.split()[0].rstrip(",")
        if ":" in token:
            minutes, rest = token.split(":", 1)
            if minutes.lstrip("-").isdigit() and rest.replace(".", "", 1).isdigit():
                times.append(int(minutes) * 60 + float(rest))
                continue
        try:
            times.append(float(token))
        except ValueError:
            continue
    return times


def moments_at(events: list[Event], times: list[float], cast: bool = False) -> list[Event]:
    """The confirmed beats, at the times that were kept.

    Step 2 does not move a beat. A marked time the face match missed stays put,
    even when the nearest matched face is a later shot. Snapping it would show
    the wrong moment.
    """
    if not times:
        raise SystemExit("no timestamps in --frames; paste the step-1 textbox, or pass 6,23.5,62.8")
    found: list[Event] = []
    seen: set[int] = set()
    step = _step(events)
    for asked in times:
        index = min(range(len(events)), key=lambda i: abs(events[i].t - asked))
        key = int(round(asked * 100))
        if key in seen:
            continue
        seen.add(key)
        # Keep the time that was pasted. Rounding to a tenth made 0:32.67 into
        # 0:32.7, a different frame, so the gallery still was no longer the kept one.
        event = replace(events[index], t=round(asked, 2), reasons=["moment", "kept"])
        if cast:
            event.cast_ids = _cast_ids_near(events, index, step)
            event.reasons.extend(_who_labels(event))
        else:
            event.reasons.append("soyeon")
        found.append(event)
    found.sort(key=lambda event: event.t)
    return found


def cmd_at(video: Path, page: Path, at: float, window: float) -> None:
    t, bgr = grab(video, at, window)
    src = add_still(page, t, bgr)
    print(f"{src}  {label(t)}  (asked {at:.1f}s, sharpest within {window:.2f}s)")


def _picture_gap(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two frames look. The same picture stays under 1; a new frame does not."""
    left = cv2.cvtColor(cv2.resize(a, (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(cv2.resize(b, (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(left - right).mean())


def cmd_final(video: Path, page: Path, text: str, window: float) -> None:
    """Step 3. The textbox is the gallery. A checked frame is the frame that is saved.

    Nothing is swapped for a sharper neighbor. A time that was not checked is
    removed, including an older still of the same beat. The same picture already
    saved under an older timestamp keeps its file and takes the checked time.
    """
    del window  # The chosen frame is the frame. A sharper neighbor is a different picture.
    times = parse_frame_times(text)
    if not times:
        raise SystemExit("no timestamps in --final; paste the step-2 textbox")
    head, stills, tail = parse_stills(page.read_text())
    media, who = media_of(page)
    fps = _fps_of(video)
    kept: list[Still] = []
    used: set[int] = set()
    added = 0
    relabeled = 0
    for asked in times:
        t = canonical_time(fps, asked)
        at = label(t)
        name = stamp(t) + ".jpg"
        exact = next((i for i, row in enumerate(stills) if i not in used and Path(row.src).name == name), None)
        if exact is not None:
            used.add(exact)
            kept.append(stills[exact])
            print(f"  kept {stills[exact].at}")
            continue
        bgr = crop_bars(grab_at(video, t))
        match = None
        for i, row in enumerate(stills):
            if i in used or abs(row.seconds() - t) > 0.5:
                continue
            old = media / Path(row.src).name
            if not old.exists():
                continue
            previous = cv2.imread(str(old))
            if previous is not None and _picture_gap(bgr, previous) < 1.2:
                match = i
                break
        if match is not None:
            row = stills[match]
            used.add(match)
            old = media / Path(row.src).name
            if old.name != name:
                dest = media / name
                if dest.exists():
                    dest.unlink()
                old.rename(dest)
                relabeled += 1
                print(f"  labeled {row.at} as {at}")
            else:
                print(f"  kept {at}")
            src = f"/media/{media.name}/{name}"
            kept.append(Still(src, alt_for(stills or kept, who, at), at))
            continue
        src = f"/media/{media.name}/{name}"
        still = Still(src, alt_for(stills or kept, who, at), at)
        save_jpeg(media / name, bgr)
        kept.append(still)
        added += 1
        print(f"{src}  {at}")
    removed = 0
    for i, row in enumerate(stills):
        if i in used:
            continue
        old = media / Path(row.src).name
        if old.exists():
            old.unlink()
        removed += 1
        print(f"  removed {row.at}")
    if not added and not removed and not relabeled:
        print(f"added 0, removed 0, gallery {len(kept)}")
        return
    kept.sort(key=lambda row: (row.seconds(), row.src))
    write_page(page, head, kept, tail)
    print(f"added {added}, relabeled {relabeled}, removed {removed}, gallery {len(kept)}")


def cmd_scan(
    video: Path,
    page: Path,
    step: float,
    limit: int | None,
    per: int,
    dry_run: bool,
    debug: bool,
    debug_out: Path | None,
    frames: str | None,
) -> None:
    events = scan(video, page, step)
    cast = page_is_trailer(page)
    if frames is not None:
        # Step 2 starts only after the moments are confirmed. It does not invent new beats.
        found = moments_at(events, parse_frame_times(frames), cast)
        _, stills, _ = parse_stills(page.read_text())
        strips = shot_strips(video, found, stills)
        nframes = sum(len(bucket) for _moment, bucket in strips)
        print(f"kept moments {len(found)}, shots {len(strips)}, frames to choose {nframes}")
        for moment, bucket in strips:
            print(f"  {label(moment.shot_lo)}–{label(moment.shot_hi)}  {len(bucket)} frames  kept {_join_times(moment.kept_times)}")
        path = write_debug(video, page, found, [], debug_out, "frames", strips)
        print(f"frames page {path}")
        print("gallery unchanged — copy the textbox to confirm the frames")
        return
    found = moments_of(events, limit, _known_times(page), cast)
    if debug:
        # Step 1 is the moment list. One frame stands in for the beat. No frame picking yet.
        print(f"moments {len(found)}")
        for event in found:
            why = ",".join(event.reasons) or "moment"
            print(f"  {label(event.t)}  {why}")
        path = write_debug(video, page, found, found, debug_out, "moments")
        print(f"debug page {path}")
        print("gallery unchanged — step 1 only; copy the textbox, then run --frames with those times")
        return
    chosen = choices_around(events, found, per, video, page, cast)
    print(f"moments {len(found)}, frames to choose {len(chosen)}")
    for event in chosen:
        why = ",".join(event.reasons) or "choice"
        print(f"  {label(event.t)}  {why}")
    if dry_run:
        print("dry run, gallery unchanged")
        return
    head, stills, tail = parse_stills(page.read_text())
    media, who = media_of(page)
    have = {row.src for row in stills}
    added = 0
    for event in chosen:
        at = label(event.t)
        name = stamp(event.t) + ".jpg"
        src = f"/media/{media.name}/{name}"
        if src in have:
            continue
        still = Still(src, alt_for(stills, who, at), at)
        path = media / name
        if not path.exists():
            save_jpeg(path, grab_at(video, event.t))
        stills.append(still)
        have.add(src)
        added += 1
        why = ",".join(event.reasons) or "choice"
        print(f"{src}  {at}  {why}")
    if not added:
        print(f"added 0, gallery {len(stills)}")
        return
    stills.sort(key=lambda row: (row.seconds(), row.src))
    write_page(page, head, stills, tail)
    print(f"added {added}, gallery {len(stills)}")


def build() -> None:
    zola = shutil.which("zola") or str(Path.home() / ".local/bin/zola")
    if not Path(zola).exists():
        print("zola not found, skipped build", file=sys.stderr)
        return
    subprocess.run([zola, "build", "--force", "--output-dir", "public"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture meaningful stills in timestamp order.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--page", type=Path, required=True, help="Gallery markdown page")
    parser.add_argument("--at", type=float, help="Capture the sharpest frame around this second")
    parser.add_argument("--window", type=float, default=0.35, help="Search +/- this many seconds around --at")
    parser.add_argument("--step", type=float, default=0.25, help="Scan interval in seconds")
    parser.add_argument("--moments", type=int, default=None, help="Optional ceiling. Omit it and the scan decides how many moments the video has")
    parser.add_argument("--per", type=int, default=3, help="Auto-gen only. Frames to keep around each moment. Step 2 shows the whole shot")
    parser.add_argument("--dry-run", action="store_true", help="Print the moments and choices, write nothing")
    parser.add_argument("--debug", action="store_true", help="Step 1. One frame per moment, so the beats can be kept or dropped. A trailer splits those moments into a tab per character. Does not change the gallery")
    parser.add_argument("--frames", help="Step 2. Timestamps kept in step 1 (a textbox paste, or 6,23.5,62.8). Shows every fifth of a second of each kept shot")
    parser.add_argument("--final", help="Step 3. The step-2 textbox. Saves each checked frame and removes every still that was not checked")
    parser.add_argument("--focus", type=float, default=0.5, help="Step 3 may move this many seconds to the nearest in-focus frame of the same shot")
    parser.add_argument("--debug-out", type=Path, help="Folder for the debug page (default tools/debug/<slug>)")
    parser.add_argument("--build", action="store_true", help="Rebuild the site after writing")
    args = parser.parse_args()
    page = args.page if args.page.is_absolute() else ROOT / args.page
    if not page.exists():
        raise SystemExit(f"no page: {page}")
    if not args.video.exists():
        raise SystemExit(f"no video: {args.video}")
    if (args.debug or args.frames is not None or args.final is not None) and args.at is not None:
        raise SystemExit("--debug, --frames, and --final start from a scan or a paste, not a single --at")
    if args.final is not None and (args.debug or args.frames is not None):
        raise SystemExit("--final is the last step; do not combine it with --debug or --frames")
    if args.at is None and args.final is not None:
        cmd_final(args.video, page, args.final, args.focus)
    elif args.at is None:
        cmd_scan(args.video, page, args.step, args.moments, args.per, args.dry_run, args.debug, args.debug_out, args.frames)
    else:
        cmd_at(args.video, page, args.at, args.window)
    if args.build:
        build()


if __name__ == "__main__":
    main()
