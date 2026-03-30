# models/video_classifier.py
import os
import cv2
import uuid
from collections import defaultdict
from image_classifier import classify_image

def extract_frames(video_path: str, max_frames: int = 5, out_dir: str = None):
    """Extract up to max_frames evenly spaced frames from video."""
    out_paths = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return out_paths
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            # read sequentially
            saved = 0
            while saved < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                path = os.path.join(out_dir or os.path.dirname(video_path), f"frame_{uuid.uuid4().hex}.jpg")
                cv2.imwrite(path, frame)
                out_paths.append(path)
                saved += 1
            cap.release()
            return out_paths
        # evenly spaced frames
        step = max(1, total // max_frames)
        idx = saved = 0
        while saved < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                path = os.path.join(out_dir or os.path.dirname(video_path), f"frame_{uuid.uuid4().hex}.jpg")
                cv2.imwrite(path, frame)
                out_paths.append(path)
                saved += 1
            idx += 1
        cap.release()
    except Exception as e:
        print("Frame extraction failed:", e)
    return out_paths

def classify_video(video_path: str, topk: int = 1, max_frames: int = 5):
    """
    Extract frames, convert each to descriptive text using LLM,
    then return top aggregated description.
    """
    frames = extract_frames(video_path, max_frames=max_frames)
    if not frames:
        return [("no_frames", 1.0)]
    
    agg_texts = defaultdict(float)
    for f in frames:
        try:
            results = classify_image(f, topk=topk)
            for text, score in results:
                agg_texts[text] += score
        except Exception:
            continue
        finally:
            if os.path.exists(f):
                os.remove(f)
    
    # return top aggregated text
    if not agg_texts:
        return [("unable_to_describe_video", 1.0)]
    
    sorted_texts = sorted(agg_texts.items(), key=lambda x: x[1], reverse=True)
    return sorted_texts[:topk]
