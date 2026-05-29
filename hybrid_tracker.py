import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO
from enum import Enum
from scipy.optimize import linear_sum_assignment

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL = os.path.join(MODELS_DIR, "yolo11n_visdrone.onnx")

class TrackState(Enum):
  SEARCHING = "SEARCHING"
  TRACKING = "TRACKING"
  LOST = "LOST"

# --- Advanced Components ---

def calculate_iou(box1, box2):
  if box1 is None or box2 is None: return 0.0
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2
  xi1, yi1 = max(x1, x2), max(y1, y2)
  xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
  inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
  union = w1 * h1 + w2 * h2 - inter
  return inter / union if union > 0 else 0

class EMAFilter:
  def __init__(self, alpha=0.85):
    self.alpha = alpha
    self.state = None
    self.count = 0

  def update(self, val):
    if val is None: return None
    v = np.array(val, dtype=np.float32)
    if self.state is None: 
      self.state = v
    else:
      # Перші 3 кадри - жорстка фіксація без інерції
      curr_alpha = 1.0 if self.count < 3 else self.alpha
      self.state = curr_alpha * v + (1 - curr_alpha) * self.state
    
    self.count += 1
    return tuple(map(int, self.state))

class GMC_Stabilizer:
  def __init__(self):
    self.orb = cv2.ORB_create(nfeatures=500)
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    self.prev_des = None
    self.prev_kps = None
  def update(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps, des = self.orb.detectAndCompute(gray, None)
    H = np.eye(3)
    if self.prev_des is not None and des is not None:
      matches = self.bf.match(self.prev_des, des)
      if len(matches) > 15:
        src_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    self.prev_kps, self.prev_des = kps, des
    return H if H is not None else np.eye(3)
  def apply(self, bbox, H):
    if np.allclose(H, np.eye(3)): return bbox
    x, y, w, h = bbox
    pts = np.float32([[x, y], [x+w, y+h]]).reshape(-1, 1, 2)
    pts_t = cv2.perspectiveTransform(pts, H)
    nx, ny = float(pts_t[0][0][0]), float(pts_t[0][0][1])
    nw, nh = float(pts_t[1][0][0]) - nx, float(pts_t[1][0][1]) - ny
    return (nx, ny, nw, nh)

class ReID_Extractor:
  def extract(self, frame, bbox):
    x, y, w, h = map(int, bbox)
    roi = frame[max(0,y):y+h, max(0,x):x+w]
    if roi.size == 0 or h < 2: return np.zeros(960, dtype=np.float32)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Use Global Hue (0-180) and Saturation (0-256) to prevent jitter from BB shifts
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist.flatten()

  def compare(self, feat1, feat2):
    if len(feat1) != len(feat2): return 1.0
    return cv2.compareHist(feat1, feat2, cv2.HISTCMP_BHATTACHARYYA)

# --- ByteTrack-inspired MOT Tracker ---

class ByteMOT:
  def __init__(self):
    self.tracks = {} # id -> bbox
    self.next_id = 1
    self.max_age = 15
    self.ages = {} # id -> age
  
  def update(self, detections, H=None):
    # Apply GMC to existing tracks
    for tid in self.tracks:
      self.tracks[tid] = self._apply_gmc(self.tracks[tid], H)
      
    if not detections:
      return self._cleanup()

    # Split detections by confidence (ByteTrack style)
    high_dets = [d for d in detections if d['conf'] > 0.5]
    low_dets = [d for d in detections if 0.1 < d['conf'] <= 0.5]
    
    # 1. Match high-conf with existing tracks
    matched_ids, remaining_high = self._match(self.tracks, high_dets, 0.3)
    
    # 2. Match low-conf with remaining tracks
    remaining_tracks = {tid: self.tracks[tid] for tid in self.tracks if tid not in matched_ids}
    matched_ids_low, _ = self._match(remaining_tracks, low_dets, 0.4)
    
    matched_ids.update(matched_ids_low)
    
    # Add new tracks from remaining high-conf
    for d_idx in remaining_high:
      self.tracks[self.next_id] = high_dets[d_idx]['bbox']
      self.ages[self.next_id] = 0
      self.next_id += 1
      
    return self._cleanup(matched_ids)

  def _match(self, tracks, dets, thresh):
    if not tracks or not dets: return {}, list(range(len(dets)))
    t_ids = list(tracks.keys())
    iou_matrix = np.zeros((len(t_ids), len(dets)))
    for i, tid in enumerate(t_ids):
      for j, d in enumerate(dets):
        iou_matrix[i, j] = calculate_iou(tracks[tid], d['bbox'])
    
    ri, ci = linear_sum_assignment(1 - iou_matrix)
    matches = {}
    used_dets = set()
    for r, c in zip(ri, ci):
      if iou_matrix[r, c] > thresh:
        tid = t_ids[r]
        self.tracks[tid] = dets[c]['bbox']
        self.ages[tid] = 0
        matches[tid] = True
        used_dets.add(c)
    return matches, [i for i in range(len(dets)) if i not in used_dets]

  def _apply_gmc(self, bbox, H):
    if H is None or np.allclose(H, np.eye(3)): return bbox
    pts = np.float32([[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]]).reshape(-1, 1, 2)
    pts_t = cv2.perspectiveTransform(pts, H)
    nx, ny = float(pts_t[0][0][0]), float(pts_t[0][0][1])
    nw, nh = float(pts_t[1][0][0]) - nx, float(pts_t[1][0][1]) - ny
    return (nx, ny, nw, nh)

  def _cleanup(self, matched_ids=None):
    if matched_ids is None: matched_ids = {}
    to_del = []
    for tid in self.tracks:
      if tid not in matched_ids:
        self.ages[tid] += 1
        if self.ages[tid] > self.max_age: to_del.append(tid)
    for tid in to_del:
      del self.tracks[tid]
      del self.ages[tid]
    return self.tracks

# --- Shared Thread Data ---

class SharedState:
  def __init__(self):
    self.frame = None
    self.results = None # {id: bbox}
    self.sot_bbox = None
    self.state = TrackState.SEARCHING
    self.running = True
    self.lock = threading.Lock()
    self.selected_target = None # bbox
    self.reset_requested = False
    self.last_valid_bbox = None # Для Ghost Box
    self.screenshot_countdown = 0 # Лічильник для серійної зйомки
    
    # User Interaction
    self.drag_start = None
    self.drag_end = None

shared = SharedState()

# --- Processing Worker (The Brain) ---

class TrackingWorker(threading.Thread):
  def __init__(self, yolo_freq=10):
    super().__init__(daemon=True)
    self.yolo_freq = yolo_freq
    self.yolo = YOLO(YOLO_MODEL, task="detect")
    self.mot = ByteMOT()
    self.reid = ReID_Extractor()
    self.gmc = GMC_Stabilizer()
    self.ema = EMAFilter(alpha=0.85)
    
    # SOT
    params = cv2.TrackerNano_Params()
    params.backbone = os.path.join(MODELS_DIR, "nanotrack_backbone_sim.onnx")
    params.neckhead = os.path.join(MODELS_DIR, "nanotrack_head_sim.onnx")
    self.params = params
    self.sot_tracker = None
    
    self.reference_feat = None
    self.loss_frames = 0
    self.frame_count = 0
    self.prev_area = 0
    self.prev_cx, self.prev_cy = 0, 0
    self.bad_reid_counter = 0

  def run(self):
    while shared.running:
      with shared.lock:
        frame = shared.frame.copy() if shared.frame is not None else None
        new_target = shared.selected_target
        shared.selected_target = None
        reset = shared.reset_requested
        shared.reset_requested = False
      
      if frame is None:
        time.sleep(0.01)
        continue

      self.frame_count += 1
      H = self.gmc.update(frame)
      
      if reset:
        shared.state = TrackState.SEARCHING
        self.sot_tracker = None

      if new_target:
        self.sot_tracker = cv2.TrackerNano_create(self.params)
        
        # [BUGFIX] Знаходимо точні координати об'єкта на ПОТОЧНОМУ кадрі
        res = self.yolo.predict(frame, classes=[0,1,2,3,4,5,6,7,8,9], verbose=False)
        best_bbox = new_target
        best_dist = float('inf')
        cx, cy = new_target[0] + new_target[2]/2, new_target[1] + new_target[3]/2
        
        for b in res[0].boxes:
          box = b.xywh.cpu().numpy()[0]
          dist = np.hypot(box[0] - cx, box[1] - cy)
          if dist < best_dist and dist < max(new_target[2], new_target[3]) * 1.5:
            best_dist = dist
            best_bbox = (int(box[0]-box[2]/2), int(box[1]-box[3]/2), int(box[2]), int(box[3]))
        
        x, y, w, h = map(int, best_bbox)
        x, y = max(0, x), max(0, y)
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        clean_target = (x, y, w, h)
        
        try:
          self.sot_tracker.init(frame, clean_target)
          self.reference_feat = self.reid.extract(frame, clean_target)
          self.prev_area = clean_target[2] * clean_target[3]
          self.prev_cx = clean_target[0] + clean_target[2]/2
          self.prev_cy = clean_target[1] + clean_target[3]/2
          shared.state = TrackState.TRACKING
          self.loss_frames = 0
          print(f"SOT Initialized successfully with {clean_target}")
        except Exception as e:
          print(f"ERROR: Failed to init SOT tracker: {e}")
          print(f"Target: {clean_target}, Frame Shape: {frame.shape}")
          shared.state = TrackState.SEARCHING

      # --- SEARCHING MODE (MOT) ---
      if shared.state == TrackState.SEARCHING:
        if self.frame_count % 3 == 0:
          res = self.yolo.predict(frame, classes=[0,1,2,3,4,5,6,7,8,9], verbose=False)
          dets = []
          for b in res[0].boxes:
            box = b.xywh.cpu().numpy()[0]
            dets.append({'bbox': (int(box[0]-box[2]/2), int(box[1]-box[3]/2), int(box[2]), int(box[3])), 
                  'conf': float(b.conf)})
          mot_res = self.mot.update(dets, H)
          with shared.lock:
            shared.results = mot_res
            shared.sot_bbox = None
        else:
          self.mot.update([], H) # Just GMC

      # --- TRACKING MODE (Hybrid SOT) ---
      elif shared.state in [TrackState.TRACKING, TrackState.LOST]:
        success = False
        resync = False
        
        if shared.state == TrackState.TRACKING:
          ok, bbox = self.sot_tracker.update(frame)
          if ok:
            bbox = self.gmc.apply(bbox, H)
            cx, cy = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
            area = bbox[2] * bbox[3]
            # Allow box to touch edges, but fail if center is completely off-screen
            if cx < 0 or cx > frame.shape[1] or cy < 0 or cy > frame.shape[0]:
              print(f"[SOT] Object center left the screen at {bbox}")
              ok = False
            
            if ok and (area > self.prev_area * 1.8 or area < self.prev_area * 0.4):
              print(f"[SOT] Area changed significantly! {self.prev_area:.0f} -> {area:.0f}. Likely rotation or occlusion.")
              # We DO NOT set ok = False here anymore. Let NanoTrack survive the rotation.
            
            if ok:
              score = self.reid.compare(self.reference_feat, self.reid.extract(frame, bbox))
              
              # [PATIENCE LOGIC] Handle long occlusions or WRONG object tracking
              if score > 0.85: # Increased threshold to be more sensitive to bad ReID
                self.bad_reid_counter += 1
                if self.bad_reid_counter % 5 == 0:
                  print(f"[SOT] Low confidence for {self.bad_reid_counter} frames... (Score: {score:.3f})")
              else:
                self.bad_reid_counter = 0

              if self.bad_reid_counter > 15:
                print(f"[SOT] ReID mismatch (Score {score:.2f}). NanoTrack likely lost. Switching to LOST.")
                ok = False
                self.bad_reid_counter = 0
                
              # ReID mismatch logging
              if ok and score > 0.75:
                pass # Just keep tracking, but we're watching the score above
            
            if ok:
              # Identity Persistence & Collision Logic
              if self.frame_count % self.yolo_freq == 0:
                res = self.yolo.predict(frame, verbose=False)
                candidates = []
                # [FIX] Збільшений gate, щоб "дотягнутися" до цілі, якщо трекер пішов за іншою
                gate = max(150, max(bbox[2], bbox[3])*2.0)
                
                for det in res[0].boxes:
                  b = det.xywh.cpu().numpy()[0]
                  db = (int(b[0]-b[2]/2), int(b[1]-b[3]/2), int(b[2]), int(b[3]))
                  if np.hypot(b[0]-cx, b[1]-cy) < gate:
                    ds = self.reid.compare(self.reference_feat, self.reid.extract(frame, db))
                    candidates.append({'bbox': db, 'score': ds, 'conf': float(det.conf)})
                
                if candidates:
                  best_reid_cand = min(candidates, key=lambda x: x['score'])
                  best_spatial_cand = max(candidates, key=lambda x: calculate_iou(bbox, x['bbox']))
                  
                  current_iou = calculate_iou(bbox, best_spatial_cand['bbox'])
                  current_yolo_score = best_spatial_cand['score']
                  
                  # [1] Чи слідкує NanoTrack за ІНШИМ об'єктом?
                  # Якщо той об'єкт, за яким ми йдемо, має значно гірший колір, ніж сусідній кандидат
                  is_wrong_object = best_reid_cand != best_spatial_cand and \
                           (current_yolo_score > best_reid_cand['score'] + 0.12) and \
                           best_reid_cand['score'] < 0.38
                           
                  # [2] Чи зісковзнув NanoTrack на фон?
                  is_drifted = current_iou < 0.4 and best_reid_cand['score'] < 0.38
                  
                  # [3] Чи вибухнула площа?
                  cand_area = best_reid_cand['bbox'][2] * best_reid_cand['bbox'][3]
                  is_area_wrong = area > (cand_area * 1.8) or area < (cand_area * 0.4)
                  
                  if is_wrong_object or is_drifted or (is_area_wrong and best_reid_cand['score'] < 0.45):
                    print(f"[HYBRID] Snap-back! WrongObj:{is_wrong_object}, Drift:{is_drifted}, Area:{is_area_wrong}")
                    bx, by, bw, bh = map(int, best_reid_cand['bbox'])
                    bx, by = max(0, bx), max(0, by)
                    bw = max(1, min(bw, frame.shape[1] - bx))
                    bh = max(1, min(bh, frame.shape[0] - by))
                    clean_best = (bx, by, bw, bh)
                    try:
                      self.sot_tracker.init(frame, clean_best)
                      bbox = clean_best
                      resync = True
                    except: pass
                    
                  # [4] Адаптація паспорта кольору (тільки якщо трекінг ідеальний)
                  elif current_iou > 0.8 and current_yolo_score < 0.30 and not is_area_wrong:
                    current_feat = self.reid.extract(frame, bbox)
                    self.reference_feat = 0.91 * self.reference_feat + 0.09 * current_feat
                    cv2.normalize(self.reference_feat, self.reference_feat, 0, 1, cv2.NORM_MINMAX)
              
              success = True
              self.raw_bbox = bbox
              self.prev_cx, self.prev_cy = cx, cy
              self.prev_area = 0.95 * self.prev_area + 0.05 * area
          
          if not success:
            print("[STATUS] Target tracking failed. Switching to LOST.")
            shared.state = TrackState.LOST

        if shared.state == TrackState.LOST:
          # Search around last known (Strict ReID to avoid jumping)
          if self.frame_count % 5 == 0:
            res = self.yolo.predict(frame, verbose=False)
            best_b, best_s = None, 1.0
            gate = min(600, 100 + self.loss_frames * 15)
            for det in res[0].boxes:
              b = det.xywh.cpu().numpy()[0]
              db = (int(b[0]-b[2]/2), int(b[1]-b[3]/2), int(b[2]), int(b[3]))
              if np.hypot(b[0]-self.prev_cx, b[1]-self.prev_cy) < gate:
                ds = self.reid.compare(self.reference_feat, self.reid.extract(frame, db))
                # [FIX] Пом'якшений поріг (0.45 замість 0.38), якщо ціль з'явилася майже там само, де зникла (наприклад, біля краю кадру)
                threshold = 0.45 if np.hypot(b[0]-self.prev_cx, b[1]-self.prev_cy) < 150 else 0.38
                if ds < best_s and ds < threshold: best_s, best_b = ds, db
            
            if best_b:
              print(f"[RECOVERY] Target found by YOLO! ReID Score: {best_s:.3f}")
              self.sot_tracker = cv2.TrackerNano_create(self.params)
              bx, by, bw, bh = map(int, best_b)
              bx, by = max(0, bx), max(0, by)
              bw = max(1, min(bw, frame.shape[1] - bx))
              bh = max(1, min(bh, frame.shape[0] - by))
              clean_best = (bx, by, bw, bh)
              try:
                self.sot_tracker.init(frame, clean_best)
                self.raw_bbox = clean_best
                self.prev_cx = clean_best[0]+clean_best[2]/2
                self.prev_cy = clean_best[1]+clean_best[3]/2
                self.prev_area = clean_best[2]*clean_best[3]
                shared.state = TrackState.TRACKING
                self.loss_frames = 0
                success = True
              except:
                success = False
            else:
              self.loss_frames += 1
          
          if shared.state == TrackState.LOST:
            if self.frame_count % 15 == 0:
              print(f"[RECOVERY] Searching... Radius: {min(600, 100 + self.loss_frames * 15)}px")
            if self.loss_frames > 30: # 1 second at 30 fps
              print("[STATUS] Recovery failed. Returning to MOT mode.")
              shared.state = TrackState.SEARCHING
              self.loss_frames = 0
            success = False

        with shared.lock:
          shared.sot_bbox = self.ema.update(self.raw_bbox if success else None)
          if success: shared.last_valid_bbox = shared.sot_bbox
          shared.is_resync = resync

# --- Interaction & Display ---

def mouse_callback(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    with shared.lock:
      if shared.state == TrackState.SEARCHING:
        hit = False
        if shared.results:
          for tid, bbox in shared.results.items():
            if bbox[0] <= x <= bbox[0]+bbox[2] and bbox[1] <= y <= bbox[1]+bbox[3]:
              shared.selected_target = bbox
              hit = True
              break
        if not hit:
          shared.drag_start = (x, y)
          shared.drag_end = (x, y)
      else:
        shared.reset_requested = True
        
  elif event == cv2.EVENT_MOUSEMOVE:
    with shared.lock:
      if shared.drag_start is not None:
        shared.drag_end = (x, y)
        
  elif event == cv2.EVENT_LBUTTONUP:
    with shared.lock:
      if shared.drag_start is not None:
        x0, y0 = shared.drag_start
        x1, y1 = x, y
        
        # Check if it was a drag (not a simple click on empty space)
        if abs(x1 - x0) > 10 and abs(y1 - y0) > 10:
          x_min, y_min = min(x0, x1), min(y0, y1)
          w, h = abs(x1 - x0), abs(y1 - y0)
          shared.selected_target = (x_min, y_min, w, h)
        
        shared.drag_start = None
        shared.drag_end = None

def main(video_source=None, yolo_freq=10):
  if video_source is None:
    video_source = os.path.join(BASE_DIR, "videos", "sample_video.mp4")
  
  # video_source може бути як шляхом до файлу (рядок), так і ID камери (ціле число, наприклад 0)
  cap = cv2.VideoCapture(video_source)
  if not cap.isOpened(): print(f"Error: Could not open video source {video_source}"); return

  worker = TrackingWorker(yolo_freq=yolo_freq)
  worker.start()

  cv2.namedWindow("Final Drone Tracker (Best Algorithms)")
  cv2.setMouseCallback("Final Drone Tracker (Best Algorithms)", mouse_callback)

  target_fps = 30
  frame_delay = 1.0 / target_fps

  while cap.isOpened() and shared.running:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    with shared.lock:
      shared.frame = frame.copy()
      current_state = shared.state
      mot_results = shared.results.copy() if shared.results else {}
      sot_bbox = shared.sot_bbox
      resync = getattr(shared, 'is_resync', False)
      drag_start = getattr(shared, 'drag_start', None)
      drag_end = getattr(shared, 'drag_end', None)
    
    # Rendering
    if current_state == TrackState.SEARCHING:
      for tid, bbox in mot_results.items():
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
               (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (180, 180, 180), 1)
      cv2.putText(frame, "MOT MODE: Select Target or Draw Box", (10, 30), 1, 1.5, (255, 255, 255), 2)
      
      if drag_start is not None and drag_end is not None:
        cv2.rectangle(frame, drag_start, drag_end, (0, 255, 255), 2, cv2.LINE_AA)
    else:
      # Ghost Box for LOST state
      if current_state == TrackState.LOST and shared.last_valid_bbox:
        gx, gy, gw, gh = map(int, shared.last_valid_bbox)
        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "LAST KNOWN", (gx, gy-10), 1, 1.0, (0, 0, 255), 1)

      if sot_bbox:
        x, y, w, h = map(int, sot_bbox)
        color = (0, 255, 0) if current_state == TrackState.TRACKING else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        if resync: 
          cv2.putText(frame, "HYBRID RE-SYNC", (x, y-15), 1, 1.2, (0, 255, 255), 2)
      
      status_color = (0, 255, 0) if current_state == TrackState.TRACKING else (0, 0, 255)
      cv2.putText(frame, f"HYBRID: {current_state.value}", (10, 30), 1, 1.5, status_color, 2)

    cv2.imshow("Final Drone Tracker (Best Algorithms)", frame)
    
    # FPS Lock
    elapsed = time.time() - t0
    wait = max(1, int((frame_delay - elapsed) * 1000))
    if cv2.waitKey(wait) & 0xFF == ord('q'): shared.running = False
    
    # Check if window was closed by the user (clicking 'X')
    if cv2.getWindowProperty("Final Drone Tracker (Best Algorithms)", cv2.WND_PROP_VISIBLE) < 1:
      shared.running = False

  shared.running = False
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Hybrid SOT/MOT Tracker")
  parser.add_argument("--source", type=str, default=None, help="Video source: path to video file or camera ID (e.g., 0)")
  parser.add_argument("--freq", type=int, default=10, help="YOLO call frequency in frames (default: 10)")
  args = parser.parse_args()
  
  source = args.source
  if source is not None and source.isdigit():
    source = int(source)
    
  main(video_source=source, yolo_freq=args.freq)
