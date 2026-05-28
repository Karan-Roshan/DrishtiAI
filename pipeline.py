"""
Dashcam Video Analysis Pipeline  —  LIVE VERSION
==================================================
Writes live_status.json every 2 seconds so dashcam_dashboard.html
can display real-time progress while the pipeline runs.

QUICK START:
  1. Set VIDEO_SOURCE below (or pass --video flag)
  2. Run:  python3 pipeline.py
  3. Open dashcam_dashboard.html in your browser
  4. Watch results appear live!

Install:
  pip3 install ultralytics deep-sort-realtime opencv-python numpy torch torchvision yt-dlp gdown
"""

import os, sys, cv2, json, re, time, threading
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

# ─── Graceful imports ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] pip3 install ultralytics")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("[WARN] pip3 install deep-sort-realtime")

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("[WARN] pip3 install yt-dlp")

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  👇  SET YOUR VIDEO SOURCE HERE  👇
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_SOURCE = "dashcam.mp4"   # ← local file, YouTube URL, or Drive link

# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_JSON      = "results.json"
LIVE_STATUS_JSON = "live_status.json"   # dashboard polls this
DOWNLOADED       = "dashcam_input.mp4"

# COCO class IDs
PERSON_ID        = 0
BICYCLE_ID       = 1
CAR_ID           = 2
MOTORCYCLE_ID    = 3
BUS_ID           = 5
TRUCK_ID         = 7
TRAFFIC_LIGHT_ID = 9
CELL_PHONE_ID    = 67
COCO_VEHICLE_IDS = {BICYCLE_ID, CAR_ID, MOTORCYCLE_ID, BUS_ID, TRUCK_ID}

CONF_PERSON  = 0.40
CONF_PHONE   = 0.35
CONF_HELMET  = 0.40


# ─── Helpers ──────────────────────────────────────────────────────────────────
def frame_to_ts(fidx, fps):
    return str(timedelta(seconds=int(fidx / fps)))

def compute_iou(a, b):
    xi1, yi1 = max(a[0],b[0]), max(a[1],b[1])
    xi2, yi2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def is_url(s):
    return s.startswith("http://") or s.startswith("https://")

def write_json_safe(path, data):
    """Write JSON atomically so the browser never reads a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)


# ─── Video downloader ─────────────────────────────────────────────────────────
def download_video(url, out=DOWNLOADED):
    if os.path.exists(out):
        print(f"[INFO] Using cached: {out}")
        return out
    print(f"[INFO] Downloading: {url}")
    if "drive.google.com" in url and GDOWN_AVAILABLE:
        m = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if m:
            gdown.download(f"https://drive.google.com/uc?id={m.group(1)}", out, quiet=False)
            if os.path.exists(out): return out
    if YTDLP_AVAILABLE:
        opts = {'outtmpl': out, 'format': 'best[ext=mp4]/best', 'merge_output_format': 'mp4', 'noplaylist': True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        for c in [out, out+'.mp4']:
            if os.path.exists(c): return c
    raise RuntimeError("Download failed — check link is public and yt-dlp is installed.")


# ─── Detectors ────────────────────────────────────────────────────────────────
class HelmetDetector:
    def has_helmet(self, frame, box):
        x1,y1,x2,y2 = [int(v) for v in box]
        crop = frame[y1:y1+max(1,int((y2-y1)*0.28)), x1:x2]
        if crop.size == 0: return False, 0.0
        sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:,:,1].mean()
        return sat < 80, float(np.clip(1.0 - sat/150.0, 0, 1))

class JunctionDetector:
    COOLDOWN = 60
    def __init__(self):
        self.prev_gray = None
        self.cooldown  = 0
    def detect(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray; return False, ""
        if self.cooldown > 0:
            self.cooldown -= 1; self.prev_gray = gray; return False, ""
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        self.prev_gray = gray
        fx = flow[:,:,0]; w = fx.shape[1]
        spread = abs(fx[:,:w//3].mean() - fx[:,2*w//3:].mean())
        if spread > 1.8:
            self.cooldown = self.COOLDOWN
            if   spread > 4.0: return True, "4-Way / X-Junction"
            elif spread > 3.0: return True, "T-Junction"
            elif spread > 2.5: return True, "Roundabout / Traffic Circle"
            elif spread > 2.0: return True, "Y-Junction"
            else:              return True, "T-Junction"
        return False, ""

class WrongSideDetector:
    def detect(self, flow, box):
        x1,y1,x2,y2 = [int(v) for v in box]
        r = flow[y1:y2, x1:x2, 0]
        if r.size == 0: return False, 0.0
        mfx = r.mean()
        return (True, float(np.clip(mfx/8.0,0,1))) if mfx > 2.5 else (False, 0.0)


# ─── Pipeline ─────────────────────────────────────────────────────────────────
class DashcamPipeline:

    def __init__(self, video_path, output_path=OUTPUT_JSON):
        self.video_path  = video_path
        self.output_path = output_path
        self.results     = self._blank()
        self.prev_gray   = None
        self.tracked_ids = set()

        # Live state (written to live_status.json every 2 s)
        self._live = {
            "status": "starting",
            "progress_pct": 0,
            "current_ts": "0:00:00",
            "fps": 30,
            "total_frames": 0,
            "duration": "",
            "current_vehicle_count": 0,
            "total_violations": 0,
            "junctions": 0,
            "vehicles_unique": 0,
            "violation_counts": {},
            "vehicle_categories": {},
            "junction_breakdown": {},
            "density_over_time": [],
            "recent_events": [],    # last 50 events for live log
        }
        self._live_lock   = threading.Lock()
        self._recent_evts = []      # rolling buffer

        print("[INFO] Loading YOLOv8x model…")
        self.model   = YOLO("yolov8x.pt") if YOLO_AVAILABLE else None
        self.tracker = DeepSort(max_age=30, n_init=3) if DEEPSORT_AVAILABLE else None
        self.jdet    = JunctionDetector()
        self.wsdet   = WrongSideDetector()
        self.hdet    = HelmetDetector()

    def _blank(self):
        return {
            "metadata": {"video": self.video_path, "fps":0, "total_frames":0, "duration":""},
            "violations": {k: {"count":0,"instances":[],"top3_detections":[]}
                           for k in ["helmetless","wrong_side","signal_jump","phone_use","triple_ride"]},
            "junctions": {"total":0, "breakdown":{}, "events":[]},
            "vehicles":  {"total_unique":0,
                          "categories":{"two_wheeler":0,"lmv":0,"hmv":0,"others":0},
                          "density_over_time":[]},
        }

    def _classify(self, cls_id):
        return {MOTORCYCLE_ID:"two_wheeler", BICYCLE_ID:"others",
                BUS_ID:"hmv", TRUCK_ID:"hmv", CAR_ID:"lmv"}.get(cls_id, "lmv")

    def _record(self, vtype, fidx, fps, box, conf):
        ts    = frame_to_ts(fidx, fps)
        entry = {"timestamp":ts, "frame":fidx,
                 "bbox":[round(v) for v in box], "confidence":round(conf,3)}
        v = self.results["violations"][vtype]
        v["count"] += 1
        v["instances"].append(entry)
        v["top3_detections"] = sorted(v["top3_detections"]+[entry],
                                      key=lambda x:x["confidence"], reverse=True)[:3]
        # Live event
        with self._live_lock:
            self._recent_evts.append(
                {"type":"violation","subtype":vtype,"timestamp":ts,
                 "frame":fidx,"confidence":round(conf,3)})
            self._recent_evts = self._recent_evts[-100:]

    def _write_live(self, fidx, fps, total_frames, vehicles_now):
        pct = round(100 * fidx / max(total_frames,1), 2)
        viol_counts = {k: self.results["violations"][k]["count"]
                       for k in self.results["violations"]}
        total_v = sum(viol_counts.values())
        with self._live_lock:
            self._live.update({
                "status": "processing",
                "progress_pct": pct,
                "current_ts": frame_to_ts(fidx, fps),
                "fps": round(fps,2),
                "total_frames": total_frames,
                "duration": frame_to_ts(total_frames, fps),
                "current_vehicle_count": vehicles_now,
                "total_violations": total_v,
                "junctions": self.results["junctions"]["total"],
                "vehicles_unique": len(self.tracked_ids),
                "violation_counts": viol_counts,
                "vehicle_categories": dict(self.results["vehicles"]["categories"]),
                "junction_breakdown": dict(self.results["junctions"]["breakdown"]),
                "density_over_time": self.results["vehicles"]["density_over_time"][-60:],
                "recent_events": list(self._recent_evts[-50:]),
            })
            snap = dict(self._live)
        write_json_safe(LIVE_STATUS_JSON, snap)

    # periodic writer thread
    def _start_live_writer(self, fps, total_frames):
        def _loop():
            while self._writing_live:
                self._write_live(self._fidx, fps, total_frames,
                                 self._cur_veh_count)
                time.sleep(2)
        self._writing_live   = True
        self._fidx           = 0
        self._cur_veh_count  = 0
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t

    def run(self):
        if not self.model:
            print("[ERROR] ultralytics not installed."); return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self.video_path}"); return

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = frame_to_ts(total_frames, fps)
        self.results["metadata"].update({"fps":round(fps,2),
                                         "total_frames":total_frames,
                                         "duration":duration})
        print(f"[INFO] {total_frames} frames  @  {fps:.1f} fps  —  {duration}")
        print(f"[INFO] Open dashcam_dashboard.html in your browser to see live results!")

        # Write initial live file so dashboard shows immediately
        write_json_safe(LIVE_STATUS_JSON, {**self._live,
                        "fps":fps,"total_frames":total_frames,"duration":duration})

        writer_thread = self._start_live_writer(fps, total_frames)

        SKIP      = 3
        red_light = False
        fidx      = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            self._fidx = fidx          # shared with writer thread

            if fidx % SKIP != 0:
                fidx += 1; continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_fr  = frame.shape[0]

            flow = None
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
            self.prev_gray = gray

            det = self.model(frame, verbose=False, conf=0.35)[0]
            vehicles, motos, persons_on_moto = [], [], defaultdict(list)

            for box in det.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                if cls == TRAFFIC_LIGHT_ID:
                    crop = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    if crop.size > 0:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        red_light = bool(cv2.inRange(hsv,(0,70,70),(10,255,255)).mean() > 10)

                if cls in COCO_VEHICLE_IDS:
                    vehicles.append((xyxy, conf, cls, self._classify(cls)))
                    if cls == MOTORCYCLE_ID: motos.append(xyxy)

                if cls == CELL_PHONE_ID and conf > CONF_PHONE:
                    self._record("phone_use", fidx, fps, xyxy, conf)

                if cls == PERSON_ID and conf > CONF_PERSON:
                    for mb in motos:
                        if compute_iou(xyxy, mb) > 0.15:
                            persons_on_moto[tuple(mb)].append(xyxy)
                    if red_light and xyxy[3] > h_fr * 0.85:
                        self._record("signal_jump", fidx, fps, xyxy, conf)

            for mb, riders in persons_on_moto.items():
                if len(riders) >= 3:
                    self._record("triple_ride", fidx, fps, list(mb),
                                 0.75 + 0.05*min(len(riders),3))
                for rb in riders:
                    has_h, hc = self.hdet.has_helmet(frame, rb)
                    if not has_h and hc > CONF_HELMET:
                        self._record("helmetless", fidx, fps, rb, hc)

            if flow is not None:
                for (xyxy,conf,cls,_) in vehicles:
                    if cls in (CAR_ID,BUS_ID,TRUCK_ID):
                        ws,wc = self.wsdet.detect(flow, xyxy)
                        if ws and wc > 0.5:
                            self._record("wrong_side", fidx, fps, xyxy, wc)

            if self.tracker and vehicles:
                dets = [([x1,y1,x2-x1,y2-y1],c,cat)
                        for (x1,y1,x2,y2),c,_,cat in vehicles]
                for trk in self.tracker.update_tracks(dets, frame=frame):
                    if trk.is_confirmed() and trk.track_id not in self.tracked_ids:
                        self.tracked_ids.add(trk.track_id)
                        cat = getattr(trk,'det_class','lmv')
                        if isinstance(cat,str) and cat in self.results["vehicles"]["categories"]:
                            self.results["vehicles"]["categories"][cat] += 1

            junc, jtype = self.jdet.detect(gray)
            if junc:
                ts = frame_to_ts(fidx, fps)
                bd = self.results["junctions"]["breakdown"]
                bd[jtype] = bd.get(jtype,0) + 1
                self.results["junctions"]["total"] += 1
                self.results["junctions"]["events"].append(
                    {"timestamp":ts,"frame":fidx,"type":jtype})
                with self._live_lock:
                    self._recent_evts.append(
                        {"type":"junction","subtype":jtype,
                         "timestamp":ts,"frame":fidx,"confidence":None})

            self._cur_veh_count = len(vehicles)
            if fidx % int(fps) == 0:
                self.results["vehicles"]["density_over_time"].append(
                    {"timestamp":frame_to_ts(fidx,fps),"count":len(vehicles)})

            fidx += 1
            if fidx % 300 == 0:
                pct = 100*fidx/max(total_frames,1)
                print(f"[INFO] {pct:5.1f}%  —  {frame_to_ts(fidx,fps)}")

        # ── Done ──────────────────────────────────────────────────────────
        cap.release()
        self._writing_live = False
        self.results["vehicles"]["total_unique"] = len(self.tracked_ids)

        # Final save
        write_json_safe(self.output_path, self.results)

        # Final live_status with done flag
        viol_counts = {k:self.results["violations"][k]["count"]
                       for k in self.results["violations"]}
        final_live = {
            "status": "done",
            "progress_pct": 100,
            "current_ts": duration,
            "fps": round(fps,2),
            "total_frames": total_frames,
            "duration": duration,
            "current_vehicle_count": 0,
            "total_violations": sum(viol_counts.values()),
            "junctions": self.results["junctions"]["total"],
            "vehicles_unique": len(self.tracked_ids),
            "violation_counts": viol_counts,
            "vehicle_categories": dict(self.results["vehicles"]["categories"]),
            "junction_breakdown": dict(self.results["junctions"]["breakdown"]),
            "density_over_time": self.results["vehicles"]["density_over_time"],
            "recent_events": list(self._recent_evts[-50:]),
        }
        write_json_safe(LIVE_STATUS_JSON, final_live)

        tv = sum(viol_counts.values())
        print(f"\n{'='*50}")
        print(f"  DONE  →  {self.output_path}")
        print(f"  Violations : {tv}")
        print(f"  Junctions  : {self.results['junctions']['total']}")
        print(f"  Vehicles   : {len(self.tracked_ids)}")
        print(f"{'='*50}")
        return self.results


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default=None)
    parser.add_argument("--output", default=OUTPUT_JSON)
    args = parser.parse_args()

    source = args.video or VIDEO_SOURCE

    if is_url(source):
        try:
            source = download_video(source, DOWNLOADED)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}"); sys.exit(1)

    if not os.path.exists(source):
        print(f"[ERROR] File not found: {source}"); sys.exit(1)

    DashcamPipeline(source, args.output).run()
