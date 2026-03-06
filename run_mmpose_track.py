"""
Basketball pose estimation + BoT-SORT player tracking
Uses RTMDet-m (detection) + RTMPose-m (17 COCO keypoints) + ultralytics BoT-SORT

Usage: python run_mmpose_track.py <video_path>
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4 --debug
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4 --debug --verbose
"""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import IterableSimpleNamespace
from mmpose.apis import MMPoseInferencer


def _patch_for_mps():
    """Apply two patches needed to run mmcv/mmengine inference on MPS:

    1. mmcv NMSop: its C++ op has no MPS backend — offload NMS to CPU.
    2. mmengine InstanceData: patch IndexType to accept MPS tensors.
    """
    import typing
    import mmengine.structures.instance_data as _idata
    from mmcv.ops.nms import NMSop, ext_module

    @staticmethod
    def mps_safe_forward(ctx, bboxes, scores, iou_threshold, offset,
                         score_threshold, max_num):
        device = bboxes.device
        inds = ext_module.nms(bboxes.cpu(), scores.cpu(), iou_threshold, offset)
        return inds.to(device)

    NMSop.forward = mps_safe_forward

    _mps_long = type(torch.zeros(1, device='mps', dtype=torch.long))
    _mps_bool = type(torch.zeros(1, device='mps', dtype=torch.bool))
    _idata.IndexType = typing.Union[
        str, slice, int, list,
        torch.LongTensor, torch.cuda.LongTensor,
        torch.BoolTensor, torch.cuda.BoolTensor,
        _mps_long, _mps_bool,
        np.ndarray,
    ]


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _bbox_iou(box, boxes):
    """IoU between one box (4,) and many boxes (N, 4). All xyxy format."""
    boxes = np.atleast_2d(boxes)
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def _pred_tlbr(mean):
    """Apply 1-step constant-velocity model to get predicted xyxy.

    Kalman state (KalmanFilterXYWH): [cx, cy, w, h, vcx, vcy, vw, vh]
    """
    cx = mean[0] + mean[4]
    cy = mean[1] + mean[5]
    w  = max(float(mean[2] + mean[6]), 1e-6)
    h  = max(float(mean[3] + mean[7]), 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _snapshot_tracks(tracks):
    """Capture per-track debug info before update."""
    info = {}
    for t in tracks:
        eid = t.track_id
        if t.mean is None:
            continue
        info[eid] = {
            'tlbr':      t.xyxy.copy(),
            'pred_tlbr': _pred_tlbr(t.mean),
            'vel':       t.mean[4:6].copy(),
            'state':     'Tracked' if t.is_activated else 'Lost',
            'last_seen': t.frame_id,
            'age':       t.frame_id - t.start_frame,
        }
    return info


def _print_debug(frame_count, fps, xyxy, conf, pre_snap, pre_tracked_ids,
                 tracker, verbose):
    """Print per-frame debug block after tracker.update()."""
    post_tracked_ids = {t.track_id for t in tracker.tracked_stracks} - {-1}
    new_track_ids    = post_tracked_ids - pre_tracked_ids
    newly_lost_ids   = pre_tracked_ids - post_tracked_ids

    interesting = bool(new_track_ids or newly_lost_ids)
    if not interesting and not verbose:
        return

    t_s = frame_count / fps
    total_ids = BaseTrack._count

    print(f"\n=== Frame {frame_count} (t={t_s:.1f}s) ===")

    if verbose:
        # Detections
        conf_str = ', '.join(f'{c:.2f}' for c in conf)
        print(f"Detections ({len(xyxy)}):  conf=[{conf_str}]")
        for i, (box, c) in enumerate(zip(xyxy, conf)):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            print(f"  [{i}] conf={c:.2f}  center=({cx:.0f},{cy:.0f}) "
                  f"xyxy=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

        # Active tracks (pre-update)
        active_snap = {eid: v for eid, v in pre_snap.items() if v['state'] == 'Tracked'}
        print(f"\nActive tracks ({len(active_snap)}):")
        for eid, info in sorted(active_snap.items()):
            pc = info['pred_tlbr']
            pc_cx = (pc[0] + pc[2]) / 2
            pc_cy = (pc[1] + pc[3]) / 2
            vx, vy = info['vel']
            print(f"  #{eid:<4}  state=Tracked  pred_center=({pc_cx:.0f},{pc_cy:.0f})"
                  f"  vel=({vx:+.1f},{vy:+.1f})"
                  f"  last_seen={info['last_seen']}  age={info['age']}")

        lost_snap = {eid: v for eid, v in pre_snap.items() if v['state'] == 'Lost'}
        if lost_snap:
            print(f"\nLost tracks ({len(lost_snap)}):")
            for eid, info in sorted(lost_snap.items()):
                pc = info['pred_tlbr']
                pc_cx = (pc[0] + pc[2]) / 2
                pc_cy = (pc[1] + pc[3]) / 2
                print(f"  #{eid:<4}  state=Lost  pred_center=({pc_cx:.0f},{pc_cy:.0f})"
                      f"  last_seen={info['last_seen']}")

        # IoU matrix
        if len(xyxy) > 0 and active_snap:
            print(f"\nIoU (detection vs predicted bbox of active tracks):")
            track_ids   = sorted(active_snap.keys())
            track_boxes = np.array([active_snap[eid]['pred_tlbr'] for eid in track_ids])
            for i, box in enumerate(xyxy):
                ious = _bbox_iou(box, track_boxes)
                for j, (eid, iou) in enumerate(zip(track_ids, ious)):
                    flag = '  ← BELOW 0.8 → will NOT match Stage-1' if iou < 0.8 else ''
                    print(f"  det[{i}] vs #{eid}  → IoU={iou:.2f}{flag}")

    # Post-update summary
    print(f"\nAfter update:")
    print(f"  Active: {sorted(post_tracked_ids)}")

    if new_track_ids:
        print(f"  NEW tracks this frame: {sorted(new_track_ids)}  ← new external IDs assigned")

    if newly_lost_ids:
        for lost_eid in sorted(newly_lost_ids):
            if lost_eid not in pre_snap:
                print(f"  Newly LOST: [#{lost_eid}]  (no pre-frame snapshot)")
                continue
            info = pre_snap[lost_eid]
            pred = info['pred_tlbr']
            pred_cx = (pred[0] + pred[2]) / 2
            pred_cy = (pred[1] + pred[3]) / 2
            if len(xyxy) > 0:
                ious    = _bbox_iou(pred, xyxy)
                best_i  = int(np.argmax(ious))
                best_iou  = float(ious[best_i])
                best_conf = float(conf[best_i])
                print(f"  Newly LOST: [#{lost_eid}]  "
                      f"(predicted=({pred_cx:.0f},{pred_cy:.0f}), "
                      f"nearest det IoU={best_iou:.2f}, "
                      f"nearest det conf={best_conf:.2f})")
            else:
                print(f"  Newly LOST: [#{lost_eid}]  (predicted=({pred_cx:.0f},{pred_cy:.0f}), no detections)")

    print(f"  Total external IDs created so far: {total_ids}")


# ── Detection wrapper ─────────────────────────────────────────────────────────

class _Dets:
    """Minimal subscriptable detections object for BOTSORT.update()."""
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf
        self.cls  = np.zeros(len(conf), dtype=np.float32)
        xywh = xyxy.copy()
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = xywh

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.xyxy[idx], self.conf[idx])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Basketball pose estimation + BoT-SORT tracking')
    parser.add_argument('video_path', type=Path,
                        help='Path to input video')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug info when a track is created or lost')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full per-frame IoU matrix (implies --debug)')
    args = parser.parse_args()

    video_path = args.video_path
    debug   = args.debug or args.verbose
    verbose = args.verbose

    if not video_path.exists():
        print(f"Error: video not found at {video_path}")
        raise SystemExit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'mps':
        _patch_for_mps()
        print("  Applied MPS patches")

    # ── Model init ────────────────────────────────────────────────────────────
    print("Initializing RTMDet-m + RTMPose-m ...")
    inferencer = MMPoseInferencer(pose2d='human', device=device)

    # ── Video I/O ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path('output/visualizations')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = video_path.stem + '_tracked.mp4'
    out_path = out_dir / out_name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # ── BoT-SORT init ────────────────────────────────────────────────────────
    botsort_args = IterableSimpleNamespace(
        track_high_thresh=0.25,      # min conf to promote to active track
        track_low_thresh=0.1,        # min conf for low-score second-stage match
        new_track_thresh=0.4,        # was 0.25 — require more confidence to spawn new track
        track_buffer=90,             # was 30 — keep lost tracks alive ~3.6s at 25fps
        match_thresh=0.7,            # was 0.8 — stricter IoU, reduces cross-player theft
        gmc_method='sparseOptFlow',  # global motion compensation (handles panning)
        proximity_thresh=0.5,        # appearance matching only when IoU dist < 0.5
        appearance_thresh=0.25,      # block embeddings with cosine dist > 0.75
        with_reid=True,              # enable appearance-based ReID
        fuse_score=False,            # do not fuse score into IoU distance
        model='yolo26n-cls.pt',      # explicit nano classifier (~5MB, auto-downloaded)
    )
    tracker = BOTSORT(args=botsort_args, frame_rate=fps)

    print(f"\nProcessing: {video_path.name}  ({w}x{h} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"Output: {out_path}")
    if debug:
        print("Debug mode ON — printing track events (new / lost)")
    print()

    # ── Label annotator colours (one per track ID, cycling) ──────────────────
    PALETTE = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
        (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
        (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
        (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
    ]

    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Pose inference (single frame) — return_vis=True gives MMPose rendering ──
        result_gen = inferencer(inputs=frame_rgb, return_vis=True, kpt_thr=0.3)
        result = next(result_gen)

        instances = result['predictions'][0]  # list of dicts, one per person

        # Use MMPose's own rendered frame (RGB) as the base — identical skeleton style
        # to run_mmpose.py. Convert back to BGR for VideoWriter.
        vis_rgb = result['visualization'][0]   # np.ndarray H×W×3, RGB, uint8
        annotated = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        if not instances:
            writer.write(annotated)
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                print(f"  Frame {frame_count}/{total_frames}  {frame_count/elapsed:.1f} fps")
            continue

        # ── Build detections from MMPose output ───────────────────────────────
        xyxy = np.array([inst['bbox'][0] for inst in instances], dtype=np.float32)  # (N, 4)
        conf = np.array([inst['bbox_score'] for inst in instances], dtype=np.float32)  # (N,)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            writer.write(annotated)
            continue

        dets = _Dets(xyxy, conf)

        # ── Debug: snapshot tracker state before update ───────────────────────
        if debug:
            pre_tracked_ids = {t.track_id for t in tracker.tracked_stracks} - {-1}
            pre_snap = _snapshot_tracks(
                list(tracker.tracked_stracks) + list(tracker.lost_stracks))

        # ── BoT-SORT update ───────────────────────────────────────────────────
        tracks = tracker.update(dets, img=frame_bgr)
        # tracks: np.ndarray shape (M, 8) → [x1,y1,x2,y2, track_id, conf, cls, idx]

        # ── Debug: print events after update ──────────────────────────────────
        if debug:
            _print_debug(frame_count, fps, xyxy, conf, pre_snap,
                         pre_tracked_ids, tracker, verbose)

        # ── Draw tracking boxes + IDs on top of MMPose skeleton frame ────────
        for row in tracks:
            x1, y1, x2, y2, tid = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
            color = PALETTE[tid % len(PALETTE)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)

        if frame_count % 30 == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_count}/{total_frames}  {frame_count/elapsed:.1f} fps  "
                  f"active tracks: {len(tracks)}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\nDone! {frame_count} frames in {elapsed:.1f}s  ({frame_count/elapsed:.1f} fps avg)")
    print(f"Output: {out_path.resolve()}")


if __name__ == '__main__':
    main()
