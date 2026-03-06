"""
Basketball pose estimation + ByteTrack player tracking
Uses RTMDet-m (detection) + RTMPose-m (17 COCO keypoints) + supervision ByteTrack

Usage: python run_mmpose_track.py <video_path>
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4 --debug
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4 --debug --verbose
       python run_mmpose_track.py ../data/videos/djurgarden1.mp4 --court
"""
import argparse
import os
from pathlib import Path
import time

from dotenv import load_dotenv

import cv2
import numpy as np
import torch
import supervision as sv
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


# ── Court filtering helpers ───────────────────────────────────────────────────

def _load_court_model():
    """Load Roboflow court keypoint model + court config. Returns (model, court_config)."""
    load_dotenv(Path(__file__).parent.parent / '.env')
    from inference import get_model
    from sports.basketball import CourtConfiguration, League
    from sports import MeasurementUnit

    model = get_model(model_id='basketball-court-detection-2-nsbav/3')
    config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
    return model, config


def _compute_court_polygon(frame, model, court_config):
    """
    Returns court boundary as (N,1,2) int32 array in pixel coords, or None if failed.
    """
    import supervision as sv
    result = model.infer(frame, confidence=0.3)[0]
    kp = sv.KeyPoints.from_inference(result)
    if kp.xy.shape[1] == 0:
        return None

    mask = kp.confidence[0] > 0.5
    frame_pts = kp.xy[0][mask]                           # (K, 2) in pixels
    court_pts = np.array(court_config.vertices)[mask]    # (K, 2) in feet

    if mask.sum() >= 4:
        # Inverse homography: court feet → frame pixels
        H, _ = cv2.findHomography(
            court_pts.astype(np.float32),
            frame_pts.astype(np.float32)
        )
        if H is None:
            hull = cv2.convexHull(frame_pts.astype(np.float32))
            return hull.astype(np.int32)

        # Project all court vertices into image space
        all_court_verts = np.array(court_config.vertices, dtype=np.float32)
        all_court_verts = all_court_verts.reshape(-1, 1, 2)
        image_verts = cv2.perspectiveTransform(all_court_verts, H)
        hull = cv2.convexHull(image_verts.reshape(-1, 2))
        return hull.astype(np.int32)

    elif mask.sum() >= 3:
        hull = cv2.convexHull(frame_pts.astype(np.float32))
        return hull.astype(np.int32)

    return None


def _filter_by_court(xyxy, conf, polygon):
    """Keep only detections whose foot point is inside the court polygon."""
    if polygon is None or len(xyxy) == 0:
        return xyxy, conf
    foot_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    foot_y = xyxy[:, 3]
    keep = np.array([
        cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
        for x, y in zip(foot_x, foot_y)
    ], dtype=bool)
    return xyxy[keep], conf[keep]


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

    Kalman state: [cx, cy, a, h, vcx, vcy, va, vh]
    """
    cx = mean[0] + mean[4]
    cy = mean[1] + mean[5]
    a  = mean[2] + mean[6]
    h  = mean[3] + mean[7]
    h  = max(float(h), 1e-6)
    w  = float(a) * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _snapshot_tracks(tracks):
    """Capture per-track debug info before update."""
    info = {}
    for t in tracks:
        eid = t.external_track_id
        if t.mean is None:
            continue
        info[eid] = {
            'tlbr':      t.tlbr.copy(),
            'pred_tlbr': _pred_tlbr(t.mean),
            'vel':       t.mean[4:6].copy(),
            'state':     t.state.name,
            'last_seen': t.frame_id,
            'age':       t.frame_id - t.start_frame,
        }
    return info


def _print_debug(frame_count, fps, xyxy, conf, pre_snap, pre_tracked_ids,
                 tracker, verbose):
    """Print per-frame debug block after tracker.update_with_detections()."""
    post_tracked_ids = {t.external_track_id for t in tracker.tracked_tracks} - {-1}
    new_track_ids    = post_tracked_ids - pre_tracked_ids
    newly_lost_ids   = pre_tracked_ids - post_tracked_ids

    interesting = bool(new_track_ids or newly_lost_ids)
    if not interesting and not verbose:
        return

    t_s = frame_count / fps
    total_ids = getattr(tracker.external_id_counter, '_count',
                getattr(tracker.external_id_counter, '_id', '?'))

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Basketball pose estimation + ByteTrack tracking')
    parser.add_argument('video_path', type=Path,
                        help='Path to input video')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug info when a track is created or lost')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full per-frame IoU matrix (implies --debug)')
    parser.add_argument('--court', action='store_true',
                        help='Filter detections to inside-court area using Roboflow keypoint model')
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

    # ── ByteTrack init ────────────────────────────────────────────────────────
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,  # restored: court filtering handles noise suppression
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,   # restored
        frame_rate=30,
    )

    # ── Court model init ──────────────────────────────────────────────────────
    if args.court:
        print("Loading court keypoint model...")
        court_model, court_cfg = _load_court_model()
        court_polygon = None           # computed on first frame
        COURT_UPDATE_INTERVAL = 150    # refresh every ~6s at 30fps
    else:
        court_model = court_cfg = court_polygon = None

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

        # ── Build sv.Detections from MMPose output ────────────────────────────
        xyxy = np.array([inst['bbox'][0] for inst in instances], dtype=np.float32)  # (N, 4)
        conf = np.array([inst['bbox_score'] for inst in instances], dtype=np.float32)  # (N,)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            writer.write(annotated)
            continue

        detections = sv.Detections(xyxy=xyxy, confidence=conf)

        # ── Court polygon refresh ─────────────────────────────────────────────
        if args.court and (court_polygon is None or frame_count % COURT_UPDATE_INTERVAL == 1):
            new_poly = _compute_court_polygon(frame_rgb, court_model, court_cfg)
            if new_poly is not None:
                court_polygon = new_poly
                if debug:
                    print(f"  [court] polygon updated ({len(new_poly)} pts)")
            elif court_polygon is None and debug:
                print(f"  [court] WARNING: no polygon on frame {frame_count}")

        # ── Filter detections to court area ───────────────────────────────────
        if args.court and court_polygon is not None:
            xyxy, conf = _filter_by_court(xyxy, conf, court_polygon)
            if len(xyxy) == 0:
                if args.court and court_polygon is not None:
                    cv2.polylines(annotated, [court_polygon], isClosed=True,
                                  color=(0, 255, 0), thickness=2)
                writer.write(annotated)
                continue
            detections = sv.Detections(xyxy=xyxy, confidence=conf)

        # ── Debug: snapshot tracker state before update ───────────────────────
        if debug:
            pre_tracked_ids = {t.external_track_id for t in tracker.tracked_tracks} - {-1}
            pre_snap = _snapshot_tracks(
                list(tracker.tracked_tracks) + list(tracker.lost_tracks))

        # ── ByteTrack update ──────────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections)

        # ── Debug: print events after update ──────────────────────────────────
        if debug:
            _print_debug(frame_count, fps, xyxy, conf, pre_snap,
                         pre_tracked_ids, tracker, verbose)

        # ── Draw tracking boxes + IDs on top of MMPose skeleton frame ────────
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue

            color = PALETTE[tid % len(PALETTE)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Draw court outline for visual verification ────────────────────────
        if args.court and court_polygon is not None:
            cv2.polylines(annotated, [court_polygon], isClosed=True,
                          color=(0, 255, 0), thickness=2)

        writer.write(annotated)

        if frame_count % 30 == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_count}/{total_frames}  {frame_count/elapsed:.1f} fps  "
                  f"active tracks: {len(tracked)}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\nDone! {frame_count} frames in {elapsed:.1f}s  ({frame_count/elapsed:.1f} fps avg)")
    print(f"Output: {out_path.resolve()}")


if __name__ == '__main__':
    main()
