"""
Basketball pose estimation — OpenMMLab RTMPose + RTMDet (MPS accelerated)
Usage: python run_mmpose.py <video_path>
       python run_mmpose.py ../data/videos/djurgarden1.mp4
"""
import sys
from pathlib import Path
import torch
from mmpose.apis import MMPoseInferencer


def _patch_for_mps():
    """Apply two patches needed to run mmcv/mmengine inference on MPS:

    1. mmcv NMSop: its C++ op has no MPS backend — offload just the NMS call
       to CPU and return indices on the original MPS device.
    2. mmengine InstanceData: IndexType only lists CPU/CUDA tensor subclasses;
       patch it to also accept MPS LongTensor / BoolTensor so that indexing
       detection results with MPS indices doesn't hit an AssertionError.
    """
    import typing
    import numpy as np
    import mmengine.structures.instance_data as _idata
    from mmcv.ops.nms import NMSop, ext_module

    # ── Patch 1: NMS offload to CPU ──────────────────────────────────────────
    @staticmethod
    def mps_safe_forward(ctx, bboxes, scores, iou_threshold, offset,
                         score_threshold, max_num):
        device = bboxes.device
        # ext_module.nms signature: (boxes, scores, iou_threshold, offset)
        inds = ext_module.nms(bboxes.cpu(), scores.cpu(), iou_threshold, offset)
        return inds.to(device)

    NMSop.forward = mps_safe_forward

    # ── Patch 2: InstanceData IndexType — add MPS tensor classes ─────────────
    _mps_long = type(torch.zeros(1, device='mps', dtype=torch.long))
    _mps_bool = type(torch.zeros(1, device='mps', dtype=torch.bool))
    _idata.IndexType = typing.Union[
        str, slice, int, list,
        torch.LongTensor, torch.cuda.LongTensor,
        torch.BoolTensor, torch.cuda.BoolTensor,
        _mps_long, _mps_bool,
        np.ndarray,
    ]


def main():
    # ── Video path ────────────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python run_mmpose.py <video_path>")
        print("Example: python run_mmpose.py ../data/videos/djurgarden1.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: video not found at {video_path}")
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("  Warning: MPS not available — falling back to CPU (will be slow)")
    else:
        _patch_for_mps()
        print("  Applied MPS patches (NMS offloaded to CPU; InstanceData MPS support)")

    # ── Model init ────────────────────────────────────────────────────────────
    print("Initializing RTMDet-nano (detection) + RTMPose-m (pose)...")
    print("  Downloading weights if not cached (~50 MB total)...")
    inferencer = MMPoseInferencer(
        pose2d='human',   # RTMPose-m: 17 COCO body keypoints
        device=device
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    out_dir = 'output'
    print(f"\nProcessing: {video_path}")
    print(f"  kpt_thr=0.3 — lower threshold for partially occluded players")
    print(f"  Output will be saved to: {out_dir}/visualizations/")
    print("  Press Q in the preview window to quit early\n")

    result_generator = inferencer(
        inputs=str(video_path),
        show=True,
        save_predictions=False,   # skip JSON keypoint dumps
        out_dir=out_dir,
        kpt_thr=0.3,
    )

    frame_count = 0
    for _ in result_generator:    # must consume — lazy generator
        frame_count += 1

    out_path = (Path(out_dir) / 'visualizations').resolve()
    print(f"\nDone! Processed {frame_count} frames.")
    print(f"Annotated video saved to:\n  {out_path}")


if __name__ == '__main__':
    main()
