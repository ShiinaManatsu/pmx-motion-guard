"""Blender-independent collision screening and parallel contextual solving.

Inputs are a fixed Character pack and local bind-delta rotation matrices.
Callers must validate their evaluated rig against this fixed-pack FK model.
Every worker owns its native packs and stateful Guard instance.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import threading
import time

import numpy as np

from .vendor.retarget_guard import RetargetGuard
from .vendor.retarget_guard._native import Pack
from .vendor.retarget_guard.streaming import frozen_settings


def collision_scan(character, poses, threshold=.002, cancel=None):
    """Return conservative sampled contact flags and minimum native gaps.

    Native signed sample gaps are a broad screening signal. They do not
    certify intersection-free full meshes or closed volumes.
    """
    gaps = np.empty(len(poses), dtype=np.float64)
    with_pack = Pack(character)
    try:
        for index, pose in enumerate(poses):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Sequence scan cancelled")
            gaps[index] = np.min(with_pack.geometry(pose, rho=0)[3])
    finally:
        with_pack.close()
    return gaps <= threshold, gaps


def contact_segments(flags, padding=15, merge_gap=15):
    """Merge nearby hits and expand by context, returning exclusive bounds."""
    if padding < 0 or merge_gap < 0:
        raise ValueError("Context padding and merge gap must be nonnegative")
    hits = np.flatnonzero(flags)
    if not len(hits):
        return []
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > merge_gap + 1) + 1)
    segments = []
    for group in groups:
        start = max(0, int(group[0]) - padding)
        end = min(len(flags), int(group[-1]) + padding + 1)
        if segments and start <= segments[-1][1]:
            segments[-1] = (segments[-1][0], end)
        else:
            segments.append((start, end))
    return segments


def solve_segment(character, poses, fps, lookahead, enabled, cancel, settings=None):
    output = []
    deltas = []
    with RetargetGuard(character, character, lookahead=lookahead, fps=fps,
                       settings=settings or frozen_settings(source_allow=0)) as guard:
        if enabled is not None:
            mask = np.ascontiguousarray(enabled, dtype=np.int32)
            if mask.shape != (21,):
                raise ValueError("Expected 21 enabled joint flags")
            native = guard._guard
            native.lib.guard_set_mask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            native.lib.guard_set_mask.restype = None
            native.lib.guard_set_mask(native.handle, mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        for pose in poses:
            if cancel.is_set():
                raise InterruptedError("Sequence solve cancelled")
            result = guard.step(pose)
            if result is not None:
                output.append(result)
                deltas.append(guard.last_delta.copy())
        for result in guard.flush():
            if cancel.is_set():
                raise InterruptedError("Sequence solve cancelled")
            output.append(result)
            deltas.append(guard.last_delta.copy())
    return np.asarray(output), np.asarray(deltas)


def solve_sequence(character, poses, fps=30, workers=1, lookahead=5,
                   padding=30, merge_gap=30, threshold=.002,
                   enabled=None, cancel=None, screen=False, boundary_tolerance=.001):
    """Screen frames and solve separate contact segments concurrently.

    ctypes CDLL releases the GIL during the native calls. Each thread owns
    separate C allocations. Blender data must be extracted by the caller.
    Screening and finite segment boundaries need full-stream comparison on
    each supported rig before enabling selective processing by default.
    """
    poses = np.array(poses, dtype=np.float64, copy=True)
    if poses.ndim != 4 or poses.shape[1:] != (character.nj, 3, 3):
        raise ValueError("Expected pose sequence [frames,joints,3,3]")
    if not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Expected a nonempty finite sequence")
    if not 0 <= lookahead <= 5 or workers < 1:
        raise ValueError("Invalid lookahead or worker count")
    poses.flags.writeable = False
    cancel = cancel or threading.Event()
    started = time.perf_counter()
    if screen:
        flags, gaps = collision_scan(character, poses, threshold, cancel)
        segments = contact_segments(flags, padding, merge_gap)
    else:
        flags = np.ones(len(poses), bool)
        gaps = None
        segments = [(0, len(poses))]
    scan_seconds = time.perf_counter() - started
    output = poses.copy()
    corrections = np.zeros((len(poses), 21, 3))
    # Compile/load before worker startup to avoid concurrent initial linking.
    from .vendor.retarget_guard._build import library_path
    library_path("guard_core")
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(segments)))) as pool:
        futures = [(start, end, pool.submit(solve_segment, character, poses[start:end],
                    fps, lookahead, enabled, cancel)) for start, end in segments]
        try:
            for start, end, future in futures:
                solved, delta = future.result()
                output[start:end] = solved
                corrections[start:end] = delta
        except BaseException:
            cancel.set()
            for _, _, future in futures:
                future.cancel()
            raise
    # Expose boundary residuals for verification rather than hiding jumps.
    boundary_max = max((float(np.linalg.norm(corrections[index], axis=-1).max())
                        for start, end in segments for index in (start, end - 1)
                        if index not in (0, len(poses) - 1)), default=0.)
    fallback = boundary_max > boundary_tolerance
    if fallback:
        # Preserve the stream's temporal history when the proposed cuts still
        # contain a material correction. The full pass is the reference path.
        output, _ = solve_segment(character, poses, fps, lookahead, enabled, cancel)
    return output, dict(scan_seconds=scan_seconds,
                       solve_seconds=time.perf_counter() - started - scan_seconds,
                       frames=len(poses), candidate_frames=int(flags.sum()),
                       solved_frames=sum(end - start for start, end in segments),
                       segments=segments, workers=min(workers, len(segments)),
                       boundary_correction_radians=boundary_max,
                       full_stream_fallback=fallback,
                       min_gap_m=float(gaps.min()) if gaps is not None else None)
