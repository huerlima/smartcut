"""CFR (constant frame rate) detection and timestamp grid snapping.

This module helps smartcut produce truly CFR output when the input source is
itself CFR. Without this, the seam between copy-mode (raw packet passthrough)
and transcode-mode (re-encoded partial GOPs) can introduce non-uniform sample
deltas in the output `stts` table, causing players/tools to treat the result
as VFR even though every frame is regularly spaced.

Public entry points:
    * detect_typical_frame_duration_pts(...)  -> int | None
    * snap_recoded_packets_to_grid(...)       -> dict | None
    * assert_cfr_output_invariants(...)       -> None  (raises CFRInvariantError)
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# Source is considered CFR if at least this fraction of consecutive PTS
# deltas equals the modal delta. Captures real-world sources where one or
# two frames at the very start/end can have slightly off PTS due to a/v
# muxing quirks but the body is uniform.
_CFR_UNIFORMITY_THRESHOLD = 0.95


def detect_typical_frame_duration_pts(frame_pts: Iterable[int] | np.ndarray) -> int | None:
    """Compute the modal inter-frame PTS delta of a CFR source.

    Returns the typical frame duration in source `time_base` units, or None
    if the source is too non-uniform to be considered CFR.
    """
    pts = np.asarray(list(frame_pts) if not isinstance(frame_pts, np.ndarray) else frame_pts)
    if pts.size < 2:
        return None
    pts = np.sort(pts)
    deltas = np.diff(pts)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    vals, counts = np.unique(deltas, return_counts=True)
    mode_idx = int(counts.argmax())
    mode = int(vals[mode_idx])
    mode_count = int(counts[mode_idx])
    if mode <= 0:
        return None
    if mode_count / deltas.size < _CFR_UNIFORMITY_THRESHOLD:
        return None
    return mode


@dataclass
class _SnapResult:
    snapped: int
    max_correction: int
    typical_delta: int
    anchor_pts: int


class CFRInvariantError(RuntimeError):
    """Raised when output mp4 fails CFR invariants after smartcut + fixup."""


def snap_recoded_packets_to_grid(
    packets: list,
    typical_delta_out: int,
    anchor_pts: int,
    max_correction_frames: float = 4.0,
) -> _SnapResult | None:
    """Rewrite packet PTS/DTS so each packet lands on a uniform grid.

    `packets` is a list of PyAV `Packet` objects produced by an encoder, in
    encoder emit order (which is decode order). Each packet's PTS is snapped
    to the *nearest* grid point of the form
    ``anchor_pts + k * typical_delta_out`` (for integer k). DTS is shifted
    by the same amount as PTS, preserving the per-packet DTS-PTS skew so
    B-frame reorder patterns are unchanged.

    This per-packet snap is robust to B-frames whose batches are split
    across `recode_segment` and `flush_encoder` calls: each packet's grid
    position is determined solely by its own existing PTS, independent of
    the other packets in the same batch.

    `anchor_pts` should be a known grid origin (e.g. the start of the cut
    in output ticks). It does not need to be reachable by any packet.

    Returns a `_SnapResult` describing what was done, or None for no-op
    (empty input, invalid typical delta, or any packet missing PTS).

    Raises `RuntimeError` if any packet would need to move by more than
    `max_correction_frames * typical_delta_out` ticks. This is a safety net
    against silent corruption when the encoder produces output wildly out
    of place (e.g. for a non-CFR source that slipped past detection).
    """
    if not packets or typical_delta_out <= 0:
        return None
    if any(p.pts is None for p in packets):
        return None

    max_correction = 0
    for packet in packets:
        old_pts = int(packet.pts)
        # Round (old_pts - anchor_pts) to the nearest multiple of typical.
        offset = old_pts - anchor_pts
        # Symmetric rounding via floor((offset + typical/2) / typical).
        k = (offset + typical_delta_out // 2) // typical_delta_out
        new_pts = anchor_pts + int(k) * typical_delta_out
        delta = new_pts - old_pts
        max_correction = max(max_correction, abs(delta))
        packet.pts = new_pts
        if packet.dts is not None:
            packet.dts = int(packet.dts) + delta
        packet.duration = typical_delta_out

    correction_limit = int(max_correction_frames * typical_delta_out)
    if max_correction > correction_limit:
        raise RuntimeError(
            f"smartcut: refusing to snap recoded packets to CFR grid: "
            f"max_correction={max_correction} ticks exceeds "
            f"{max_correction_frames}x typical_frame_duration ({typical_delta_out}). "
            f"Source may not be true CFR."
        )

    return _SnapResult(
        snapped=len(packets),
        max_correction=max_correction,
        typical_delta=typical_delta_out,
        anchor_pts=anchor_pts,
    )


# ---------------------------------------------------------------------------
# Post-mux invariant checking. Parses moov/trak/mdia/minf/stbl/stts directly
# so it works without a full ffprobe roundtrip.
# ---------------------------------------------------------------------------

def _find_box(box: str, data: bytes) -> bytes | None:
    pos, end = 0, len(data)
    while pos < end - 8:
        sz = struct.unpack(">I", data[pos : pos + 4])[0]
        tp = data[pos + 4 : pos + 8]
        if sz == 0:
            sz = end - pos
        if sz < 8:
            break
        if tp == box.encode():
            return data[pos + 8 : pos + sz]
        pos += sz
    return None


def _parse_stts(stts: bytes) -> list[tuple[int, int]]:
    cnt = struct.unpack(">I", stts[4:8])[0]
    entries = []
    off = 8
    for _ in range(cnt):
        n, d = struct.unpack(">II", stts[off : off + 8])
        off += 8
        entries.append((n, d))
    return entries


def assert_cfr_output_invariants(
    mp4_path: str,
    expected_typical_delta_out: int,
    raise_on_mismatch: bool = True,
) -> None:
    """Verify that an mp4 output's video track is genuinely CFR.

    Walks the moov atom, extracts the video stts table, and asserts:
        - all stts entries have the same delta (== expected_typical_delta_out)
        - mdhd duration equals sum_of_deltas (sample_count * delta)

    On mismatch, either raises CFRInvariantError or logs a warning depending
    on `raise_on_mismatch`. Must be called after libavformat finishes muxing
    AND after `mp4_fixup.fix_mp4_track_duration` runs.

    No-ops silently for non-mp4/mov containers (file lacks a moov atom).
    """
    try:
        with open(mp4_path, "rb") as f:
            data = f.read()
    except OSError as exc:
        logger.warning("CFR invariant check skipped: %s", exc)
        return

    moov = _find_box("moov", data)
    if moov is None:
        return  # not an mp4/mov; nothing to check
    trak = _find_box("trak", moov)
    if trak is None:
        return
    mdia = _find_box("mdia", trak)
    if mdia is None:
        return
    minf = _find_box("minf", mdia)
    if minf is None:
        return
    stbl = _find_box("stbl", minf)
    if stbl is None:
        return
    stts = _find_box("stts", stbl)
    if stts is None:
        return

    entries = _parse_stts(stts)
    if not entries:
        return

    bad_deltas = [(n, d) for (n, d) in entries if d != expected_typical_delta_out]
    sample_count = sum(n for (n, _) in entries)
    sum_of_deltas = sum(n * d for (n, d) in entries)
    expected_sum = sample_count * expected_typical_delta_out

    msg_lines = []
    if bad_deltas:
        msg_lines.append(
            f"video track has {len(bad_deltas)} stts entries with delta != "
            f"{expected_typical_delta_out}: {bad_deltas[:5]}"
            + (" ..." if len(bad_deltas) > 5 else "")
        )
    if sum_of_deltas != expected_sum:
        msg_lines.append(
            f"video track sum_of_deltas={sum_of_deltas} != "
            f"sample_count*typical_delta={expected_sum} "
            f"(diff={sum_of_deltas - expected_sum})"
        )

    mdhd = _find_box("mdhd", mdia)
    if mdhd is not None:
        v = mdhd[0]
        if v == 0:
            mdhd_dur = struct.unpack(">I", mdhd[16:20])[0]
        else:
            mdhd_dur = struct.unpack(">Q", mdhd[24:32])[0]
        if mdhd_dur != expected_sum:
            msg_lines.append(
                f"mdhd.duration={mdhd_dur} != expected={expected_sum} "
                f"(diff={mdhd_dur - expected_sum})"
            )

    if not msg_lines:
        return  # all good

    detail = "; ".join(msg_lines)
    full = f"CFR invariants failed for {mp4_path}: {detail}"
    if raise_on_mismatch:
        raise CFRInvariantError(full)
    logger.warning(full)
