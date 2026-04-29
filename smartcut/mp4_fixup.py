"""
mp4_fixup: post-write track-duration repair for smartcut MP4 outputs.

When libavformat's MP4 muxer writes the trailer for files produced by
smartcut, the very last sample in the `stts` table can be written with a
delta of 0 instead of the correct per-frame delta. This makes
`mdhd.duration`, `tkhd.duration` and `mvhd.duration` short by exactly one
frame's worth of ticks, even though `nb_frames` is correct and every
preceding `stts` entry is correct.

This module rewrites the offending atoms in place after the file has been
finalized, so the output presents itself as the CFR file that smartcut in
fact produced.

Only ISO BMFF / MP4 (`mp4`, `mov`, `m4a`, `m4v`, etc.) outputs are
affected. The fixup is a no-op for non-MP4 containers (MKV, WebM, ...)
and for files that already have a self-consistent `stts`.

The implementation is dependency-free (pure-Python `struct` parsing) and
does an in-place rewrite of fixed-width fields plus, when needed, a
single 4-byte stts entry value. No moov rewrite is required because
total atom sizes do not change.
"""

from __future__ import annotations

import os
import struct
from typing import Optional


_BMFF_EXTS = {".mp4", ".m4a", ".m4v", ".mov", ".3gp", ".3g2", ".mj2"}


def _is_bmff_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _BMFF_EXTS


def _scan_atoms(buf: bytes, start: int, end: int):
    """Yield (atom_type:bytes, body_start:int, body_end:int) for each top-level atom."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos:pos + 4])[0]
        atype = buf[pos + 4:pos + 8]
        if size == 1:
            # 64-bit extended size
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            body_start = pos + 16
        elif size == 0:
            # extends to end
            size = end - pos
            body_start = pos + 8
        else:
            body_start = pos + 8
        if size < 8 or pos + size > end:
            return
        yield atype, body_start, pos + size
        pos += size


def _find_atom(buf: bytes, start: int, end: int, want: bytes) -> Optional[tuple[int, int]]:
    for atype, bs, be in _scan_atoms(buf, start, end):
        if atype == want:
            return bs, be
    return None


def _find_path(buf: bytes, start: int, end: int, *path: bytes) -> Optional[tuple[int, int]]:
    cur = (start, end)
    for name in path:
        nxt = _find_atom(buf, cur[0], cur[1], name)
        if nxt is None:
            return None
        cur = nxt
    return cur


def fix_mp4_track_duration(path: str) -> bool:
    """Repair last-sample stts delta and track-header durations in an MP4.

    Returns True if the file was modified, False if no fix was needed
    (or the file is not an MP4 / has no usable `stts`).
    """
    if not _is_bmff_path(path):
        return False
    try:
        with open(path, "rb") as f:
            buf = bytearray(f.read())
    except OSError:
        return False

    moov = _find_atom(buf, 0, len(buf), b"moov")
    if moov is None:
        return False
    moov_s, moov_e = moov

    # mvhd
    mvhd = _find_atom(buf, moov_s, moov_e, b"mvhd")
    if mvhd is None:
        return False
    mvhd_s, _ = mvhd
    mvhd_ver = buf[mvhd_s]
    if mvhd_ver == 0:
        mvhd_ts_off = mvhd_s + 12
        mvhd_dur_off = mvhd_s + 16
        mvhd_dur_sz = 4
    else:
        mvhd_ts_off = mvhd_s + 20
        mvhd_dur_off = mvhd_s + 24
        mvhd_dur_sz = 8
    mvhd_timescale = struct.unpack(">I", buf[mvhd_ts_off:mvhd_ts_off + 4])[0]

    modified = False

    # Iterate every trak inside moov and fix its mdhd/tkhd/stts independently.
    new_mvhd_dur = 0
    for atype, bs, be in _scan_atoms(buf, moov_s, moov_e):
        if atype != b"trak":
            continue
        # tkhd
        tkhd = _find_atom(buf, bs, be, b"tkhd")
        if tkhd is None:
            continue
        tkhd_s, _ = tkhd
        tkhd_ver = buf[tkhd_s]
        if tkhd_ver == 0:
            tkhd_dur_off = tkhd_s + 24
            tkhd_dur_sz = 4
        else:
            tkhd_dur_off = tkhd_s + 32
            tkhd_dur_sz = 8

        # mdia → mdhd, minf/stbl/stts
        mdia = _find_atom(buf, bs, be, b"mdia")
        if mdia is None:
            continue
        mdia_s, mdia_e = mdia

        mdhd = _find_atom(buf, mdia_s, mdia_e, b"mdhd")
        if mdhd is None:
            continue
        mdhd_s, _ = mdhd
        mdhd_ver = buf[mdhd_s]
        if mdhd_ver == 0:
            mdhd_ts_off = mdhd_s + 12
            mdhd_dur_off = mdhd_s + 16
            mdhd_dur_sz = 4
        else:
            mdhd_ts_off = mdhd_s + 20
            mdhd_dur_off = mdhd_s + 24
            mdhd_dur_sz = 8
        mdhd_timescale = struct.unpack(">I", buf[mdhd_ts_off:mdhd_ts_off + 4])[0]
        old_mdhd_dur = (
            struct.unpack(">I", buf[mdhd_dur_off:mdhd_dur_off + 4])[0]
            if mdhd_dur_sz == 4
            else struct.unpack(">Q", buf[mdhd_dur_off:mdhd_dur_off + 8])[0]
        )

        stbl_path = _find_path(buf, mdia_s, mdia_e, b"minf", b"stbl")
        if stbl_path is None:
            continue
        stbl_s, stbl_e = stbl_path
        stts = _find_atom(buf, stbl_s, stbl_e, b"stts")
        if stts is None:
            continue
        stts_s, stts_e = stts
        # version+flags (4) + entry_count (4)
        if stts_e - stts_s < 8:
            continue
        entry_count = struct.unpack(">I", buf[stts_s + 4:stts_s + 8])[0]
        if entry_count == 0:
            continue

        # Walk entries; collect last entry offset and the previous run's delta.
        last_entry_off = stts_s + 8 + (entry_count - 1) * 8
        prev_entry_off = last_entry_off - 8 if entry_count >= 2 else None
        if last_entry_off + 8 > stts_e:
            continue
        last_count, last_delta = struct.unpack(
            ">II", buf[last_entry_off:last_entry_off + 8]
        )

        # If the last run is a single sample with delta=0, replace its delta
        # with the typical delta (taken from the previous run if available,
        # else the median of all runs).
        replacement_delta = None
        if last_count == 1 and last_delta == 0 and prev_entry_off is not None:
            _, prev_delta = struct.unpack(
                ">II", buf[prev_entry_off:prev_entry_off + 8]
            )
            if prev_delta > 0:
                replacement_delta = prev_delta

        if replacement_delta is not None:
            buf[last_entry_off + 4:last_entry_off + 8] = struct.pack(
                ">I", replacement_delta
            )
            last_delta = replacement_delta
            modified = True

        # Recompute correct mdhd duration as sum of (count * delta) over all runs.
        total_in_mdhd_ts = 0
        for k in range(entry_count):
            off = stts_s + 8 + k * 8
            cnt, dlt = struct.unpack(">II", buf[off:off + 8])
            total_in_mdhd_ts += cnt * dlt

        if total_in_mdhd_ts != old_mdhd_dur:
            if mdhd_dur_sz == 4:
                buf[mdhd_dur_off:mdhd_dur_off + 4] = struct.pack(
                    ">I", min(total_in_mdhd_ts, 0xFFFFFFFF)
                )
            else:
                buf[mdhd_dur_off:mdhd_dur_off + 8] = struct.pack(
                    ">Q", total_in_mdhd_ts
                )
            modified = True

        # tkhd duration is in movie timescale, not media timescale.
        if mvhd_timescale > 0 and mdhd_timescale > 0:
            tkhd_dur = (total_in_mdhd_ts * mvhd_timescale) // mdhd_timescale
        else:
            tkhd_dur = total_in_mdhd_ts
        old_tkhd_dur = (
            struct.unpack(">I", buf[tkhd_dur_off:tkhd_dur_off + 4])[0]
            if tkhd_dur_sz == 4
            else struct.unpack(">Q", buf[tkhd_dur_off:tkhd_dur_off + 8])[0]
        )
        if tkhd_dur != old_tkhd_dur:
            if tkhd_dur_sz == 4:
                buf[tkhd_dur_off:tkhd_dur_off + 4] = struct.pack(
                    ">I", min(tkhd_dur, 0xFFFFFFFF)
                )
            else:
                buf[tkhd_dur_off:tkhd_dur_off + 8] = struct.pack(
                    ">Q", tkhd_dur
                )
            modified = True

        if tkhd_dur > new_mvhd_dur:
            new_mvhd_dur = tkhd_dur

    if new_mvhd_dur > 0:
        old_mvhd_dur = (
            struct.unpack(">I", buf[mvhd_dur_off:mvhd_dur_off + 4])[0]
            if mvhd_dur_sz == 4
            else struct.unpack(">Q", buf[mvhd_dur_off:mvhd_dur_off + 8])[0]
        )
        if new_mvhd_dur != old_mvhd_dur:
            if mvhd_dur_sz == 4:
                buf[mvhd_dur_off:mvhd_dur_off + 4] = struct.pack(
                    ">I", min(new_mvhd_dur, 0xFFFFFFFF)
                )
            else:
                buf[mvhd_dur_off:mvhd_dur_off + 8] = struct.pack(
                    ">Q", new_mvhd_dur
                )
            modified = True

    if not modified:
        return False

    # Rewrite the file in place.
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(bytes(buf))
    return True
