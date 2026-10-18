import argparse
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction
from itertools import pairwise

import numpy as np
import psutil
from av import datasets as av_datasets
from av import logging as av_logging
from av import open as av_open
from av.stream import Disposition
from test_utils import (
    cached_download,
    check_audio_pts_timestamps,
    check_no_discard_packets,
    check_stream_dispositions,
    check_videos_equal,
    check_videos_equal_segment,
    compare_tracks,
    create_test_video,
    generate_sine_wave,
    get_attachment_stream_metadata,
    get_media_test_data_file,
    get_tears_of_steel_annexb,
    get_testvideos_jellyfish_h265_ts,
    make_video_and_audio_mkv,
    make_video_with_attachment,
    make_video_with_forced_subtitle,
    make_video_with_subtitles,
    run_cut_on_keyframes_test,
    run_partial_smart_cut,
    run_smartcut_test,
    to_fraction_segments,
)

from smartcut.media_container import MediaContainer
from smartcut.media_utils import VideoExportMode, VideoExportQuality
from smartcut.misc_data import AudioExportInfo, AudioExportSettings
from smartcut.smart_cut import make_cut_segments, smart_cut
from smartcut.video_cutter import VideoSettings

DEFAULT_SEED = 12345

# Set the log level to silence the None dts warnings. I believe those can be ignored since
# we do set dts, except when it's not set in the source in which case it's not clear what
# value dts should take. It would be nice to occasionally check that there aren't more warnings.
# UPDATE 2024-09-13: setting logging level to FATAL because mpeg2 tests spam a lot of buffer errors I didn't manage to silence
av_logging.set_level(av_logging.FATAL)

data_dir = 'test_data'

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Smart Media Cutter Tests')
    parser.add_argument('--category', choices=['basic', 'h264', 'h265', 'codecs', 'containers', 'audio', 'mixed', 'transforms', 'long', 'external', 'real_world', 'real_world_h264', 'real_world_h265', 'real_world_av1', 'real_world_vp9', 'all'],
                       help='Run tests from specific category')
    parser.add_argument('--single', type=str, help='Run a single specific test function (e.g., test_h264_non_idr_keyframes)')
    parser.add_argument('--list-categories', action='store_true', help='List available test categories')
    parser.add_argument('--list-tests', action='store_true', help='List all available test functions')
    parser.add_argument('--flaky', type=int, help='Repeat each selected test N times with different seeds')
    parser.add_argument('--seed', type=int, help='Base PRNG seed (defaults to 12345)')
    parser.add_argument('--pixel-tolerance', type=int, default=50, help='Pixel color difference tolerance for file tests (default: 50)')
    parser.add_argument('files', nargs='*', help='Specific video files to test')

    args = parser.parse_args()

    if args.list_categories:
        print("Available test categories:")
        print("  basic      - Quick fundamental H.264 tests")
        print("  h264       - H.264 codec tests including profiles")
        print("  h265       - H.265/HEVC codec tests")
        print("  codecs     - Other video codecs (VP9, AV1, etc.)")
        print("  containers - Container format tests (MP4, AVI, MOV, etc.)")
        print("  audio      - Audio-only tests (MP3, Vorbis, FLAC)")
        print("  mixed      - Tests with video and audio tracks")
        print("  transforms - Video transformation and recoding tests")
        print("  long       - Long-running or large file tests")
        print("  external   - Tests requiring external downloads")
        print("  real_world - All real-world video tests from public sources")
        print("  real_world_h264 - Real-world H.264 videos (Google, test-videos.co.uk, etc)")
        print("  real_world_h265 - Real-world H.265/HEVC videos")
        print("  real_world_av1  - Real-world AV1 videos")
        print("  real_world_vp9  - Real-world VP9 videos")
        print("  all        - Run all tests (default)")
        sys.exit(0)

    if args.list_tests:
        print("Available test functions:")
        test_categories = get_test_categories()
        all_tests = []
        for cat_tests in test_categories.values():
            all_tests.extend(cat_tests)

        # Remove duplicates while preserving order
        seen = set()
        unique_tests = []
        for test in all_tests:
            if test not in seen:
                seen.add(test)
                unique_tests.append(test)

        for test in sorted(unique_tests, key=lambda t: t.__name__):
            print(f"  {test.__name__}")
        sys.exit(0)

    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_base_seed(args: argparse.Namespace | None) -> int:
    if args is None or args.seed is None:
        return DEFAULT_SEED
    return args.seed

os.chdir(os.path.dirname(__file__))
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

short_h264_path = 'short_h264.mkv'
short_h265_path = 'short_h265.mkv'


def test_cfr_recode_preserves_decode_delay() -> None:
    from types import SimpleNamespace

    from smartcut.video_cutter import VideoCutter

    for encoder_delay in (0, 2002):
        cutter = VideoCutter.__new__(VideoCutter)
        cutter.out_stream = None
        cutter.out_time_base = Fraction(1, 30000)
        cutter.last_dts = -100_000_000
        cutter._typical_frame_duration_out_pts = 1001
        cutter._cfr_decode_start = None
        cutter._cfr_decode_index = 0
        cutter._cfr_reorder_delay = 2002
        cutter.typical_frame_duration = None
        cutter._snap_total_emitted = 0
        recoded = [SimpleNamespace(pts=0, dts=-encoder_delay, duration=1001),
                   SimpleNamespace(pts=1001, dts=1001 - encoder_delay, duration=1001)]
        cutter._maybe_snap_recoded_packets(recoded)
        for packet in recoded:
            cutter._fix_packet_timestamps(packet)
        copied = [SimpleNamespace(pts=2002, dts=0, duration=1001),
                  SimpleNamespace(pts=6006, dts=1001, duration=1001)]
        for packet in copied:
            cutter._fix_packet_timestamps(packet)
        packets = recoded + copied
        assert [packet.pts for packet in packets] == [0, 1001, 2002, 6006]
        assert [packet.dts for packet in packets] == [-2002, -1001, 0, 1001]
        assert all(packet.duration == 1001 for packet in packets)
        invalid = SimpleNamespace(pts=0, dts=2002, duration=1001)
        try:
            cutter._fix_packet_timestamps(invalid)
        except RuntimeError as error:
            assert 'decode grid' in str(error)
        else:
            raise AssertionError('Invalid presentation time was silently changed')
        assert invalid.pts == 0


def test_h264_cfr_no_b_frames_boundary_flush() -> None:
    source_path = 'cfr_no_b_frames_source.mp4'
    rate = Fraction(30000, 1001)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=64x36:rate=30000/1001', '-frames:v', '300', '-c:v', 'libx264',
        '-threads', '2', '-g', '30', '-bf', '0', '-sc_threshold', '0',
        '-video_track_timescale', '30000', source_path], check=True)
    for start_frame, end_frame in ((30, 250), (29, 184)):
        source = MediaContainer(source_path)
        output_path = f'cfr_no_b_frames_{start_frame}_{end_frame}.mp4'
        smart_cut(source, [(start_frame / rate, end_frame / rate)], output_path, log_level='warning')
        expected = end_frame - start_frame
        with av_open(output_path) as container:
            stream = container.streams.video[0]
            assert stream.average_rate == rate
            assert stream.frames == expected
            packets = [packet for packet in container.demux(stream) if packet.size]
            expected_times = [Fraction(index, 1) / rate for index in range(expected)]
            assert [packet.pts * packet.time_base for packet in packets] == expected_times
            assert [packet.dts * packet.time_base for packet in packets] == expected_times
            assert all(packet.duration * packet.time_base == 1 / rate for packet in packets)
        with av_open(output_path) as container:
            assert sum(1 for _frame in container.decode(video=0)) == expected


def test_h264_cut_on_keyframes() -> None:
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_h264_cut_on_keyframes.__name__ + '.mkv'
    run_cut_on_keyframes_test(short_h264_path, output_path)

def test_h264_smart_cut() -> None:
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_h264_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10, 30, 100]:
        run_smartcut_test(short_h264_path, output_path, c)

def test_h264_multiple_cuts() -> None:
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    source = MediaContainer(short_h264_path)

    output_path = test_h264_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10, 30, 100]:
        cutpoints = source.video_frame_times
        cutpoints = [0, *np.sort(np.random.choice(cutpoints, c, replace=False)), source.duration]

        segments = list(pairwise(cutpoints))
        segments = to_fraction_segments(segments)

        smart_cut(source, segments, output_path, log_level='warning')
        result_container = MediaContainer(output_path)
        check_videos_equal(source, result_container)

def test_h265_cut_on_keyframes() -> None:
    create_test_video(short_h265_path, 30, 'hevc', 'yuv422p10le', 60, (256, 144))
    output_path = test_h265_cut_on_keyframes.__name__ + '.mkv'
    run_cut_on_keyframes_test(short_h265_path, output_path)

def test_h265_smart_cut() -> None:
    create_test_video(short_h265_path, 30, 'hevc', 'yuv422p10le', 60, (256, 144))
    output_path = test_h265_smart_cut.__name__ + '.mkv'
    for c in [1, 2]:
        run_smartcut_test(short_h265_path, output_path, c)

def test_h265_smart_cut_large() -> None:
    input_file = 'h265_large.mkv'
    create_test_video(input_file, 17, 'hevc', 'yuv422p', 25, (1280, 720))
    output_path = test_h265_smart_cut_large.__name__ + '.mkv'
    for c in [1, 2, 5]:
        run_smartcut_test(input_file, output_path, c)

def test_peaks_mkv_memory_usage() -> None:
    """Test that peaks.mkv doesn't cause excessive memory usage and has proper GOP detection."""
    peaks_path = get_media_test_data_file("peaks.mkv")

    # Test GOP detection
    container = MediaContainer(peaks_path)

    actual_gops = len(container.gop_start_times_pts_s)
    expected_gops = 10

    assert actual_gops >= expected_gops, f"Expected {expected_gops} GOPs but found {actual_gops}"

    container.close()

    # Test memory usage during cutting
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    peak_memory = initial_memory
    stop_monitoring = False

    def monitor_thread() -> None:
        nonlocal peak_memory
        while not stop_monitoring:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
            time.sleep(0.1)

    monitor = threading.Thread(target=monitor_thread)
    monitor.daemon = True
    monitor.start()

    output_path = test_peaks_mkv_memory_usage.__name__ + '.mp4'

    try:
        segments = [(Fraction(0), Fraction(5))]

        container = MediaContainer(peaks_path)
        audio_settings: list[AudioExportSettings | None] = [AudioExportSettings(codec='passthru')] * len(container.audio_tracks)
        export_info = AudioExportInfo(output_tracks=audio_settings)
        video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)

        smart_cut(
            container,
            segments,
            output_path,
            audio_export_info=export_info,
            video_settings=video_settings,
            log_level='error'
        )

        container.close()

    finally:
        stop_monitoring = True
        if monitor.is_alive():
            monitor.join(timeout=1)

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

    # Memory usage check - should not exceed 700MB
    memory_increase = peak_memory - initial_memory
    max_allowed_memory = 700 * 1024 * 1024

    print(f"Peak memory increase: {memory_increase / (1024*1024):.1f} MB")
    assert memory_increase < max_allowed_memory, f"Memory usage too high: {memory_increase / (1024*1024):.1f} MB"

def test_hevc_mkv_nal_type_detection() -> None:
    """Test that HEVC NAL type detection finds IDR frames, not VPS headers.

    Regression test for a bug where get_h265_nal_unit_type() only read the first
    NAL unit in MP4-format packets. For MKV files with HEVC, keyframe packets
    contain multiple NAL units (VPS, SPS, PPS, IDR), and the function would
    return the VPS type (39) instead of the actual IDR type (19/20).
    This caused most keyframes to be rejected as "unsafe", resulting in
    very few GOPs being detected.
    """
    test_path = get_media_test_data_file("hevc_mkv_no_dts.mkv")

    container = MediaContainer(test_path)

    # The test file should have multiple GOPs (at least 3 in ~10 seconds)
    # Before the fix, only 1 or 2 GOPs would be detected
    actual_gops = len(container.gop_start_times_pts_s)
    expected_min_gops = 3

    assert actual_gops >= expected_min_gops, (
        f"Expected at least {expected_min_gops} GOPs but found {actual_gops}. "
        "This suggests NAL type detection is returning VPS/SPS types instead of IDR."
    )

    container.close()

def test_h265_zcam() -> None:
    """Test smartcut with ZCAM H.265/HEVC MOV file."""
    test_path = get_media_test_data_file("zcam.mov")

    output_path = test_h265_zcam.__name__ + '.mov'
    run_smartcut_test(test_path, output_path, n_cuts=1, audio_export_info='auto')


def test_h265_zcam_containers() -> None:
    """Test smartcut with ZCAM HEVC converted to different containers.

    Converts zcam.mov to MP4 and MKV, then tests smartcut on each.
    This verifies the hev1 codec tag handling works correctly across containers.
    """
    source_path = get_media_test_data_file("zcam.mov")

    # Test MP4 container
    mp4_path = 'test_zcam_converted.mp4'
    if not os.path.exists(mp4_path):
        subprocess.run([
            'ffmpeg', '-y', '-i', source_path,
            '-c', 'copy', mp4_path
        ], check=True, capture_output=True)
    run_smartcut_test(mp4_path, 'test_zcam_mp4_out.mp4', n_cuts=1, audio_export_info='auto')

    # Test MKV container
    mkv_path = 'test_zcam_converted.mkv'
    if not os.path.exists(mkv_path):
        subprocess.run([
            'ffmpeg', '-y', '-i', source_path,
            '-c', 'copy', mkv_path
        ], check=True, capture_output=True)
    run_smartcut_test(mkv_path, 'test_zcam_mkv_out.mkv', n_cuts=1, audio_export_info='auto')


def test_hevc_mkv_no_dts_remux() -> None:
    """Test that smartcut handles HEVC MKV files with DTS=N/A packets.

    Regression test for a bug where packets with DTS=None would get monotonically
    increasing DTS values generated that could exceed PTS (violating PTS >= DTS).
    This happened because the code used max(pts, last_dts + 1) which would push
    DTS above PTS for B-frames.
    """
    test_path = get_media_test_data_file("hevc_mkv_no_dts.mkv")

    output_path = test_hevc_mkv_no_dts_remux.__name__ + '.mkv'

    container = MediaContainer(test_path)
    segments = [(Fraction(0), Fraction(5))]  # First 5 seconds

    audio_settings: list[AudioExportSettings | None] = [AudioExportSettings(codec='passthru')] * len(container.audio_tracks)
    export_info = AudioExportInfo(output_tracks=audio_settings)
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)

    # This should not raise an error - before the fix it would fail with:
    # av.error.ValueError: [Errno 22] Invalid argument
    smart_cut(
        container,
        segments,
        output_path,
        audio_export_info=export_info,
        video_settings=video_settings,
        log_level='error'
    )

    container.close()

    # Verify output was created
    assert os.path.exists(output_path), "Output file was not created"

    # Clean up
    os.remove(output_path)

def test_h264_24_fps_long() -> None:
    filename = 'long_h264.mkv'
    # 15 mins
    create_test_video(filename, 60 * 15, 'h264', 'yuv420p', 24, (32, 18))
    output_path = test_h264_24_fps_long.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_1080p() -> None:
    filename = '1080p_h264.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (1920, 1080))
    output_path = test_h264_1080p.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_baseline() -> None:
    filename = 'h264_baseline.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='baseline')
    output_path = test_h264_profile_baseline.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_main() -> None:
    filename = 'h264_main.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='main')
    output_path = test_h264_profile_main.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high() -> None:
    filename = 'h264_high.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='high')
    output_path = test_h264_profile_high.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high10() -> None:
    filename = 'h264_high10.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p10le', 30, (32, 18), profile='high10')
    output_path = test_h264_profile_high10.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high422() -> None:
    filename = 'h264_high422.mkv'
    create_test_video(filename, 15, 'h264', 'yuv422p', 30, (32, 18), profile='high422')
    output_path = test_h264_profile_high422.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high444() -> None:
    filename = 'h264_high444.mkv'
    create_test_video(filename, 15, 'h264', 'yuv444p', 30, (32, 18), profile='high444')
    output_path = test_h264_profile_high444.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_mp4_cut_on_keyframe() -> None:
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_cut_on_keyframe.__name__ + '.mp4'
    run_cut_on_keyframes_test(filename, output_path)

def test_mp4_smart_cut() -> None:
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mp4_to_mkv_smart_cut() -> None:
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_to_mkv_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mkv_to_mp4_smart_cut() -> None:
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mkv_to_mp4_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(short_h264_path, output_path, c)

def test_mp4_h265_smart_cut() -> None:
    filename = 'h265.mp4'
    create_test_video(filename, 30, 'hevc', 'yuv420p', 30, (256, 144))
    output_path = test_mp4_h265_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mpg_smart_cut() -> None:
    filename = cached_download('https://filesamples.com/samples/video/mpg/sample_640x360.mpg', 'mpeg640x360.mpg')
    output_path = test_mpg_smart_cut.__name__ + '.mpg'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mpg_cut_on_keyframes() -> None:
    filename = cached_download('https://filesamples.com/samples/video/mpg/sample_640x360.mpg', 'mpeg640x360.mpg')
    output_path = test_mpg_cut_on_keyframes.__name__ + '.mpg'
    run_cut_on_keyframes_test(filename, output_path)

def test_m2ts_mpeg2_smart_cut() -> None:
    filename = cached_download('https://filesamples.com/samples/video/m2ts/sample_960x540.m2ts', 'm2ts960x540.m2ts')
    output_path = test_m2ts_mpeg2_smart_cut.__name__ + '.m2ts'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_m2ts_h264_smart_cut() -> None:
    filename = cached_download('https://www.dwsamplefiles.com/?dl_id=311', 'm2ts_h264_636x360.m2ts')
    output_path = test_m2ts_h264_smart_cut.__name__ + '.m2ts'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_ts_smart_cut() -> None:
    filename = cached_download('https://filesamples.com/samples/video/ts/sample_1280x720.ts', 'mpeg2.ts')
    output_path = test_ts_smart_cut.__name__ + '.ts'
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, c, pixel_tolerance=30)

# Video transform functionality has been removed
# def test_vertical_transform():
#     pass

def test_video_recode_codec_override() -> None:
    input_path = 'video_settings_in.mkv'
    file_duration = 10
    n_cuts = 5
    create_test_video(input_path, file_duration, 'h264', 'yuv420p', 30, (854, 480))

    source_container = MediaContainer(input_path)

    cutpoints = source_container.video_frame_times
    cutpoints = [0, *np.sort(np.random.choice(cutpoints, n_cuts, replace=False)), source_container.duration]

    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)

    output_path_a = test_video_recode_codec_override.__name__ + 'a.mkv'
    output_path_b = test_video_recode_codec_override.__name__ + 'b.mkv'

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, codec_override='hevc')
    smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_a)
    video_stream = output_container.video_stream
    assert video_stream is not None, 'Expected video stream in output'
    assert video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, codec_override='hevc')
    smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_b)
    video_stream = output_container.video_stream
    assert video_stream is not None, 'Expected video stream in output'
    assert video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, codec_override='vp9')
    smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_a)
    video_stream = output_container.video_stream
    assert video_stream is not None, 'Expected video stream in output'
    assert video_stream.codec_context.name == 'vp9', f'codec should be vp9, found {video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, codec_override='vp9')
    smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_b)
    video_stream = output_container.video_stream
    assert video_stream is not None, 'Expected video stream in output'
    assert video_stream.codec_context.name == 'vp9', f'codec should be vp9, found {video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

    # These tests are very slow because the encoders are slow
    # video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, codec_override='av1')
    # smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    # output_container = MediaContainer(output_path_a)
    # assert output_container.video_stream.codec_context.name == 'libdav1d', f'codec should be av1, found {output_container.video_stream.codec_context.name}'
    # check_videos_equal(source_container, output_container)

    # video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, codec_override='vp9')
    # smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    # output_container = MediaContainer(output_path_b)
    # assert output_container.video_stream.codec_context.name == 'libdav1d', f'codec should be av1 found {output_container.video_stream.codec_context.name}'
    # check_videos_equal(source_container, output_container)

    # assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

def test_vorbis_passthru() -> None:
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_passthru.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0, *[Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))], file_duration]

    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)

    settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    output_container = MediaContainer(output_path)

    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # Test case from bug report: export 0-2 seconds
    prefix_output_path = test_vorbis_passthru.__name__ + '_prefix.ogg'
    cutpoints = [0, 2]
    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)
    smart_cut(source_container, segments, prefix_output_path, audio_export_info=export_info)
    prefix_container = MediaContainer(prefix_output_path)

    # Check duration is correct
    assert prefix_container.duration > 1.9 and prefix_container.duration < 2.1, \
        f"Expected duration ~2s, got {prefix_container.duration}s"

    # Check PTS timestamps are correct (not offset by ~10 seconds)
    check_audio_pts_timestamps(prefix_container, 2.0, "Prefix (0-2s)")

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.duration > 14.9 and suffix_container.duration < 15.1

    # Check PTS timestamps for suffix case
    check_audio_pts_timestamps(suffix_container, 15.0, "Suffix (15-30s)")

def test_mp3_passthru() -> None:
    filename = 'basic.mp3'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_mp3_passthru.__name__ + '.mp3'

    n_cuts = 5
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0, *[Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))], file_duration]

    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)

    settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    output_container = MediaContainer(output_path)

    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # Test case: export 0-2 seconds
    prefix_output_path = test_mp3_passthru.__name__ + '_prefix.mp3'
    cutpoints = [0, 2]
    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)
    smart_cut(source_container, segments, prefix_output_path, audio_export_info=export_info)
    prefix_container = MediaContainer(prefix_output_path)

    # Check duration is correct
    assert prefix_container.duration > 1.9 and prefix_container.duration < 2.1, \
        f"Expected duration ~2s, got {prefix_container.duration}s"

    # Check PTS timestamps are correct
    check_audio_pts_timestamps(prefix_container, 2.0, "Prefix (0-2s)")

    # partial file i.e. suffix
    suffix_output_path = test_mp3_passthru.__name__ + '_suffix.mp3'

    cutpoints = [15, file_duration]
    segments = list(pairwise(cutpoints))
    segments = to_fraction_segments(segments)
    smart_cut(source_container, segments, suffix_output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(suffix_output_path)
    assert suffix_container.duration > 14.8 and suffix_container.duration < 15.1

    # Check PTS timestamps for suffix case
    check_audio_pts_timestamps(suffix_container, 15.0, "Suffix (15-30s)")

def test_mkv_with_video_and_audio_passthru() -> None:
    file_duration = 30

    final_input = 'video_and_two_audio.mkv'
    make_video_and_audio_mkv(final_input, file_duration)

    output_path = test_mkv_with_video_and_audio_passthru.__name__ + '.mkv'

    source_container = MediaContainer(final_input)

    passthru_settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[passthru_settings, passthru_settings])
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)

    assert len(result_container.audio_tracks) == 2
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[1])

    export_info = AudioExportInfo(output_tracks=[None, passthru_settings])
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0])

    export_info = AudioExportInfo(output_tracks=[passthru_settings, None])
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])

def test_vp9_smart_cut() -> None:
    filename = 'vp9.mkv'

    create_test_video(filename, 30, 'vp9', 'yuv420p', 30, (256, 144))
    output_path = test_vp9_smart_cut.__name__ + '.mkv'
    for c in [2, 6]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_vp9_profile_1() -> None:
    filename = 'vp9_p1_422.mkv'

    create_test_video(filename, 30, 'vp9', 'yuv422p', 30, (256, 144), profile='1')
    output_path = test_vp9_profile_1.__name__ + '.mkv'
    for c in [2, 6]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_av1_smart_cut() -> None:
    filename = 'av1.mkv'

    create_test_video(filename, 30, 'av1', 'yuv420p', 30, (256, 144))
    output_path = test_av1_smart_cut.__name__ + '.mkv'
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_avi_mpeg4_smart_cut() -> None:
    filename = 'mpeg4.avi'

    create_test_video(filename, 30, 'mpeg4', 'yuv420p', 30, (32, 18))
    output_path = test_avi_mpeg4_smart_cut.__name__ + '.avi'
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_avi_mjpeg_smart_cut() -> None:
    filename = 'mjpeg.avi'
    create_test_video(filename, 30, 'mjpeg', 'yuvj420p', 30, (32, 18))
    output_path = test_avi_mjpeg_smart_cut.__name__ + '.avi'
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_avi_h263_smart_cut() -> None:
    filename = 'h263.avi'
    create_test_video(filename, 30, 'h263', 'yuv420p', 30, (128, 96))
    output_path = test_avi_h263_smart_cut.__name__ + '.avi'
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_avi_to_mkv_smart_cut() -> None:
    filename = 'h264_input.avi'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (128, 96))
    output_path = test_avi_to_mkv_smart_cut.__name__ + '.mkv'
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c, pixel_tolerance=50)

def test_flv_smart_cut() -> None:
    filename = 'flv.flv'
    create_test_video(filename, 30, 'flv', 'yuv420p', 30, (32, 16))
    output_path = test_flv_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_mov_smart_cut() -> None:
    filename = 'h264.mov'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 16))
    output_path = test_mov_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'prores.mov'
    create_test_video(filename, 30, 'prores', 'yuv422p10le', 30, (32, 16))
    output_path = test_mov_smart_cut.__name__ + filename
    for c in [2, 5]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_wmv_smart_cut() -> None:
    filename = 'mpeg4.wmv'
    create_test_video(filename, 30, 'mpeg4', 'yuv420p', 30, (32, 16))
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'wmv1.wmv'
    create_test_video(filename, 30, 'wmv1', 'yuv420p', 30, (32, 16))
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 5]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'wmv2.wmv'
    create_test_video(filename, 30, 'wmv2', 'yuv420p', 30, (32, 16))
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 5]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = cached_download('https://filesamples.com/samples/video/wmv/sample_960x540.wmv', 'msmpeg.wmv')
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c, pixel_tolerance=50)

def test_night_sky() -> None:
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
    output_path = test_night_sky.__name__ + '.mp4'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_night_sky_to_mkv() -> None:
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
    output_path = test_night_sky_to_mkv.__name__ + '.mkv'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_sunset() -> None:
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-sunset-by-the-sea-854400.mp4")
    output_path = test_sunset.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings)

def test_seeking() -> None:
    in_file = "seek_in.mkv"
    ref_file = "seek_ref.mkv"
    create_test_video(in_file, 600, 'h264', 'yuv420p', 30, (32, 18))
    create_test_video(ref_file, 10, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_seeking.__name__ + '.mkv'

    source = MediaContainer(in_file)
    segments = to_fraction_segments([(590, 600)])

    smart_cut(source, segments, output_path, log_level='warning')

    ref_container = MediaContainer(ref_file)
    result_container = MediaContainer(output_path)

    check_videos_equal(ref_container, result_container)

# This tests cutting of interlaced video and it fails. I don't see a way to make it work.
# Therefore interlacing is unsupported, probably forever. Leaving the test here for future reference.
def test_fate_interlaced_crop() -> None:
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.fate("h264/interlaced_crop.mp4")
    output_path = test_fate_interlaced_crop.__name__ + '.mp4'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_broken_ref_vid() -> None:
    # os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = '../ref_videos/remove_mistakes_short.mkv'
    output_path = test_broken_ref_vid.__name__ + '.mkv'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_manual() -> None:
    seed_all(12345)
    test_h265_zcam()

# Real-world video tests using publicly available videos

def test_google_bigbuckbunny() -> None:
    """Test with Google's hosted Big Buck Bunny H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', 'google_bigbuckbunny.mp4')
    output_base = test_google_bigbuckbunny.__name__
    run_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=4, audio_export_info='auto')

def test_google_elephantsdream() -> None:
    """Test with Google's hosted Elephant's Dream H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4', 'google_elephantsdream.mp4')
    output_base = test_google_elephantsdream.__name__
    run_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=4, audio_export_info='auto')

def test_google_forbiggerblaze() -> None:
    """Test with Google's hosted ForBiggerBlazes H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4', 'google_forbiggerblaze.mp4')
    output_path = test_google_forbiggerblaze.__name__ + '.mp4'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_google_forbiggeresc() -> None:
    """Test with Google's hosted ForBiggerEscapes H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4', 'google_forbiggeresc.mp4')
    output_path = test_google_forbiggerblaze.__name__ + '.mp4'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_google_subaru() -> None:
    """Test with Google's hosted Subaru Outback commercial H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4', 'google_subaru.mp4')
    output_base = test_google_subaru.__name__
    run_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=4, audio_export_info='auto')

def test_google_tears_of_steel() -> None:
    """Test with Google's hosted Tears of Steel H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4', 'google_tears_of_steel.mp4')
    output_base = test_google_tears_of_steel.__name__
    run_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=4, audio_export_info='auto', pixel_tolerance=30)

def test_libde265_tears_of_steel_h265() -> None:
    """Test Tears of Steel (HEVC) from libde265 bitstreams using partial segments.

    Downloads a long HEVC MKV sample and validates smart cut against a full
    recode. For speed, force the recode path to H.264 while preserving HEVC
    for the smart-cut path.

    This file has 367 CRA frames and only 1 IDR frame, with RASL
    pictures following each CRA.

    Key validation: frame counts must match between smartcut and recode,
    confirming that RASL pictures are properly handled by recoding CRA+RASL
    segments instead of dropping them.
    """
    filename = cached_download('https://www.libde265.org/hevc-bitstreams/tos-1720x720-cfg01.mkv', 'libde265_tos_1720x720_cfg01.mkv')
    output_base = test_libde265_tears_of_steel_h265.__name__
    # Use partial segments for long video; override recode codec to H.264 for speed
    run_partial_smart_cut(
        filename,
        output_base,
        segment_duration=15,
        n_segments=3,
        audio_export_info='auto',
        pixel_tolerance=60,
        allow_failed_frames=12,
        recode_codec_override='h264',
        allow_failed_pixels_per_frame=3,
    )

def test_h264_non_idr_keyframes() -> None:
    """
    Test that H.264 non-IDR I-frame issues are properly handled.

    Cuts Google Tears of Steel video at specific points that would force segments
    to start with non-IDR I-frames (NAL type 1). Without proper NAL filtering,
    this causes '[h264] no frame!' errors during playback validation.

    This test should PASS after implementing H.264 NAL unit filtering.
    """
    original_filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4', 'google_tears_of_steel.mp4')

    source = MediaContainer(original_filename)

    # Cut just before the first 3 non-IDR keyframes (same as comprehensive test)
    segments = [
        (Fraction(0), Fraction(18400, 1000)),           # 0 to 18.4s (before 18.5s non-IDR)
        (Fraction(18400, 1000), Fraction(143280, 1000)), # 18.4s to 143.28s (before 143.375s non-IDR)
        (Fraction(143280, 1000), Fraction(183230, 1000)), # 143.28s to 183.23s (before 183.33s non-IDR)
        (Fraction(183230, 1000), source.duration)     # 183.23s to end
    ]

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)

    output_filename = 'h264_non_idr_minimal_test.mp4'

    smart_cut(source, segments, output_filename,
                audio_export_info=audio_export_info,
                video_settings=video_settings,
                log_level='info')

    result = MediaContainer(output_filename)

    # Test video playback - this should work after NAL filtering is implemented
    check_videos_equal_segment(source, result, 16.5, 4, pixel_tolerance=20)

def test_ts_h264_to_mp4_cut_on_keyframes() -> None:
    """
    Verify TS -> MP4 remux works when cutting on keyframes only.

    This exercises the pure passthrough path (no recoding) to ensure
    Annex B H.264 from TS containers can be muxed into MP4 correctly.
    """
    ts_input = get_tears_of_steel_annexb()
    source = MediaContainer(ts_input)

    # Pick GOP-aligned segments so the cut stays in passthrough mode
    gop_times = source.gop_start_times_pts_s
    assert len(gop_times) > 14, "Expected Tears of Steel sample to provide enough GOPs"

    segments = [
        (Fraction(gop_times[5]), Fraction(gop_times[7])),
        (Fraction(gop_times[12]), Fraction(gop_times[14])),
    ]

    cut_segments = make_cut_segments(source, segments, keyframe_mode=True)
    assert all(not segment.require_recode for segment in cut_segments), "Keyframe cuts should not require recoding"

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))

    output_filename = test_ts_h264_to_mp4_cut_on_keyframes.__name__ + ".mp4"

    smart_cut(
        source,
        segments,
        output_filename,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.KEYFRAMES, VideoExportQuality.HIGH),
        log_level='warning'
    )

    with av_open(output_filename) as container:
        assert 'mp4' in container.format.name, f"Expected MP4 container, got {container.format.name}"
        video_streams = [stream for stream in container.streams if stream.type == 'video']
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert video_streams[0].codec_context.name == 'h264', \
            f"Expected H.264 video codec, got {video_streams[0].codec_context.name}"

    result = MediaContainer(output_filename)
    assert len(result.video_frame_times) > 0, "Remuxed output should contain video frames"
    result_video_stream = result.video_stream
    assert result_video_stream is not None, "Expected video stream in result container"
    assert result_video_stream.codec_context.codec_tag == 'avc1', "MP4 remux should use avc1 codec tag"


def test_ts_h264_to_mp4_smart_cut() -> None:
    """
    Verify converting H.264 in MPEG-TS to MP4 works.

    Uses a known-good H.264 stream repackaged into TS (Annex B) as input,
    then performs smart cut to MP4 and compares against a full recode MP4
    of the same segments to validate correctness. This exercises the
    remux path and container conversion logic from TS -> MP4.
    """
    # Prepare H.264-in-TS input
    ts_input = get_tears_of_steel_annexb()

    source = MediaContainer(ts_input)

    # Pick two short non-overlapping segments well inside the file
    # to keep the test quick and stable.
    # Tears of Steel is long enough for these constants.
    segments = [
        (Fraction(10, 1), Fraction(18, 1)),
        (Fraction(30, 1), Fraction(38, 1)),
    ]

    # Passthrough all audio tracks by default
    s = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[s] * len(source.audio_tracks))

    # Output targets (force MP4 container)
    smartcut_output = test_ts_h264_to_mp4_smart_cut.__name__ + "_smartcut.mp4"
    recode_output = test_ts_h264_to_mp4_smart_cut.__name__ + "_recode.mp4"

    # Smart cut MP4
    smart_cut(
        source,
        segments,
        smartcut_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH),
        log_level='warning'
    )

    # Full recode MP4 for comparison
    smart_cut(
        source,
        segments,
        recode_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH),
        log_level='warning'
    )

    # Validate MP4 container and codec
    with av_open(smartcut_output) as c:
        assert 'mp4' in c.format.name, f"Expected MP4 container, got {c.format.name}"
        vstreams = [s for s in c.streams if s.type == 'video']
        assert len(vstreams) == 1, f"Expected 1 video stream, found {len(vstreams)}"
        assert vstreams[0].codec_context.name in ['h264', 'libx264'], \
            f"Expected H.264 video codec, got {vstreams[0].codec_context.name}"

    # Compare smart cut vs full recode outputs for equivalence
    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)

    # Allow a slightly looser tolerance for real-world content
    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=40)


def test_ts_h265_to_mp4_smart_cut() -> None:
    """Verify converting H.265 in MPEG-TS to MP4 works through the smart-cut path."""
    ts_input = get_testvideos_jellyfish_h265_ts()

    source = MediaContainer(ts_input)

    segments = [
        (Fraction(0, 1), Fraction(4, 1)),
        (Fraction(5, 1), Fraction(9, 1)),
    ]

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))

    smartcut_output = test_ts_h265_to_mp4_smart_cut.__name__ + '_smartcut.mp4'
    recode_output = test_ts_h265_to_mp4_smart_cut.__name__ + '_recode.mp4'

    smart_cut(
        source,
        segments,
        smartcut_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH),
        log_level='warning'
    )

    smart_cut(
        source,
        segments,
        recode_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH),
        log_level='warning'
    )

    with av_open(smartcut_output) as container:
        assert 'mp4' in container.format.name, f"Expected MP4 container, got {container.format.name}"
        video_streams = [stream for stream in container.streams if stream.type == 'video']
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert video_streams[0].codec_context.name in ['hevc', 'libx265'], \
            f"Expected H.265 video codec, got {video_streams[0].codec_context.name}"

    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)

    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=80, allow_failed_frames=2)


def test_ts_h264_to_mkv_smart_cut() -> None:
    """
    Verify converting H.264 in MPEG-TS to MKV works using the smart-cut pipeline.

    Mirrors the MP4 smart cut test but targets MKV to ensure remux + recode
    coverage without interfering codec-tag logic.
    """
    ts_input = get_tears_of_steel_annexb()
    source = MediaContainer(ts_input)

    segments = [
        (Fraction(10, 1), Fraction(18, 1)),
        (Fraction(30, 1), Fraction(38, 1)),
    ]

    audio_passthru = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_passthru] * len(source.audio_tracks))

    smartcut_output = test_ts_h264_to_mkv_smart_cut.__name__ + "_smartcut.mkv"
    recode_output = test_ts_h264_to_mkv_smart_cut.__name__ + "_recode.mkv"

    smart_cut(
        source,
        segments,
        smartcut_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH),
        log_level='warning'
    )

    smart_cut(
        source,
        segments,
        recode_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH),
        log_level='warning'
    )

    with av_open(smartcut_output) as container:
        assert 'matroska' in container.format.name, f"Expected MKV container, got {container.format.name}"
        video_streams = [s for s in container.streams if s.type == 'video']
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert video_streams[0].codec_context.name in ['h264', 'libx264'], \
            f"Expected H.264 video codec, got {video_streams[0].codec_context.name}"

    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)
    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=40)

def test_h264_non_idr_keyframes_annexb() -> None:
    """
    Test that H.264 non-IDR I-frame issues are properly handled in Annex B format.

    Uses Tears of Steel converted to TS format (which uses Annex B NAL format instead
    of length-prefixed). This verifies our NAL parsing works for both MP4 (ISOBMFF)
    and TS (Annex B) formats.

    This test should PASS after implementing H.264 NAL unit filtering.
    """
    ts_filename = get_tears_of_steel_annexb()

    source = MediaContainer(ts_filename)

    # Use the same cut points that expose non-IDR keyframe issues
    # These times should still be valid in the converted TS file
    segments = [
        (Fraction(0), Fraction(18400, 1000)),           # 0 to 18.4s (before 18.5s non-IDR)
        (Fraction(18400, 1000), Fraction(143280, 1000)), # 18.4s to 143.28s (before 143.375s non-IDR)
        (Fraction(143280, 1000), Fraction(183230, 1000)), # 143.28s to 183.23s (before 183.33s non-IDR)
        (Fraction(183230, 1000), source.duration)     # 183.23s to end
    ]

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)

    output_filename = 'h264_non_idr_annexb_test.ts'

    smart_cut(source, segments, output_filename,
                audio_export_info=audio_export_info,
                video_settings=video_settings,
                log_level='info')

    result = MediaContainer(output_filename)

    # Test video playback - this should work after NAL filtering is implemented
    check_videos_equal_segment(source, result, 16.5, 4, pixel_tolerance=20)

def test_testvideos_bigbuckbunny_h264() -> None:
    """Test with test-videos.co.uk Big Buck Bunny H.264"""
    filename = cached_download('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4', 'testvideos_bbb_h264.mp4')
    output_path = test_testvideos_bigbuckbunny_h264.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_bigbuckbunny_h265() -> None:
    """Test with test-videos.co.uk Big Buck Bunny H.265/HEVC"""
    filename = cached_download('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h265/360/Big_Buck_Bunny_360_10s_1MB.mp4', 'testvideos_bbb_h265.mp4')
    output_path = test_testvideos_bigbuckbunny_h265.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_bigbuckbunny_vp9() -> None:
    """Test with test-videos.co.uk Big Buck Bunny VP9"""
    filename = cached_download('https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm', 'testvideos_bbb_vp9.webm')
    output_path = test_testvideos_bigbuckbunny_vp9.__name__ + '.webm'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_jellyfish_h264() -> None:
    """Test with test-videos.co.uk Jellyfish H.264"""
    filename = cached_download('https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4', 'testvideos_jellyfish_h264.mp4')
    output_path = test_testvideos_jellyfish_h264.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_jellyfish_h265() -> None:
    """Test with test-videos.co.uk Jellyfish H.265"""
    filename = cached_download('https://test-videos.co.uk/vids/jellyfish/mp4/h265/360/Jellyfish_360_10s_1MB.mp4', 'testvideos_jellyfish_h265.mp4')
    output_path = test_testvideos_jellyfish_h265.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=80, allow_failed_frames=10)

def test_subtitle_disposition_preservation() -> None:
    """Test that subtitle disposition flags (especially forced) are preserved during smart_cut"""
    file_duration = 10
    input_path = make_video_with_forced_subtitle('test_subtitle_forced.mkv', file_duration)
    output_path = test_subtitle_disposition_preservation.__name__ + '.mkv'

    # Load source container and verify it has forced subtitles
    source = MediaContainer(input_path)

    # Verify the source has the expected subtitle with forced flag
    with av_open(input_path) as container:
        assert len(container.streams.subtitles) == 1, "Source should have exactly 1 subtitle stream"
        sub_stream = container.streams.subtitles[0]
        assert Disposition.forced in sub_stream.disposition, \
            f"Source subtitle should have forced flag, got: {sub_stream.disposition}"

    # Run smart_cut with a simple segment
    segments = [(Fraction(2), Fraction(8))]  # Cut from 2s to 8s
    smart_cut(source, segments, output_path, log_level='warning')

    # Verify disposition preservation
    check_stream_dispositions(input_path, output_path)

def test_multiple_language_subtitles() -> None:
    """Test that multiple non-forced subtitle tracks with different languages are preserved"""
    file_duration = 10

    # Define English subtitle content
    en_subtitle_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:05,000 --> 00:00:07,000
This is English

3
00:00:08,500 --> 00:00:09,500
The end
"""

    # Define Finnish subtitle content
    fi_subtitle_content = """1
00:00:01,000 --> 00:00:03,000
Hei maailma

2
00:00:05,000 --> 00:00:07,000
Tämä on suomea

3
00:00:08,500 --> 00:00:09,500
Loppu
"""

    # Configure both subtitle tracks
    subtitle_configs = [
        {
            'content': en_subtitle_content,
            'language': 'en',
            'disposition': 'default',  # English is default
            'temp_file': 'tmp_test_subtitle_en.srt'
        },
        {
            'content': fi_subtitle_content,
            'language': 'fi',
            'disposition': '',  # Finnish has no special disposition
            'temp_file': 'tmp_test_subtitle_fi.srt'
        }
    ]

    # Create test video with both subtitles
    input_path = make_video_with_subtitles('test_multi_subtitle.mkv', file_duration, subtitle_configs)
    output_path = test_multiple_language_subtitles.__name__ + '.mkv'

    # Verify source has both subtitle tracks
    with av_open(input_path) as container:
        assert len(container.streams.subtitles) == 2, f"Source should have 2 subtitle streams, got {len(container.streams.subtitles)}"

        # Check English subtitle (first stream)
        en_sub = container.streams.subtitles[0]
        assert Disposition.default in en_sub.disposition, \
            f"English subtitle should have default flag, got: {en_sub.disposition}"

        # Check Finnish subtitle (second stream)
        fi_sub = container.streams.subtitles[1]
        assert fi_sub.disposition.value == 0, \
            f"Finnish subtitle should have no disposition flags, got: {fi_sub.disposition}"

    # Run smart_cut
    source = MediaContainer(input_path)
    segments = [(Fraction(1), Fraction(9))]  # Cut from 1s to 9s to include all subtitles
    smart_cut(source, segments, output_path, log_level='warning')

    # Verify both subtitle tracks are preserved with correct dispositions
    with av_open(output_path) as container:
        assert len(container.streams.subtitles) == 2, f"Output should have 2 subtitle streams, got {len(container.streams.subtitles)}"

        en_sub_out = container.streams.subtitles[0]
        fi_sub_out = container.streams.subtitles[1]

        # Check that dispositions are preserved
        assert Disposition.default in en_sub_out.disposition, \
            f"English subtitle disposition not preserved: {en_sub_out.disposition}"
        assert fi_sub_out.disposition.value == 0, \
            f"Finnish subtitle disposition not preserved: {fi_sub_out.disposition}"

    # Run comprehensive disposition check
    check_stream_dispositions(input_path, output_path)


def test_mkv_attachment_preservation() -> None:
    """Verify that attachment streams are preserved during smart_cut operations."""
    input_path = make_video_with_attachment('test_mkv_with_attachment.mkv')
    output_path = test_mkv_attachment_preservation.__name__ + '.mkv'

    source = MediaContainer(input_path)
    segments = [(Fraction(0), source.duration)]
    smart_cut(source, segments, output_path, log_level='warning')
    source.close()

    input_attachments = get_attachment_stream_metadata(input_path)
    output_attachments = get_attachment_stream_metadata(output_path)

    assert input_attachments, "Source file is expected to carry at least one attachment"
    assert input_attachments == output_attachments, "Attachment streams metadata mismatch after smart_cut"


def test_no_discard_flag_on_output_packets() -> None:
    """Verify that smartcut output never has video packets with discard flag set.

    Regression test for a bug where the last frame in smartcut output had
    AV_PKT_FLAG_DISCARD set, causing decoders to skip it and show "No frame available".

    The bug manifests when the cut endpoint is at a time that causes the last frame
    to be included at the exact boundary. For example, cutting to 81.033s (which is
    81s + 1 frame at 30fps) causes frame 2431 to be included but marked with discard.

    The key to reproducing is using an MP4 output (which uses edit lists) and
    cutting to a time just beyond a frame boundary.
    """
    input_path = 'test_discard_flag_input.mkv'
    create_test_video(input_path, 30, 'h264', 'yuv420p', 30, (256, 144))

    source = MediaContainer(input_path)

    # Cut to a time just past a frame boundary (e.g., 20 seconds + epsilon)
    # At 30fps, each frame is 1/30 = 0.0333... seconds
    # Cutting to 20.034s should include 601 frames (0-600, where frame 600 is at 20.0s)
    output_path = test_no_discard_flag_on_output_packets.__name__ + '.mp4'
    segments = [(Fraction(0), Fraction(20034, 1000))]  # 20.034 seconds

    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)
    smart_cut(source, segments, output_path, video_settings=video_settings, log_level='warning')

    # The key assertion: no packets should have discard flag
    check_no_discard_packets(output_path)

    source.close()


def test_no_discard_flag_multiple_cuts() -> None:
    """Verify no discard flags when cutting multiple segments from a video.

    Tests that the discard flag issue doesn't appear with more complex
    cut patterns involving multiple segments, including segments that end
    just past a frame boundary.
    """
    input_path = 'test_discard_flag_multi_input.mkv'
    create_test_video(input_path, 30, 'h264', 'yuv420p', 30, (256, 144))

    source = MediaContainer(input_path)

    # Multiple segments with gaps - tests segment boundaries
    # Use times that are just past frame boundaries (at 30fps, frame duration is ~0.0333s)
    output_path = test_no_discard_flag_multiple_cuts.__name__ + '.mp4'
    segments = [
        (Fraction(2), Fraction(8034, 1000)),   # Ends just past 8s
        (Fraction(12), Fraction(18034, 1000)), # Ends just past 18s
        (Fraction(22), Fraction(28034, 1000)), # Ends just past 28s - this is the final segment
    ]

    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)
    smart_cut(source, segments, output_path, video_settings=video_settings, log_level='warning')

    check_no_discard_packets(output_path)

    source.close()


def get_test_categories() -> dict[str, list]:
    """
    Returns a dictionary of test categories.
    """
    test_categories = {
        # Core synthetic test categories (generated test videos)
        'basic': [
            test_cfr_recode_preserves_decode_delay,
            test_h264_cut_on_keyframes,
            test_h264_smart_cut,
            test_mp4_cut_on_keyframe,
            test_mp4_smart_cut,
            test_no_discard_flag_on_output_packets,
            test_no_discard_flag_multiple_cuts,
        ],

        'h264': [
            test_h264_multiple_cuts,
            test_h264_profile_baseline,
            test_h264_profile_main,
            test_h264_profile_high,
            test_h264_profile_high10,
            test_h264_profile_high422,
            test_h264_profile_high444,
            test_h264_non_idr_keyframes,
            test_h264_non_idr_keyframes_annexb,
            test_mp4_to_mkv_smart_cut,
            test_mkv_to_mp4_smart_cut,
            test_ts_h264_to_mp4_cut_on_keyframes,
            test_ts_h264_to_mkv_smart_cut,
        ],

        'h265': [
            test_h265_cut_on_keyframes,
            test_h265_smart_cut,
            test_mp4_h265_smart_cut,
            test_peaks_mkv_memory_usage,
            test_ts_h265_to_mp4_smart_cut,
            test_hevc_mkv_nal_type_detection,
            test_hevc_mkv_no_dts_remux,
        ],

        'codecs': [
            test_vp9_smart_cut,
            test_vp9_profile_1,
            test_av1_smart_cut,
            test_mpg_cut_on_keyframes,
            test_mpg_smart_cut,
            test_m2ts_mpeg2_smart_cut,
            # test_m2ts_h264_smart_cut,
            test_ts_smart_cut,
        ],

        'containers': [
            test_avi_mpeg4_smart_cut,
            test_avi_mjpeg_smart_cut,
            test_avi_h263_smart_cut,
            test_avi_to_mkv_smart_cut,
            test_flv_smart_cut,
            test_mov_smart_cut,
            test_wmv_smart_cut,
            test_ts_h264_to_mp4_cut_on_keyframes,
            test_ts_h264_to_mp4_smart_cut,
            test_ts_h264_to_mkv_smart_cut,
            test_mkv_attachment_preservation,
        ],

        'audio': [
            test_vorbis_passthru,
            test_mp3_passthru,
        ],

        'mixed': [
            test_mkv_with_video_and_audio_passthru,
            test_subtitle_disposition_preservation,
            test_multiple_language_subtitles,
        ],

        'transforms': [
            # test_vertical_transform, # removed - video transform feature removed
            test_video_recode_codec_override,
        ],

        'long': [
            test_h264_24_fps_long,
            test_h264_1080p,
            test_h265_smart_cut_large,
        ],

        'external': [
            test_night_sky,
            test_night_sky_to_mkv,
            test_sunset,
            test_seeking,
        ],

        # Real-world test categories (using actual videos from the wild)
        'real_world_h264': [
            # Google Common Data Storage videos (H.264)
            test_google_bigbuckbunny,
            test_google_elephantsdream,
            test_google_forbiggerblaze,
            test_google_forbiggeresc,
            test_google_subaru,
            test_google_tears_of_steel,
            # test-videos.co.uk H.264 samples
            test_testvideos_bigbuckbunny_h264,
            test_testvideos_jellyfish_h264,
        ],

        'real_world_h265': [
            # test-videos.co.uk H.265/HEVC samples
            test_testvideos_bigbuckbunny_h265,
            test_testvideos_jellyfish_h265,
            test_libde265_tears_of_steel_h265,
            test_h265_zcam,
            test_h265_zcam_containers,
        ],

        'real_world_vp9': [
            # test-videos.co.uk VP9 samples
            test_testvideos_bigbuckbunny_vp9,
        ],

        # Meta-categories that combine multiple subcategories
        'real_world': [
            # This will be populated dynamically by combining all real_world_* categories
        ],
        'manual': [
            test_manual
        ]
    }

    # Populate meta-categories dynamically
    real_world_categories = [key for key in test_categories if key.startswith('real_world_')]
    for category in real_world_categories:
        test_categories['real_world'].extend(test_categories[category])

    return test_categories

def run_tests(category: str | None = None, single_test: str | None = None, flaky_runs: int | None = None, base_seed: int | None = None) -> None:
    """
    Runs tests from specified category, single test, or all tests.
    """
    test_categories = get_test_categories()

    if single_test:
        # Find the specific test function
        all_tests = []
        for cat_tests in test_categories.values():
            all_tests.extend(cat_tests)

        # Remove duplicates
        seen = set()
        unique_tests = []
        for test in all_tests:
            if test not in seen:
                seen.add(test)
                unique_tests.append(test)

        # Find the test by name
        target_test = None
        for test in unique_tests:
            if test.__name__ == single_test:
                target_test = test
                break

        if target_test is None:
            print(f"Test function '{single_test}' not found.")
            print("Available test functions:")
            for test in sorted(unique_tests, key=lambda t: t.__name__):
                print(f"  {test.__name__}")
            return

        tests_to_run = [target_test]
        print(f"Running single test: {single_test}")

    elif category and category != 'all':
        if category not in test_categories:
            print(f"Unknown category: {category}")
            print("Available categories:", list(test_categories.keys()))
            return

        tests_to_run = test_categories[category]
        print(f"Running {category} tests ({len(tests_to_run)} tests)")
    else:
        # Run all tests
        tests_to_run = []
        test_categories.pop('manual')
        for cat_tests in test_categories.values():
            tests_to_run.extend(cat_tests)

        # Remove duplicates while preserving order
        seen = set()
        unique_tests = []
        for test in tests_to_run:
            if test not in seen:
                seen.add(test)
                unique_tests.append(test)
        tests_to_run = unique_tests

        print(f"Running all tests ({len(tests_to_run)} tests)")

    if flaky_runs is not None and flaky_runs < 1:
        print("--flaky requires a positive integer")
        return

    perf_timer = time.time()
    passed = 0
    failed = 0

    total_runs = flaky_runs if flaky_runs else 1
    seed_value = base_seed if base_seed is not None else DEFAULT_SEED

    if not flaky_runs:
        seed_all(seed_value)

    for test in tests_to_run:
        test_name = test.__name__

        for run_index in range(total_runs):
            run_seed = seed_value if not flaky_runs else (seed_value + run_index) & 0x7FFFFFFF
            seed_all(run_seed)

            try:
                # print(f"Running {test_name}")
                test()
                if flaky_runs:
                    print(f"{test_name} [{run_index + 1}/{total_runs}] seed={run_seed}: ✅ PASS")
                else:
                    print(f"{test_name}: ✅ PASS")
                passed += 1
            except Exception:
                if flaky_runs:
                    print(f"{test_name} [{run_index + 1}/{total_runs}] seed={run_seed}: ❌ FAIL")
                else:
                    print(f"{test_name}: ❌ FAIL:")
                traceback.print_exc()
                failed += 1
                # Continue running remaining iterations to gather full flake info

    elapsed = time.time() - perf_timer
    print(f'\nResults: {passed} passed, {failed} failed')
    print(f'Tests ran in {elapsed:0.1f}s')

if __name__ == "__main__":
    args = parse_args()
    base_seed = resolve_base_seed(args)

    if args.files:
        # Test specific files
        seed_all(base_seed)
        for file_path in args.files:
            try:
                print(f"Testing {file_path}")
                run_smartcut_test(file_path, os.path.split(file_path)[1], 2, audio_export_info='auto', pixel_tolerance=args.pixel_tolerance)
                print(f"Done: {file_path}")
            except Exception:
                print(f"Fail: {file_path}")
                traceback.print_exc()
    else:
        # Run test categories or single tests
        run_tests(args.category, args.single, args.flaky, base_seed)
