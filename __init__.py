import json
import os
import random
import re
import shutil
import subprocess
import uuid

from typing_extensions import override

from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, io, Types, ui
from folder_paths import (
    base_path,
    exists_annotated_filepath,
    get_annotated_filepath,
    get_output_directory,
    get_save_image_path,
    get_temp_directory,
)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    if exists_annotated_filepath(path):
        return get_annotated_filepath(path)
    return get_annotated_filepath(path, base_path)


def load_wildcard_lines(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError(f"No valid entries in wildcard file: {path}")
    return lines


def pick_random_line(lines, seed):
    if not lines:
        raise ValueError("Cannot pick from an empty list")
    return random.Random(seed).choice(lines)


def wildcard_replace(text, file_paths, seed):
    rng = random.Random(seed)
    lines = [load_wildcard_lines(resolve_path(p)) for p in file_paths]

    def pick(match):
        idx = int(match.group(1))
        if idx < 1 or idx > len(lines):
            raise ValueError(
                f"Placeholder __{idx}__ has no matching file (only {len(lines)} file(s))"
            )
        return rng.choice(lines[idx - 1])

    return re.sub(r"__(\d+)__", pick, text)


class LoadWildcardFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadWildcardFile",
            display_name="Load Wildcard File",
            category="text",
            description="Read a text file into a list of lines, skipping blank lines and # comments.",
            inputs=[
                io.String.Input("file_path", tooltip="Path to the wildcard text file."),
            ],
            outputs=[
                io.String.Output(
                    display_name="lines",
                    is_output_list=True,
                    tooltip="List of lines from the file.",
                ),
            ],
        )

    @classmethod
    def execute(cls, file_path):
        return io.NodeOutput(load_wildcard_lines(resolve_path(file_path)))


class RandomFromList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RandomFromList",
            display_name="Random From List",
            category="text",
            description="Pick a random entry from a list, seeded for reproducibility.",
            is_input_list=True,
            inputs=[
                io.String.Input("strings", tooltip="List of strings to pick from.", force_input=True),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed."
                ),
            ],
            outputs=[
                io.String.Output(tooltip="Randomly picked string."),
            ],
        )

    @classmethod
    def execute(cls, strings, seed):
        return io.NodeOutput(pick_random_line(strings, seed[0]))


class WildcardReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        file_template = io.Autogrow.TemplatePrefix(
            io.String.Input("file_path"),
            prefix="file_path",
            min=0,
        )
        return io.Schema(
            node_id="WildcardReplace",
            display_name="Wildcard Replace",
            category="text",
            description="Replace __1__, __2__, ... placeholders in the template with random lines from the corresponding files.",
            inputs=[
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Autogrow.Input("file_paths", template=file_template),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed."
                ),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, text, file_paths, seed):
        return io.NodeOutput(wildcard_replace(text, list(file_paths.values()), seed))


class VideoInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoInfo",
            display_name="Video Info",
            category="video",
            description="Get the duration and frame count of a video.",
            inputs=[
                io.Video.Input("video", tooltip="The video to inspect."),
            ],
            outputs=[
                io.Float.Output(display_name="duration", tooltip="Duration in seconds."),
                io.Int.Output(display_name="frame_count", tooltip="Total number of frames."),
            ],
        )

    @classmethod
    def execute(cls, video):
        return io.NodeOutput(video.get_duration(), video.get_frame_count())


def _find_ffmpeg_tool(name):
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(
            f"{name} not found on PATH. Install FFmpeg (ffmpeg + ffprobe) and add it to PATH."
        )
    return exe


def _run_process(cmd):
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
    return result


def _materialize_video(video, temp_dir):
    """Return a real file path for a VIDEO input, writing a temp file when it is not already on disk."""
    if isinstance(video, VideoFromFile):
        source = video.get_stream_source()
        if isinstance(source, (str, os.PathLike)):
            return os.path.abspath(os.fspath(source)), False
    path = os.path.join(temp_dir, f"concat_input_{uuid.uuid4().hex}.mp4")
    video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    return path, True


def _probe_video(path):
    result = _run_process(
        [_find_ffmpeg_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", path]
    )
    streams = json.loads(result.stdout).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream found in {path}")
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    return video, audio


def _stream_signature(video, audio):
    video_sig = (
        video.get("codec_name"),
        video.get("width"),
        video.get("height"),
        video.get("pix_fmt"),
        video.get("avg_frame_rate"),
    )
    audio_sig = tuple(
        (a.get("codec_name"), a.get("sample_rate"), a.get("channels")) for a in audio
    )
    return video_sig, audio_sig


def _concat_list_line(path):
    # The concat demuxer wants forward slashes; single quotes are backslash-escaped.
    return "file '" + path.replace("\\", "/").replace("'", "\\'") + "'"


def _write_concat_list(paths, temp_dir):
    list_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(_concat_list_line(path) + "\n")
    return list_path


def _output_path(filename_prefix, ref_video):
    width = ref_video.get("width") or 1
    height = ref_video.get("height") or 1
    full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(
        filename_prefix, get_output_directory(), width, height
    )
    file = f"{filename}_{counter:05}_.mp4"
    return os.path.join(full_output_folder, file), file, subfolder


def _concat_stream_copy(paths, filename_prefix, ref_video, temp_dir):
    list_path = _write_concat_list(paths, temp_dir)
    out_path, file, subfolder = _output_path(filename_prefix, ref_video)
    try:
        _run_process(
            [
                _find_ffmpeg_tool("ffmpeg"),
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                out_path,
            ]
        )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    return out_path, file, subfolder


def _concat_reencode(paths, filename_prefix, probes, temp_dir):
    ref_video, ref_audio = probes[0]
    width = ref_video.get("width")
    height = ref_video.get("height")
    fps = ref_video.get("avg_frame_rate") or "30"
    if not width or not height:
        raise RuntimeError("Could not determine the reference video's dimensions.")

    n = len(paths)
    cmd = [_find_ffmpeg_tool("ffmpeg"), "-y"]
    for path in paths:
        cmd.extend(["-i", path])

    filter_parts = []
    video_labels = []
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p,settb=AVTB[v{i}]"
        )
        video_labels.append(f"[v{i}]")

    has_audio = all(len(audio) > 0 for _, audio in probes)
    audio_labels = []
    if has_audio:
        ref_audio_stream = ref_audio[0]
        sample_rate = ref_audio_stream.get("sample_rate") or 44100
        layout = ref_audio_stream.get("channel_layout") or "stereo"
        for i in range(n):
            filter_parts.append(
                f"[{i}:a]aresample={sample_rate},aformat=channel_layouts={layout}[a{i}]"
            )
            audio_labels.append(f"[a{i}]")

    concat_inputs = "".join(video_labels + audio_labels)
    filter_parts.append(
        f"{concat_inputs}concat=n={n}:v=1:a={1 if has_audio else 0}[outv]"
        + ("[outa]" if has_audio else "")
    )
    cmd.extend(["-filter_complex", ";".join(filter_parts), "-map", "[outv]"])
    if has_audio:
        cmd.extend(["-map", "[outa]", "-c:a", "aac"])
    cmd.extend(
        ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )

    out_path, file, subfolder = _output_path(filename_prefix, ref_video)
    cmd.append(out_path)
    _run_process(cmd)
    return out_path, file, subfolder


class ConcatVideos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        video_template = io.Autogrow.TemplatePrefix(
            io.Video.Input("video"),
            prefix="video",
            min=2,
            max=50,
        )
        return io.Schema(
            node_id="ConcatVideos",
            display_name="Concat Videos",
            category="video",
            search_aliases=["concatenate videos", "merge videos", "join videos", "splice videos"],
            description="Concatenate multiple videos into one with FFmpeg. Uses lossless stream copy when all inputs share the same encoding, resolution, and frame rate; otherwise re-encodes to the first video's parameters.",
            is_output_node=True,
            inputs=[
                io.Autogrow.Input("videos", template=video_template),
                io.String.Input(
                    "filename_prefix",
                    default="video/ConcatVideo",
                    tooltip="Prefix for the output filename in the output directory.",
                ),
            ],
            outputs=[
                io.Video.Output(tooltip="Concatenated video."),
            ],
        )

    @classmethod
    def execute(cls, videos: io.Autogrow.Type, filename_prefix):
        videos = list(videos.values())
        temp_dir = get_temp_directory()
        paths = []
        temp_paths = []
        try:
            for video in videos:
                path, is_temp = _materialize_video(video, temp_dir)
                paths.append(path)
                if is_temp:
                    temp_paths.append(path)

            probes = [_probe_video(path) for path in paths]
            signatures = [_stream_signature(video, audio) for video, audio in probes]
            if len(set(signatures)) == 1:
                out_path, file, subfolder = _concat_stream_copy(paths, filename_prefix, probes[0][0], temp_dir)
            else:
                out_path, file, subfolder = _concat_reencode(paths, filename_prefix, probes, temp_dir)
            return io.NodeOutput(
                VideoFromFile(out_path),
                ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
            )
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass


class ResampleFPS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResampleFPS",
            display_name="Resample FPS",
            category="video",
            search_aliases=["adjust fps", "change frame rate", "convert fps", "set fps"],
            description="Re-encode a video to a new frame rate with FFmpeg's fps filter, which duplicates or drops frames while keeping the duration.",
            is_output_node=True,
            inputs=[
                io.Video.Input("video", tooltip="The video to resample."),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.01,
                    max=240.0,
                    step=0.01,
                    tooltip="Target frame rate in frames per second.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="video/ResampleFPS",
                    tooltip="Prefix for the output filename in the output directory.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video, fps, filename_prefix):
        temp_dir = get_temp_directory()
        path, is_temp = _materialize_video(video, temp_dir)
        temp_paths = [path] if is_temp else []
        try:
            ref_video, audio = _probe_video(path)
            out_path, file, subfolder = _output_path(filename_prefix, ref_video)
            cmd = [
                _find_ffmpeg_tool("ffmpeg"),
                "-y",
                "-i", path,
                "-vf", f"fps={fps:g}",
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            if audio:
                cmd.extend(["-c:a", "aac"])
            cmd.append(out_path)
            _run_process(cmd)
            return io.NodeOutput(
                VideoFromFile(out_path),
                ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
            )
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass


class WildcardExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadWildcardFile,
            RandomFromList,
            WildcardReplace,
            VideoInfo,
            ConcatVideos,
            ResampleFPS,
        ]


async def comfy_entrypoint() -> WildcardExtension:
    return WildcardExtension()
