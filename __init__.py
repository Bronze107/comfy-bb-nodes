import os
import sys

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# The directory name "comfy-bb-nodes" is not a valid Python module name, so the
# package cannot use relative imports. Expose the sibling modules via sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_nodes import LoadWildcardFile, RandomFromList, WildcardReplace
from video_nodes import VideoInfo, ConcatVideos, ResampleFPS, SplitVideoByFrames
from image_nodes import ScaleImageToTotalPixelsCoverCrop
from qwen_nodes import TextEncodeQwenImage21Latent


class BBExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadWildcardFile,
            RandomFromList,
            WildcardReplace,
            VideoInfo,
            ConcatVideos,
            ResampleFPS,
            SplitVideoByFrames,
            ScaleImageToTotalPixelsCoverCrop,
            TextEncodeQwenImage21Latent,
        ]


async def comfy_entrypoint() -> BBExtension:
    return BBExtension()
