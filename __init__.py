import os
import random

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from folder_paths import base_path, exists_annotated_filepath, get_annotated_filepath


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


class WildcardExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadWildcardFile,
            RandomFromList,
        ]


async def comfy_entrypoint() -> WildcardExtension:
    return WildcardExtension()
