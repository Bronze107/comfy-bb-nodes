import folder_paths
import comfy.sd
import comfy.utils
from comfy_api.latest import io


def parse_lora_spec(spec):
    # one entry per '|' separated segment: "name:strength", strength optional and defaults to 1.0
    entries = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        name, sep, strength = part.rpartition(":")
        if not sep:
            name, strength = part, "1.0"
        entries.append((name.strip(), float(strength)))
    return entries


class MultiLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MultiLoraLoader",
            display_name="Multi LoRA Loader",
            category="model/loaders",
            description="Load multiple LoRAs in one node. One entry per LoRA as 'name:strength', "
            "separated by '|', e.g. 'detail_tweak.safetensors:0.8 | style_x.safetensors:0.5'. "
            "An entry without ':strength' uses strength 1.0.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("lora_spec", multiline=True,
                                placeholder="detail_tweak.safetensors:0.8 | style_x.safetensors:0.5",
                                tooltip="LoRA names are relative to models/loras, one 'name:strength' entry per LoRA, separated by '|'. Entries without ':strength' use strength 1.0."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_spec) -> io.NodeOutput:
        model_lora = model
        for name, strength in parse_lora_spec(lora_spec):
            if strength == 0:
                continue
            lora_path = folder_paths.get_full_path_or_raise("loras", name)
            lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            model_lora, _ = comfy.sd.load_lora_for_models(model_lora, None, lora, strength, 0, lora_metadata=metadata)
        return io.NodeOutput(model_lora)
