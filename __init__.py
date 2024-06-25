import folder_paths
import os


def _add_folder_path(folder_name: str, extensions: list):
    path = os.path.join(folder_paths.models_dir, folder_name)
    _, current_extensions = folder_paths.folder_names_and_paths.setdefault(
        folder_name, ([path], set())
    )
    if isinstance(current_extensions, set):
        current_extensions.update(extensions)
    elif isinstance(current_extensions, list):
        current_extensions.extend(extensions)
    else:
        e = f"Failed to register models/inpaint folder. Found existing value: {current_extensions}"
        raise Exception(e)


_add_folder_path("inpaint", [".pt", ".pth", ".safetensors", ".patch"])

from . import nodes

NODE_CLASS_MAPPINGS = {
    "INPAINT_VAEEncode": nodes.VAEEncode,
    "INPAINT_VAEDecode": nodes.VAEDecode,
    "INPAINT_ColorCorrection": nodes.ColorCorrection,
    "ImagePreprocess": nodes.ImagePreprocess,
    "ImagePostprocess": nodes.ImagePostprocess,
    "INPAINT_LoadModel": nodes.LoadModel,
    "INPAINT_InpaintingWithModel": nodes.InpaintingWithModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INPAINT_VAEEncode": "VAE Encode Inpaint",
    "INPAINT_VAEDecode": "VAE Decode Inpaint",
    "INPAINT_ColorCorrection": "ColorCorrection Inpaint",
    "ImagePreprocess": "ImagePreprocess Inpaint",
    "ImagePostprocess": "ImagePostprocess Inpaint",
    "INPAINT_LoadModel": "Load Model Inpaint",
    "INPAINT_InpaintingWithModel": "Inpainting (using Model)",
}
