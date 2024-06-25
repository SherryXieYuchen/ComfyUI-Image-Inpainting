import torch
import copy
import math
from spandrel import ModelLoader, ImageModelDescriptor
from torch import Tensor
from tqdm import trange

import folder_paths
import comfy.utils
from comfy import model_management
from comfy.model_management import get_torch_device
from .util import(
    to_torch,
    resize_square,
    undo_resize_square,
    to_comfy,
)

class VAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE",),
                             "vae": ("VAE",),
                             "mask": ("MASK",),
                             "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}), }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"

    def encode(self, vae, pixels, mask, grow_mask_by=6):
        x = (pixels.shape[1] // vae.downscale_ratio + 1) * vae.downscale_ratio
        y = (pixels.shape[2] // vae.downscale_ratio + 1) * vae.downscale_ratio
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        x_offset = (x - pixels.shape[1]) // 2 + 1
        y_offset = (y - pixels.shape[2]) // 2 + 1

        d1, d2, d3, d4 = pixels.size()
        new_pixels = torch.ones(
            (d1, x, y, d4),
            dtype=torch.float32
        ) * 0.5
        new_mask = torch.ones(
            (d1, 1, x, y),
            dtype=torch.float32
        ) * 0.5
        new_pixels[:, x_offset: pixels.shape[1] + x_offset, y_offset: pixels.shape[2] + y_offset, :] = pixels
        new_mask[:, :, x_offset: pixels.shape[1] + x_offset, y_offset: pixels.shape[2] + y_offset] = mask

        #grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = new_mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(new_mask.round(), kernel_tensor, padding=padding), 0,
                                       1)

        m = (1.0 - new_mask.round()).squeeze(1)
        for i in range(3):
            new_pixels[:, :, :, i] -= 0.5
            new_pixels[:, :, :, i] *= m
            new_pixels[:, :, :, i] += 0.5
        t = vae.encode(new_pixels)

        return ({"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())},)


class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",),
                             "vae": ("VAE",),
                             "pixels": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples, pixels):
        image = vae.decode(samples["samples"])
        x = (pixels.shape[1] // vae.downscale_ratio + 1) * vae.downscale_ratio
        y = (pixels.shape[2] // vae.downscale_ratio + 1) * vae.downscale_ratio
        pixels = pixels.clone()
        x_offset = (x - pixels.shape[1]) // 2 + 1
        y_offset = (y - pixels.shape[2]) // 2 + 1
        pixels = image[:, x_offset: pixels.shape[1] + x_offset, y_offset: pixels.shape[2] + y_offset, :]

        return (pixels,)


class ColorCorrection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "round": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image"

    def composite(self, image, reference, factor, mask=None):
        image = image.clone().movedim(-1, 1)  # [B x C x W x H]
        reference = reference.clone().movedim(-1, 1)  # [B x C x W x H]

        if mask is None:
            mask = torch.ones_like(image)
        else:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                                   size=(image.shape[2], image.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, image.shape[0])

        inverse_mask = torch.ones_like(mask) - mask
        image_masked = inverse_mask * image
        reference_masked = inverse_mask * reference

        t = copy.deepcopy(image)
        for c in range(t.size(1)):
            r_sd, r_mean = torch.std_mean(reference_masked[:, c, :, :], dim=None)  # index by original dim order
            i_sd, i_mean = torch.std_mean(image_masked[:, c, :, :], dim=None)

            t[:, c, :, :] = ((t[:, c, :, :] - i_mean) / i_sd) * r_sd + r_mean

        t = torch.lerp(image.movedim(1, -1), t.movedim(1, -1), factor)
        return (t,)


class ImagePreprocess:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "upscale_method": (s.upscale_methods,), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_preprocess"

    CATEGORY = "image/transform"

    def image_preprocess(self, image, upscale_method):
        samples = image.movedim(-1, 1)
        d1, d2, d3, d4 = samples.size()
        image_max_length = max(d3, d4)
        if image_max_length <= 1024:
            s = samples.movedim(1, -1)
        else:
            scale = 1024 / image_max_length
            width = round(samples.shape[3] * scale)
            height = round(samples.shape[2] * scale)
            s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
            s = s.movedim(1, -1)
        return (s,)


class ImagePostprocess:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"input_image": ("IMAGE",),
                             "output_image": ("IMAGE",),
                             "upscale_method": (s.upscale_methods,), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_postprocess"

    CATEGORY = "image/transform"

    @staticmethod
    def load_model(model_name):
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")
        return out

    @staticmethod
    def upscale(upscale_model, image):
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale,
                                                                        1.0) * 384.0  #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2],
                                                                            tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile,
                                            overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return s

    def rescale(self, model_name, image, image_max_length, upscale_method):
        upscale_model = self.load_model(model_name)
        upscale_image = self.rescale(upscale_model, image)
        upscale_samples = upscale_image.movedim(-1, 1)
        scale = image_max_length / max(upscale_samples.shape[2], upscale_samples.shape[3])
        width = round(upscale_samples.shape[3] * scale)
        height = round(upscale_samples.shape[2] * scale)
        s = comfy.utils.common_upscale(upscale_samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return s

    def image_postprocess(self, input_image, output_image, upscale_method):
        samples = input_image.movedim(-1, 1)
        d1, d2, d3, d4 = samples.size()
        image_max_length = max(d3, d4)
        if image_max_length <= 1024:
            s = output_image
        elif 1024 < image_max_length <= 2048:
            s = self.rescale('RealESRGAN_x2plus.pth', output_image, image_max_length, upscale_method)
        elif image_max_length > 2048:
            s = self.rescale('RealESRGAN_x4plus.pth', output_image, image_max_length, upscale_method)

        return (s,)


class LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("inpaint"),),
            }
        }

    RETURN_TYPES = ("INPAINT_MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "load"

    def load(self, model_name: str):
        model_file = folder_paths.get_full_path("inpaint", model_name)
        if model_file is None:
            raise RuntimeError(f"Model file not found: {model_name}")
        if model_file.endswith(".pt"):
            sd = torch.jit.load(model_file, map_location="cpu").state_dict()
        else:
            sd = comfy.utils.load_torch_file(model_file, safe_load=True)
        model = ModelLoader().load_from_state_dict(sd)
        model = model.eval()
        return (model,)


class InpaintWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpaint_model": ("INPAINT_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "optional_upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "inpaint"

    def inpaint(
        self,
        inpaint_model,
        image: Tensor,
        mask: Tensor,
        seed: int,
        optional_upscale_model=None,
    ):
        if inpaint_model.architecture.id == "LaMa":
            required_size = 1024
        else:
            raise ValueError(f"Unknown model_arch {type(inpaint_model)}")

        if optional_upscale_model != None:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

            upscaler = ImageUpscaleWithModel

        image, mask = to_torch(image, mask)
        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

        image_device = image.device
        device = get_torch_device()
        inpaint_model.to(device)
        batch_image = []
        pbar = comfy.utils.ProgressBar(batch_size)

        for i in trange(batch_size):
            work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
            work_image, work_mask, original_size = resize_square(
                work_image, work_mask, required_size
            )
            work_mask = work_mask.floor()

            torch.manual_seed(seed)
            work_image = inpaint_model(work_image.to(device), work_mask.to(device))

            if optional_upscale_model != None:
                work_image = work_image.movedim(1, -1)
                work_image = upscaler.upscale(upscaler, optional_upscale_model, work_image)
                work_image = work_image[0].movedim(-1, 1)

            work_image.to(image_device)
            work_image = undo_resize_square(work_image.to(image_device), original_size)
            work_image = image[i] + (work_image - image[i]) * mask[i].floor()

            batch_image.append(work_image)
            pbar.update(1)

        inpaint_model.cpu()
        result = torch.cat(batch_image, dim=0)
        return (to_comfy(result),)

