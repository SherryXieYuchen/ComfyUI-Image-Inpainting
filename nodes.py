import torch
import copy
import math
import cv2
from spandrel import ModelLoader, ImageModelDescriptor
from torch import Tensor
from tqdm import trange
import numpy as np

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
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "max_length": ("INT", {"default": 1024, "min": 512, "max": 1024, "step": 8}), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_preprocess"

    CATEGORY = "image/transform"

    def image_preprocess(self, image, max_length):
        samples = image.movedim(-1, 1)
        d1, d2, d3, d4 = samples.size()
        image_max_length = max(d3, d4)
        if image_max_length <= max_length:
            s = samples.movedim(1, -1)
        else:
            scale = max_length / image_max_length
            width = round(samples.shape[3] * scale)
            height = round(samples.shape[2] * scale)

            samples_numpy = samples.squeeze(0).detach().numpy().transpose(1, 2, 0)
            samples_numpy = cv2.resize(samples_numpy, (width, height), cv2.INTER_LANCZOS4)
            s = torch.tensor(samples_numpy.transpose(2, 0, 1)).unsqueeze(0)
            s = s.movedim(1, -1)
        return (s,)


class ImagePostprocess:
    def __init__(self) -> None:
        self.upscale_2x_model = self.load_model('RealESRGAN_x2plus.pth')
        # self.upscale_4x_model = self.load_model('RealESRGAN_x4plus.pth')
        # self.upscale_tiny_model = self.load_model('realesr-general-x4v3.pth')

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"input_image": ("IMAGE",),
                             "output_image": ("IMAGE",),
                             "max_length": ("INT", {"default": 1024, "min": 512, "max": 1024, "step": 8}), }}

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

    def rescale(self, upscale_model, image, image_max_length):
        upscale_image = self.upscale(upscale_model, image)
        upscale_samples = upscale_image.movedim(-1, 1)
        scale = image_max_length / max(upscale_samples.shape[2], upscale_samples.shape[3])
        width = round(upscale_samples.shape[3] * scale)
        height = round(upscale_samples.shape[2] * scale)
        upscale_samples_numpy = upscale_samples.squeeze(0).detach().numpy().transpose(1, 2, 0)
        upscale_samples_numpy = cv2.resize(upscale_samples_numpy, (width, height), cv2.INTER_LANCZOS4)
        s = torch.tensor(upscale_samples_numpy.transpose(2, 0, 1)).unsqueeze(0)
        s = s.movedim(1, -1)
        return s

    def image_postprocess(self, input_image, output_image, max_length):
        samples = input_image.movedim(-1, 1)
        d1, d2, d3, d4 = samples.size()
        image_max_length = max(d3, d4)
        if image_max_length <= max_length:
            s = output_image
        elif image_max_length > max_length:
            s = self.rescale(self.upscale_2x_model, output_image, image_max_length)
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


class InpaintingWithModel:
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


class CropImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "pad_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", "RECT", )
    CATEGORY = "image"
    FUNCTION = "crop_image"

    def crop_image(self, image, mask, pad_ratio):
        image_sample = image.movedim(-1, 1)
        image_numpy = image_sample.squeeze(0).detach().numpy().transpose(1, 2, 0)
        mask_numpy = (mask * 255).squeeze(0).detach().numpy().astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_numpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = []
        for cont in contours:
            contour.extend(cont)
        x, y, w, h = cv2.boundingRect(np.array(contour))
        
        # padding
        x_pad = int(w * pad_ratio // 2)
        y_pad = int(h * pad_ratio // 2)
        x_min = x - x_pad if x - x_pad > 0 else 0
        x_max = x + w + x_pad if x + w + x_pad < image_numpy.shape[1] else image_numpy.shape[1]
        y_min = y - y_pad if y - y_pad > 0 else 0
        y_max = y + h + y_pad if y + h + y_pad < image_numpy.shape[0] else image_numpy.shape[0]
        rect = (x_min, y_min, x_max, y_max)

        image_crop_numpy = image_numpy[y_min:y_max, x_min:x_max]
        image_crop = torch.tensor(image_crop_numpy.transpose(2, 0, 1)).unsqueeze(0)
        image_crop = image_crop.movedim(1, -1)

        mask_crop_numpy = mask_numpy[y_min:y_max, x_min:x_max]
        mask_crop = torch.tensor(mask_crop_numpy.astype(np.float32)).unsqueeze(0) / 255.0

        return (image_crop, mask_crop, rect, )


class CropImageByRect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image" : ("IMAGE",),
                "image": ("IMAGE",),
                "rect": ("RECT",),
            }
        }
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "image"
    FUNCTION = "crop_image"
    
    def crop_image(self, original_image, image, rect):
        x_min, y_min, x_max, y_max = rect
        image_sample = image.movedim(-1, 1)
        image_numpy = image_sample.squeeze(0).detach().numpy().transpose(1, 2, 0)

        image_numpy = cv2.resize(image_numpy, (original_image.shape[2], original_image.shape[1]))
        image_crop_numpy = image_numpy[y_min:y_max, x_min:x_max]
        image_crop = torch.tensor(image_crop_numpy.transpose(2, 0, 1)).unsqueeze(0)
        image_crop = image_crop.movedim(1, -1)

        return (image_crop, )


class PasteBackCropImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "rect": ("RECT",),
            }
        }
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "image"
    FUNCTION = "paste_back_crop_image"

    def paste_back_crop_image(self, image, crop_image, rect):
        image_sample = image.movedim(-1, 1)
        image_numpy = image_sample.squeeze(0).detach().numpy().transpose(1, 2, 0)

        crop_image_sample = crop_image.movedim(-1, 1)
        crop_image_numpy = crop_image_sample.squeeze(0).detach().numpy().transpose(1, 2, 0)

        x_min, y_min, x_max, y_max = rect
        image_numpy[y_min:y_max, x_min:x_max] = crop_image_numpy
        image_update = torch.tensor(image_numpy.transpose(2, 0, 1)).unsqueeze(0)
        image_update = image_update.movedim(1, -1)

        return (image_update, )


