import os
import glob
import json
import sys
import numpy as np
import torch
from PIL import Image, ImageFilter, PngImagePlugin
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import folder_paths
import comfy.controlnet
import comfy.sample
import comfy.utils
import latent_preview
from nodes import MAX_RESOLUTION

#---------------------------------------------------------------------------------------------------------------------#

class RoundFloat2String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"label": "float to Round", "forceInput": True}),
                "string": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "round_float_to_string"

    CATEGORY = "Text/Utilities"

    def round_float_to_string(self, float, string):
        rounded_float = round(float, 2)
        output_string = f"{string}{rounded_float}"
        return (output_string,)

#---------------------------------------------------------------------------------------------------------------------#

class SaveImageToFolder:
    def __init__(self):
        # Initialize with your default output directory if necessary
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4  # Set compression level if you want to customize this

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE',),
                'folder_path': ('STRING', {'default': 'OutputFolder'}),  # Custom folder path
                'filename': ('STRING', {'default': 'Image'}),  # Custom filename
                'disable_preview': ('BOOLEAN', {'default': False})  # Option to disable preview
            },
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'},  # Keep hidden inputs for metadata
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_to_folder"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images_to_folder(self, images, folder_path='OutputFolder', filename='Image', disable_preview=False, prompt=None, extra_pnginfo=None):
        full_output_folder = os.path.join(self.output_dir, folder_path)
        os.makedirs(full_output_folder, exist_ok=True)
        results = []

        for i, image in enumerate(images):
            img_array = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            metadata = PngInfo()  # Initialize metadata

            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))  # Add prompt to metadata
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))  # Add extra PNG info to metadata

            # If only one image, save with 'filename.png', else include an index
            file_name = f"{filename}.png" if len(images) == 1 else f"{filename}_{i + 1}.png"
            img.save(os.path.join(full_output_folder, file_name), pnginfo=metadata, compress_level=self.compress_level)
            
            if not disable_preview:
                results.append({'filename': file_name, 'subfolder': folder_path, 'type': self.type})

        return {'ui': {'images': results}} if not disable_preview else {}


#---------------------------------------------------------------------------------------------------------------------#

class FixCheckpointName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fix_name"

    CATEGORY = "Text Processing"

    @classmethod
    def fix_name(cls, input_text):
        # Extract the substring after the last backslash `\`
        last_backslash_index = input_text.rfind('\\') + 1
        # Extract the substring before '.safetensors'
        safetensors_index = input_text.find('.safetensors')
        # Combine the two operations above to extract the desired substring
        output_text = input_text[last_backslash_index:safetensors_index]
        return (output_text,)

#---------------------------------------------------------------------------------------------------------------------#

class SelectControlNet:
    
    @classmethod
    def INPUT_TYPES(cls):
        controlnet_files = ["None"] + folder_paths.get_filename_list("controlnet")
        
        return {"required": {"control_net_name1": (controlnet_files,),
                             "control_net_name2": (controlnet_files,),
                             "control_net_name3": (controlnet_files,),
                             "control_net_name4": (controlnet_files,),
                             "control_net_name5": (controlnet_files,),
                             "select_controlnet": ("INT", {"default": 1, "min": 1, "max": 5}),
                            }    
               }

    RETURN_TYPES = ("CONTROL_NET", "STRING")
    RETURN_NAMES = ("CONTROL_NET", "control_net_name")
    FUNCTION = "select_controlnet"

    def select_controlnet(self, control_net_name1, control_net_name2, control_net_name3, control_net_name4, control_net_name5, select_controlnet):
            
        if select_controlnet == 1:
            control_net_name = control_net_name1
        elif select_controlnet == 2:
            control_net_name = control_net_name2
        elif select_controlnet == 3:
            control_net_name = control_net_name3
        elif select_controlnet == 4:
            control_net_name = control_net_name4
        elif select_controlnet == 5:
            control_net_name = control_net_name5
            
        if control_net_name == "None":
            print(f"CR Select ControlNet: No ControlNet selected")
            return None, "None"

        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            
        return controlnet, control_net_name

#---------------------------------------------------------------------------------------------------------------------#

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class SimpleLoadImageBatch:
    # Store the state as a class attribute
    index = 0
    image_paths = []
    current_path = ""

    def __init__(self):
        # Initialize the class
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
                "allow_RGBA_output": (["true","false"],),
                "reset": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEED", "STRING")
    FUNCTION = "load_images"

    CATEGORY = "Simplified/IO"

    @classmethod
    def load_images(cls, path, allow_RGBA_output='false', reset=False, seed=0):
        if reset or path != cls.current_path:
            cls.index = 0
            cls.image_paths = []
            cls.current_path = path

        if cls.index == 0 or cls.index >= len(cls.image_paths) - 2:
            cls.image_paths = sorted(glob.glob(os.path.join(path, '*.png')))

        if cls.index >= len(cls.image_paths):
            return (None, {"seed": cls.index}, "")

        image = Image.open(cls.image_paths[cls.index])
        if allow_RGBA_output != 'true':
            image = image.convert("RGB")

        file_name = os.path.basename(cls.image_paths[cls.index])
        file_name_without_extension = os.path.splitext(file_name)[0]

        cls.index += 1
        return (pil2tensor(image), {"seed": cls.index}, file_name_without_extension)

#---------------------------------------------------------------------------------------------------------------------#

def is_noise_identical(noise_batch):
    first_noise = noise_batch[0]
    for i in range(1, noise_batch.size(0)):
        if not torch.equal(first_noise, noise_batch[i]):
            return False
    return True

def prepare_common_noise(latent_image, seed):
    """
    Creates the same random noise for all images in a latent image batch given a seed.
    """
    generator = torch.manual_seed(seed)
    noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    noise_batch = noise.repeat(latent_image.size(0), 1, 1, 1)
    return noise_batch

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        noise = prepare_common_noise(latent_image, seed)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

    # Compare the latent tensors before outputting
    identical_latents = is_noise_identical(samples)
    print(f"All latent tensors in the batch are identical: {identical_latents}")

    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

#---------------------------------------------------------------------------------------------------------------------#

class Batch12Images:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                f"image_{i+1}": ("IMAGE",) for i in range(12)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_images"
    CATEGORY = "image/processing"

    def batch_images(self, **kwargs):
        images = [kwargs[f"image_{i+1}"].squeeze(0) for i in range(12)]  # Remove extra batch dimension
        batched_images = torch.stack(images, dim=0)  # Stack images into a batch
        return (batched_images,)

    def prepare_for_preview(self, batched_images):
        # Extract the first image from the batch for preview
        image = batched_images[0]
        np_image = image.cpu().numpy().transpose((1, 2, 0))  # Convert to HxWxC format
        np_image = np.clip(np_image, 0, 1)  # Ensure values are within [0, 1] range
        pil_image = Image.fromarray((np_image * 255).astype(np.uint8))  # Convert to uint8 and create PIL Image
        return pil_image

#---------------------------------------------------------------------------------------------------------------------#

def composite(layer_0, layer_1, x, y, mask = None, multiplier = 8, resize_layer_1 = False):
    if resize_layer_1:
        layer_1 = torch.nn.functional.interpolate(layer_1, size=(layer_0.shape[2], layer_0.shape[3]), mode="bilinear")

    layer_1 = comfy.utils.repeat_to_batch_size(layer_1, layer_0.shape[0])

    x = max(-layer_1.shape[3] * multiplier, min(x, layer_0.shape[3] * multiplier))
    y = max(-layer_1.shape[2] * multiplier, min(y, layer_0.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + layer_1.shape[3], top + layer_1.shape[2],)

    if mask is None:
        mask = torch.ones_like(layer_1)
    else:
        mask = mask.clone()
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(layer_1.shape[2], layer_1.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, layer_1.shape[0])

    # calculate the bounds of the layer_1 that will be overlapping the layer_0
    # this prevents the layer_1 trying to overwrite latent pixels that are out of bounds
    # of the layer_0
    visible_width, visible_height = (layer_0.shape[3] - left + min(0, x), layer_0.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    layer_1_portion = mask * layer_1[:, :, :visible_height, :visible_width]
    layer_0_portion = inverse_mask  * layer_0[:, :, top:bottom, left:right]

    layer_0[:, :, top:bottom, left:right] = layer_1_portion + layer_0_portion
    return layer_0

class ImageLinearGammaCompositeMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layer_1": ("IMAGE",),
                "layer_0": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_layer_1": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image"

    def composite(self, layer_0, layer_1, x, y, resize_layer_1, mask=None):
        # Linearize gamma before compositing
        layer_0 = torch.pow(layer_0, 2.2)
        layer_1 = torch.pow(layer_1, 2.2)

        layer_0 = layer_0.clone().movedim(-1, 1)
        output = composite(layer_0, layer_1.movedim(-1, 1), x, y, mask, 1, resize_layer_1).movedim(1, -1)

        # Apply inverse gamma after compositing
        output = torch.pow(output, 1/2.2)
        return (output,)

#---------------------------------------------------------------------------------------------------------------------#

class MaskGlow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("INT", {"default": 0, "min": 0, "max": 255}),
                "glow_size": ("INT", {"default": 100, "min": 1, "max": 256}),
                "subsample": ("INT", {"default": 3, "min": 1, "max": 5}),
                "blur_radius": ("FLOAT", {"default": 5.5, "min": 0, "max": 10}),
                "overlay_original": ("BOOLEAN", {"default": True}),
                "fadeout": ("FLOAT", {"default": 7, "min": 0, "max": 100}),
                "fadeout_minimum_glow": ("INT", {"default": 5, "min": 1, "max": 256})
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_glow"
    CATEGORY = "Custom/mask"

    def apply_glow(self, mask, glow_size=64, subsample=1, blur_radius=0, overlay_original=False, threshold=128, fadeout=0, fadeout_minimum_glow=5):
        original_height, original_width = mask.shape[-2:]
        new_height, new_width = original_height // subsample, original_width // subsample

        mask_image = Image.fromarray(np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255)).convert("L")
        original_thresholded = mask_image.point(lambda x: 255 if x > threshold else 0)
        mask_image = mask_image.resize((new_width, new_height), Image.BILINEAR)
        np_glow_image = np.array(mask_image)

        np_glow_image = np.where(np_glow_image > threshold, 255, 0)
        white_pixels = np.sum(np_glow_image == 255)
        total_pixels = np_glow_image.size
        white_pixel_percentage = (white_pixels / total_pixels) * 100

        if fadeout > 0 and white_pixel_percentage < fadeout:
            fadeout_factor = fadeout_minimum_glow + ((glow_size - fadeout_minimum_glow) * (white_pixel_percentage / fadeout))
            glow_size = max(int(fadeout_factor), fadeout_minimum_glow)

        expanded_mask = np_glow_image > 0
        adjusted_glow_size = max(1, glow_size // subsample)
        intensity_reduction = 255.0 / (adjusted_glow_size * subsample)

        for i in range(adjusted_glow_size):
            new_mask = np.array(mask_image.filter(ImageFilter.MaxFilter(3)).convert("L")) > 0
            new_expanded_pixels = np.logical_and(new_mask, np.logical_not(expanded_mask))
            reduced_intensity = np.clip(255 - intensity_reduction * (i * subsample + 1), 0, 255)
            np_glow_image[new_expanded_pixels] = reduced_intensity
            expanded_mask = new_mask.copy()
            mask_image = Image.fromarray(np_glow_image)

        glow_image = Image.fromarray(np_glow_image).resize((original_width, original_height), Image.BILINEAR).convert("L")

        if blur_radius > 0:
            glow_image = glow_image.filter(ImageFilter.GaussianBlur(blur_radius))

        if overlay_original:
            np_glow_image = np.array(glow_image)
            np_original_thresholded = np.array(original_thresholded)
            np_glow_image = np.maximum(np_glow_image, np_original_thresholded)
            glow_image = Image.fromarray(np_glow_image)

        return torch.from_numpy(np.array(glow_image).astype(np.float32) / 255.0).unsqueeze(0)

#---------------------------------------------------------------------------------------------------------------------#

class OffsetImage:
    """
    Custom node to perform an offset operation on an image.
    This node offsets an image by the specified horizontal and vertical amounts.
    Assumes the image tensor format is (batch, height, width, channels).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "horizontal_offset": ("INT", {
                    "default": 0,
                    "min": -sys.maxsize,
                    "max": sys.maxsize,
                    "step": 1
                }),
                "vertical_offset": ("INT", {
                    "default": 0,
                    "min": -sys.maxsize,
                    "max": sys.maxsize,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "offset_image"
    CATEGORY = "image/transform"

    def offset_image(self, image, horizontal_offset, vertical_offset):
        if image is None:
            return None

        if isinstance(image, torch.Tensor) and len(image.shape) == 4:
            offset_image = torch.roll(image, shifts=(horizontal_offset, vertical_offset), dims=(2, 1))
            return (offset_image,)
        else:
            raise ValueError("Unsupported image format")

#---------------------------------------------------------------------------------------------------------------------#



NODE_CLASS_MAPPINGS = {
    "Round Float to String": RoundFloat2String,
    "SaveImageToFolder": SaveImageToFolder,
    "Fix Checkpoint Name": FixCheckpointName,
    "Select ControlNet": SelectControlNet,
    "Simple Load Image Batch": SimpleLoadImageBatch,
    "KSampler Same Noise": KSampler,
    "Batch 12 Images": Batch12Images,
    "ImageLinearGammaCompositeMasked": ImageLinearGammaCompositeMasked,
    "MaskGlow": MaskGlow,
    "OffsetImage": OffsetImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RoundFloat2String": "Round Float to String",
    "SaveImageToFolder": "Save Image to Folder",
    "FixCheckpointName": "Fix Checkpoint Name",
    "SelectControlNet": "Select ControlNet",
    "SimpleLoadImageBatch": "Simple Load Image Batch",
    "KSampler Same Noise": "KSamplerSameNoise",
    "Batch12Images": "Batch 12 Images",
    "ImageLinearGammaCompositeMasked": "Image Linear Gamma Composite Masked",
    "MaskGlow": "Apply Glow to Mask",
    "OffsetImage": "Offset Image",
}