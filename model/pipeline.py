import torch
from diffusers import StableDiffusionInpaintPipeline

def load_pipeline(device):
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    ).to(device)

    pipe.safety_checker = None
    return pipe