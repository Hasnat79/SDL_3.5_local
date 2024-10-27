from huggingface_hub import login
login("HF_token")
import torch
import diffusers
import transformers
import accelerate
print("Torch Version",torch.__version__)
print("Diffusers Version",diffusers.__version__)
print("Transformers Version",transformers.__version__)
print("Accelerate Version",accelerate.__version__)
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", cache_dir="cache")
pipe.enable_model_cpu_offload()
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt,
num_inference_steps=30,
        guidance_scale=7.0).images[0]
image.save("output_1.png")

