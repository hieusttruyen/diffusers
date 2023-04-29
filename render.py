import torch
from diffusers import StableDiffusionPipeline ,DPMSolverMultistepScheduler,StableDiffusionUpscalePipeline
import numpy as np
from dataclasses import dataclass
import contextlib
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from io import BytesIO
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
#torch.backends.cudnn.enabled = False

autocast = contextlib.nullcontext

np = "illustration, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, ((monochrome)), ((grayscale)), collapsed eyeshadow, multiple eyeblows, vaginas in breasts, (cropped), oversaturated, extra limb, missing limbs, deformed hands, long neck, long body, imperfect, (bad hands), signature, watermark, username, artist name, conjoined fingers, deformed fingers, ugly eyes, imperfect eyes, skewed eyes, unnatural face, unnatural body, error"




def render_x4():

    prompt= " 1girl, looking at viewer, upper body, 3D, realistic, excessively frilled princess dress, draped clothes, jewelry, ornament, flower, lace trim, masterpiece, best quality, 8k, detailed skin texture, detailed cloth texture, beautiful detailed face, intricate details, ultra detailed, rim lighting, side lighting, cinematic light, ultra high res, 8k uhd, film grain,best shadow, delicate, RAW"
    model_id = "stabilityai/stable-diffusion-x4-upscaler"

    # pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    #     model_id, revision="fp16", torch_dtype=torch.float16
    # )

    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id
    )
    
    pipeline = pipeline.to("cpu")
    
    # pipeline.unet.set_attn_processor(AttnProcessor2_0())
    # pipeline.unet.to(memory_format=torch.channels_last) 
    # pipeline.enable_sequential_cpu_offload()

    pipeline.enable_attention_slicing()

    # pipeline.enable_vae_slicing()
    # pipeline.enable_xformers_memory_efficient_attention()

    #pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

   # im = Image.open(r"test.png") 
    low_res_img = Image.open(r"test.png").convert("RGB")
    # low_res_img = low_res_img.resize((128, 128))
    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    upscaled_image.save("upsampled_cat.png")


def render(data,model,lora_path):
    data.prompt=" 1girl, looking at viewer, upper body, 3D, realistic, excessively frilled princess dress, draped clothes, jewelry, ornament, flower, lace trim, masterpiece, best quality, 8k, detailed skin texture, detailed cloth texture, beautiful detailed face, intricate details, ultra detailed, rim lighting, side lighting, cinematic light, ultra high res, 8k uhd, film grain,best shadow, delicate, RAW"

    if torch.cuda.is_available() == True:
        model.to("cuda")
    else:
        model.to("cpu")
    model.unet.load_attn_procs(lora_path)
    model.unet.set_attn_processor(AttnProcessor2_0())
    model.unet.to(memory_format=torch.channels_last) 
    # model.enable_sequential_cpu_offload()
    model.enable_attention_slicing()
    model.enable_vae_slicing()
    model.enable_xformers_memory_efficient_attention()

    with torch.no_grad():
        image = model(
        data.prompt,
        num_inference_steps=20,
        num_images_per_prompt=1,
        height=768,
        width=512,
        negative_prompt=np,
        guidance_scale=7,
    ).images[0]
    image.save("test.png")
    print(image)
    return image

def render_cpu(
    model_id="DreamFul-V2",
    prompt="",
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    negative_prompt="",
    num_images_per_prompt=1,
):
    negative_prompt =np
    # print(prompt,height,negative_prompt)
    # ddim = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, safety_checker=None
        )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.to("cpu")

    # generator = torch.Generator(device="cpu").manual_seed(33)
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
  
    with torch.inference_mode():
        image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    ).images[0]
        
    try:
        image.save("test.png")
        return True
    except:
        return False



def render_gpu(
    model_id="DreamFul-V2",
    prompt="",
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    negative_prompt="",
    num_images_per_prompt=1,
):
    negative_prompt =np
    # print(prompt,height,negative_prompt)
    ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, safety_checker=None, scheduler=ddim
        ).to("cuda")
    @dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = pipe.unet.in_channels
            self.device = pipe.unet.device

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)
        
    pipe.unet = TracedUNet()

    # generator = torch.Generator(device="cpu").manual_seed(33)
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
  
    with torch.inference_mode():
        image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    ).images[0]
        
    try:
        image.save("test.png")
        return True
    except:
        return False


# print(image)
