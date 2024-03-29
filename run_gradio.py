import gradio as gr
from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
import torch 
import numpy as np
import cv2
from PIL import Image, ImageFilter
from interface.extension import CustomStableDiffusionControlNetPipeline

negative_prompt = ""
device = torch.device('cuda')
controlnet = ControlNetModel.from_pretrained("partialsketchcontrolnet", torch_dtype=torch.float16).to(device)
pipe = CustomStableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=torch.float16
).to(device)
pipe.safety_checker = None
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
threshold = 250
curr_num_samples = 2

all_gens = []

num_images = 5

with gr.Blocks() as demo:
    start_state = []
    with gr.Row():
        with gr.Column():
            with gr.Row():
                stroke_type = gr.Radio(["Blocking", "Detail"], value="Detail", label="Stroke Type"),
                dilation_strength = gr.Slider(7, 117, value=65, step=2, label="Dilation Strength"),
            canvas = gr.Image(source="canvas", shape=(512, 512), tool="color-sketch",
                        min_width=512, brush_radius = 2).style(width=512, height=512)
            prompt_box = gr.Textbox(width="50vw", label="Prompt")
            with gr.Row():
                btn = gr.Button("Generate").style(width=100, height=80)
                btn2 = gr.Button("Reset").style(width=100, height=80)
        with gr.Column():
            num_samples = gr.Slider(1, 5, value=2, step=1, label="Num Samples to Generate"),
            with gr.Tab("Renoised Images"):
                gallery0 = gr.Gallery(show_label=False, columns=[num_samples[0].value], rows=[2], object_fit="contain", height="auto", preview=True, interactive=False).style(width=512, height=512)
            with gr.Tab("Renoised Overlay"):
                gallery1 = gr.Gallery(show_label=False, columns=[num_samples[0].value], rows=[2], object_fit="contain", height="auto", preview=True, interactive=False).style(width=512, height=512)
            with gr.Tab("Pre-Renoise Images"):
                gallery2 = gr.Gallery(show_label=False, columns=[num_samples[0].value], rows=[2], object_fit="contain", height="auto", preview=True, interactive=False).style(width=512, height=512)
            with gr.Tab("Pre-Renoise Overlay"):
                gallery3 = gr.Gallery(show_label=False, columns=[num_samples[0].value], rows=[2], object_fit="contain", height="auto", preview=True, interactive=False).style(width=512, height=512)
    for k in range(num_images):
        start_state.append([None, None])
    sketch_states = gr.State(start_state)
    checkbox_state = gr.State(True)
        
    def sketch(curr_sketch_image, dilation_mask, prompt, seed, num_steps, dilation):
        global curr_num_samples
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(seed)

        negative_prompt = ""
        guidance_scale = 7
        controlnet_conditioning_scale = 1.0
        images = pipe([prompt]*curr_num_samples, [curr_sketch_image.convert("RGB").point( lambda p: 256 if p > 128 else 0)]*curr_num_samples, guidance_scale=guidance_scale, controlnet_conditioning_scale = controlnet_conditioning_scale, negative_prompt = [negative_prompt] * curr_num_samples, num_inference_steps=num_steps, generator=generator, key_image=None, neg_mask=None).images

        # run blended renoising if blocking strokes are provided
        if dilation_mask is not None: 
            new_images = pipe.collage([prompt] * curr_num_samples, images, [dilation_mask] * curr_num_samples, num_inference_steps=50, strength=0.8)["images"]
        else:
            new_images = images
        return images, new_images

    def run_sketching(prompt, curr_sketch, sketch_states, dilation, contour_dilation=11):
        seed = sketch_states[k][1]
        if seed is None:
            seed = np.random.randint(1000)
            sketch_states[k][1] = seed

        curr_sketch_image = Image.fromarray(curr_sketch[:, :, 0]).resize((512, 512))

        curr_construction_image = Image.fromarray(255 - curr_sketch[:, :, 2] + curr_sketch[:, :, 0])
        if np.sum(255 - np.array(curr_construction_image)) == 0:
            curr_construction_image = None

        curr_detail_image = Image.fromarray(curr_sketch[:, :, 2]).resize((512, 512))

        if curr_construction_image is not None:
            dilation_mask = Image.fromarray(255 - np.array(curr_construction_image)).filter(ImageFilter.MaxFilter(dilation))
            dilation_mask = dilation_mask.point( lambda p: 256 if p > 0 else 25).filter(ImageFilter.GaussianBlur(radius = 5))

            neg_dilation_mask = Image.fromarray(255 - np.array(curr_detail_image)).filter(ImageFilter.MaxFilter(contour_dilation)) 
            neg_dilation_mask = np.array(neg_dilation_mask.point( lambda p: 256 if p > 0 else 0))
            dilation_mask = np.array(dilation_mask)
            dilation_mask[neg_dilation_mask > 0] = 25
            dilation_mask = Image.fromarray(dilation_mask).filter(ImageFilter.GaussianBlur(radius = 5))
        else:
            dilation_mask = None
        
        images, new_images = sketch(curr_sketch_image, dilation_mask, prompt, seed, num_steps = 40, dilation = dilation)

        save_sketch = np.array(Image.fromarray(curr_sketch).convert("RGBA"))
        save_sketch[:, :, 3][save_sketch[:, :, 0] > 128] = 0

        overlays = []
        for i in images:
            background = i.copy()
            background.putalpha(80)
            background = Image.alpha_composite(Image.fromarray(255 * np.ones((512, 512)).astype(np.uint8)).convert("RGBA"), background)
            overlay = Image.alpha_composite(background.resize((512, 512)), Image.fromarray(save_sketch).convert("RGBA"))
            overlays.append(overlay.convert("RGB"))

        new_overlays = []
        for i in new_images:
            background = i.copy()
            background.putalpha(80)
            background = Image.alpha_composite(Image.fromarray(255 * np.ones((512, 512)).astype(np.uint8)).convert("RGBA"), background)
            overlay = Image.alpha_composite(background.resize((512, 512)), Image.fromarray(save_sketch).convert("RGBA"))
            new_overlays.append(overlay.convert("RGB"))
        
        global all_gens
        all_gens = new_images

        return new_images, new_overlays, images, overlays

    def reset(sketch_states):
        for k in range(len(sketch_states)):
            sketch_states[k] = [None, None]
        return None, sketch_states
    
    def change_color(stroke_type):
        if stroke_type == "Blocking":
            color = "#0000FF"
        else:
            color = "#000000"
        return gr.Image(source="canvas", shape=(512, 512), tool="color-sketch",
                                min_width=512, brush_radius = 2, brush_color=color).style(width=400, height=400)
    
    def change_background(option):
        global all_gens
        if option == "None" or len(all_gens) == 0:
            return None
        elif option == "Sample 0":
            image_overlay = all_gens[0].copy()
        elif option == "Sample 1":
            image_overlay = all_gens[0].copy()
        else:
            return None
        image_overlay.putalpha(80)
        return image_overlay

    def change_num_samples(change):
        global curr_num_samples
        curr_num_samples = change
        return None
    
    btn.click(run_sketching, [prompt_box, canvas, sketch_states, dilation_strength[0]], [gallery0, gallery1, gallery2, gallery3])
    btn2.click(reset, sketch_states, [canvas, sketch_states])
    stroke_type[0].change(change_color, [stroke_type[0]], canvas)
    num_samples[0].change(change_num_samples, [num_samples[0]], None)


demo.launch(share = True, debug = True)