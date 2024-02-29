from flask import Flask, request, jsonify, render_template
import uuid
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import base64
import datetime
import cv2
import glob 
import time
import json
from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler, LCMScheduler
import torch
import numpy as np 
from PIL import Image, ImageFilter
import os
from extension import CustomStableDiffusionControlNetPipeline
import argparse
import open_clip
import shutil
from concurrent.futures import ThreadPoolExecutor, wait
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="../partialsketchcontrolnet")
parser.add_argument("--port", type=int, default=9000)
parser.add_argument("--ui_only", action="store_true") # default = False
args = parser.parse_args()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

# This folder will store the uploaded images
PREV_UPLOAD_FOLDER = 'static/previous_uploads'
app.config['PREV_UPLOAD_FOLDER'] = PREV_UPLOAD_FOLDER

os.makedirs("./static/previous_uploads", exist_ok=True)
os.makedirs("./static/generated", exist_ok=True)
os.makedirs("./static/sketch", exist_ok=True)

# default parameters
curr_out_dir = ""
cluster_mode = "pix"
num_images = 2
num_gpus = torch.cuda.device_count()
batch_size = num_images // (num_gpus)
generating = False
curr_fidelity = "low"
curr_model = "our"
fidelity_list = ["low", "mid", "high"]
fidelity_filters = {"high":1, "mid":100, "low":200}
mask_size = {"high":200, "mid":1000, "low":2000}
selected_reference_image_path = "" # used for transforming the reference image into strokes
session_id = "" # used for keeping track of the current session
session_prompt = "none"
scheduler = "euler"

# load models only if not in ui debugging mode
if (not args.ui_only): 
    model_paths = {"our" : args.model,
                "orig" : "lllyasviel/sd-controlnet-scribble"}
    existing = []
    hide = ["dummy"]

    text_size, hole_scale, island_scale=512,100,100
    text, text_part, text_thresh='','','0.0'

    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)

    negative_prompt = ""
    device = torch.device('cuda')
    controlnets = [] 
    pipes = []
    segmenters = {"low":[], "mid":[], "high":[]}
    for i in range(num_gpus):                              
        controlnets.append(ControlNetModel.from_pretrained(model_paths["our"], torch_dtype=torch.float16).to(f"cuda:{i}"))
        pipes.append(CustomStableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnets[i], torch_dtype=torch.float16
        ).to(f"cuda:{i}"))
    for i in range(num_gpus):
        if scheduler == "euler":
            pipes[i].safety_checker = None
            pipes[i].scheduler = EulerAncestralDiscreteScheduler.from_config(pipes[i].scheduler.config)
        else:
            pipes[i].scheduler = LCMScheduler.from_config(pipes[i].scheduler.config)
            pipes[i].load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    threshold = 250
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

def sketch(construction, curr_sketch_image, dilation_mask, pipe_id, prompt, seed, negative_prompt, num_steps, guidance_scale, controlnet_conditioning_scale, strength):
    generator = torch.Generator(device=f"cuda:{pipe_id}")
    generator.manual_seed(seed)
    start = time.time()

    # generate initial images
    global batch_size
    images = pipes[pipe_id]([prompt]*batch_size, [curr_sketch_image.convert("RGB").point( lambda p: 256 if p > 128 else 0)]*batch_size, guidance_scale=guidance_scale, controlnet_conditioning_scale = controlnet_conditioning_scale, negative_prompt = [negative_prompt] * batch_size, num_inference_steps=num_steps, generator=generator, key_image=None, neg_mask=None).images
    
    # run blended renoising if blocking strokes are provided
    if construction: 
        new_image_2 = pipes[pipe_id].collage([prompt] * batch_size, images, [dilation_mask] * batch_size, num_inference_steps=50, strength=strength)["images"]
    else:
        new_image_2 = images
    print("time one image", time.time()-start)

    return {"im": images, "idx": pipe_id, "reshape_im2": new_image_2}

def run_sketching(construction, curr_sketch_image, dilation_mask, prompt, sketch_path, strength=0.8, guidance_scale=1.0, num_steps=4, controlnet_conditioning_scale = 0.5):
    seeds = range(num_gpus)
    with ThreadPoolExecutor() as executor:
        futures = []

        # run image generation and renoising
        for k in range(num_gpus):
            futures.append(executor.submit(sketch, construction, curr_sketch_image, dilation_mask, k, prompt, seeds[k] * 10, negative_prompt, num_steps, guidance_scale, controlnet_conditioning_scale, strength))
        complete_futures, incomplete_futures = wait(futures)
        all_images = [0] * num_images
        reshape_images2 = [0] * num_images

        # collect all generated images
        for f in complete_futures:
            result = f.result()
            im = result["im"]
            _idx = result["idx"]
            all_images[_idx * batch_size : (_idx+1) * batch_size] = im
            reshape_images2[_idx * batch_size : (_idx+1) * batch_size] = result["reshape_im2"]
            
    return Image.open(sketch_path).convert("RGB"), all_images, reshape_images2

def generate_session_id():
    return str(uuid.uuid4())

@app.route('/')
def home():

    # each time we load the page, we generate a new session id
    global session_id
    session_id =  generate_session_id()
    global session_prompt
    session_prompt = "none"
    global app

    # folder for saving generated results and sketches for visualization
    SAVE_OVERLAY_FOLDER = f"./static/saved_overlay/{session_prompt}_{session_id}"
    app.config['SAVE_OVERLAY_FOLDER'] = SAVE_OVERLAY_FOLDER 

    return render_template('index.html')


@app.route('/save-drawing', methods=['POST'])
def save_drawing():
    drawing_data = request.json
    pixel_data = drawing_data["data"] 
    construction_data = drawing_data["construction"] 
    detail_data = drawing_data["detail"] 
    
    # Save the pixel data as an image file
    image_data = pixel_data.split(",")[1]
    construction_data = construction_data.split(",")[1]
    detail_data = detail_data.split(",")[1]
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = secure_filename(f'drawing-{time_stamp}.png')
    construction_filename = secure_filename(f'construction-drawing-{time_stamp}.png')
    detail_filename = secure_filename(f'detail-drawing-{time_stamp}.png')

    filepath = os.path.join(app.config['PREV_UPLOAD_FOLDER'], filename)
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(image_data))

    filepath = os.path.join(app.config['PREV_UPLOAD_FOLDER'], construction_filename)
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(construction_data))

    filepath = os.path.join(app.config['PREV_UPLOAD_FOLDER'], detail_filename)
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(detail_data))
        
        
    return jsonify({'imageUrl': filepath, "drawid":filename.split(".")[0]})

@app.route('/gpu-count', methods=['GET'])
def gpu_count():
    return jsonify({'num_gpus': torch.cuda.device_count()})

@app.route('/clear-all', methods=['GET'])
def clear_all():
    global existing 
    existing = []
    images = glob.glob(os.path.join(app.config['PREV_UPLOAD_FOLDER'], "*.jpg")) + glob.glob(os.path.join(app.config['PREV_UPLOAD_FOLDER'], "*.png"))
    for im in images:
        os.remove(im)
    return jsonify(images) 

@app.route('/remove-cluster', methods=['POST'])
def remove_cluster():
    path = request.get_json()['path']
    global hide
    if path.split("?")[0] in hide:
        hide.remove(path.split("?")[0])
    else:
        hide.append(path.split("?")[0])
    return jsonify(path)

@app.route('/set-cluster', methods=['POST'])
def set_cluster():
    cluster = request.get_json()['cluster']
    global cluster_mode
    cluster_mode = cluster
    print(cluster_mode)
    return jsonify(cluster) 

@app.route('/get-gen-images', methods=['POST'])
def get_gen_images():

    # retrieve generated image paths
    vis_mode = request.get_json()['vis_mode']

    if vis_mode == "image":
        return_dict = {'hide': hide, 'labels':["0"]}
        return_dict[f"cluster0"] = [sorted(glob.glob(f"{curr_out_dir}/gen_*.png"))]
        return jsonify(return_dict)
    
    if vis_mode == "reshape_image2":
        return_dict = {'hide': hide, 'labels':["0"]}
        return_dict[f"cluster0"] = [sorted(glob.glob(f"{curr_out_dir}/reshape2_*.png"))]
        return jsonify(return_dict)

    return jsonify(return_dict)

@app.route('/get-images', methods=['GET'])
def get_images():
    global existing
    images = glob.glob(os.path.join(app.config['PREV_UPLOAD_FOLDER'], "*.png")) + glob.glob(os.path.join(app.config['PREV_UPLOAD_FOLDER'], "sketch.png"))
    for im in images:
        if os.path.exists(im):
            Image.fromarray(255 - np.array(Image.open(im))[:, :, -1]).convert("RGB").resize((256, 256)).save(im.replace(".png", ".png"))
    image_urls = [file.replace(".png", ".png") for file in images if file.replace(".png", ".png") not in existing]
    existing += image_urls
    return jsonify(image_urls)


@app.route('/generate-images', methods=['POST'])
def generate_images():
    start = time.time()
    global generating
    print("is generating", generating)
    if generating:
        return jsonify({"generating": 1})
    generating = True
    
    # get inputs and parameters
    prompt = request.get_json()['prompt']
    model = request.get_json()['model']
    strength = float(request.get_json()['strength'])
    dilation = int(request.get_json()['dilation'])
    contour_dilation = int(request.get_json()['contour_dilation'])
    new_scheduler = request.get_json()['scheduler']
    
    global curr_fidelity
    curr_fidelity = "low"
    global curr_model
    global num_images
    global batch_size
    global scheduler
    num_images = int(request.get_json()['num_samples'])
    batch_size = int(num_images) // (num_gpus)

    # load model only if changed
    if model != curr_model:
        global controlnets
        global pipes
        del controlnets
        del pipes
        controlnets = []
        pipes = []
        for i in range(num_gpus):
            controlnets.append(ControlNetModel.from_pretrained(model_paths[model], torch_dtype=torch.float16).to(f"cuda:{i}"))
            pipes.append(CustomStableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnets[i], torch_dtype=torch.float16
                ).to(f"cuda:{i}"))
        for i in range(num_gpus):
            pipes[i].safety_checker = None
            if scheduler == "euler":
                pipes[i].scheduler = EulerAncestralDiscreteScheduler.from_config(pipes[i].scheduler.config)
            else:
                pipes[i].scheduler = LCMScheduler.from_config(pipes[i].scheduler.config)
                pipes[i].load_lora_weights("latent-consistency/lcm-lora-sdv1-5")       
        curr_model = model

        # change scheduler only if changed
        if new_scheduler != scheduler:
            scheduler = new_scheduler
            if scheduler == "euler":
                pipes[i].scheduler = EulerAncestralDiscreteScheduler.from_config(pipes[i].scheduler.config)
            else:
                pipes[i].scheduler = LCMScheduler.from_config(pipes[i].scheduler.config)
                pipes[i].load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

    guidance_scale = float(request.get_json()['guidancescale'])
    controlnet_conditioning_scale = float(request.get_json()['controlnetscale'])
    num_steps = int(request.get_json()['numsteps'])
    print(prompt[-3:], prompt[:-3])
    if prompt[-3:] == " | ":
        prompt = prompt[:-3]
    print("Generating images for prompt:", prompt)

    global session_prompt
    if session_prompt == "none":
        session_prompt = "_".join(prompt.split(" "))
        app.config['SAVE_OVERLAY_FOLDER'] = app.config['SAVE_OVERLAY_FOLDER'].replace("none", session_prompt)
    
    folder = "./examples/"+"_".join(prompt.split(" "))
    os.makedirs(folder, exist_ok=True)

    paths = glob.glob(os.path.join("./static/previous_uploads/drawing*.png"))
    # also get pngs
    paths += glob.glob(os.path.join("./static/previous_uploads/drawing*.png"))
    print("Has input sketch of layers:", len(paths))
    if (len(paths) == 0):
        print("No input sketch, exiting")
        generating = False
        return jsonify({})

    subdir = "_".join(prompt.split(" "))
    out_dir = f"./static/examples/{subdir}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    global curr_out_dir
    curr_out_dir = out_dir
    
    # consturct input sketch and dilation mask
    input_sketch = np.zeros((512, 512))
    for path in paths:
        input_sketch += (np.array(Image.open(path).resize((512, 512), resample=0))[:, :, -1])
    input_sketch[input_sketch > 255] = 255
    Image.fromarray(255 - input_sketch).convert("RGB").save("./static/sketch/input.png")
    Image.fromarray(255 - np.array(Image.open(glob.glob(os.path.join("./static/previous_uploads/construction*.png"))[0]).resize((512, 512), resample=0))[:, :, -1]).convert("RGB").save("./static/sketch/construction.png")
    Image.fromarray(255 - np.array(Image.open(glob.glob(os.path.join("./static/previous_uploads/construction*.png"))[0]).resize((512, 512), resample=0))[:, :, -1]).convert("RGB").save(f"{folder}/construction.png")
    Image.fromarray(255 - np.array(Image.open(glob.glob(os.path.join("./static/previous_uploads/detail*.png"))[0]).resize((512, 512), resample=0))[:, :, -1]).convert("RGB").save("./static/sketch/detail.png")
    Image.fromarray(255 - np.array(Image.open(glob.glob(os.path.join("./static/previous_uploads/detail*.png"))[0]).resize((512, 512), resample=0))[:, :, -1]).convert("RGB").save(f"{folder}/detail.png")
    Image.fromarray(input_sketch).convert("RGB").save(f"{folder}/input.png")

    Image.fromarray(np.zeros((512, 512)).astype(np.uint8)).save(f"./static/sketch/contour_dilation.png")
    Image.fromarray(np.zeros((512, 512)).astype(np.uint8)).save(f"./static/sketch/dilation.png")
    subdir = "_".join(prompt.split(" "))
    out_dir = f"./static/examples/{subdir}"
    os.makedirs(out_dir, exist_ok=True)
    curr_sketch_image = Image.fromarray(np.array(Image.open("./static/sketch/input.png").convert("L"))).resize((512, 512))
    curr_sketch = 255 - np.expand_dims(np.array(curr_sketch_image).astype(np.uint8), axis=-1).repeat(3, axis=-1)
    curr_sketch = cv2.dilate(curr_sketch, np.ones((2, 2), np.uint8), iterations=1)
    curr_sketch = cv2.medianBlur(curr_sketch, 3)
    curr_sketch = (curr_sketch + cv2.GaussianBlur(curr_sketch, (3, 3), 10))[:, :, 0]
    if curr_model == "our":
        curr_sketch_image = Image.fromarray((255 - curr_sketch).astype(np.uint8)).convert("L")
    else:
        curr_sketch_image = Image.fromarray((curr_sketch).astype(np.uint8)).convert("L")
    curr_sketch_image.resize((512, 512)).save(f"./static/examples/{subdir}/input.png")
    curr_sketch_image.resize((512, 512)).save(f"./examples/{subdir}/input.png")

    construction = True
    dilation_mask = None
    curr_construction_image = Image.fromarray(np.array(Image.open("./static/sketch/input.png".replace("input", "construction")).convert("L"))).resize((512, 512))
    if np.sum(255 - np.array(curr_construction_image)) == 0:
        construction = False
    curr_construction = 255 - np.expand_dims(np.array(curr_construction_image).astype(np.uint8), axis=-1).repeat(3, axis=-1)
    curr_construction = cv2.dilate(curr_construction, np.ones((2, 2), np.uint8), iterations=1)
    curr_construction = cv2.medianBlur(curr_construction, 3)
    curr_construction = (curr_construction + cv2.GaussianBlur(curr_construction, (3, 3), 10))[:, :, 0]
    curr_construction_image = Image.fromarray((255 - curr_construction).astype(np.uint8)).convert("L")
    curr_construction_image.resize((512, 512)).save(f"./static/examples/{subdir}/construction.png")
    curr_construction_image.resize((512, 512)).save(f"./examples/{subdir}/construction.png")

    curr_detail_image = Image.fromarray(np.array(Image.open("./static/sketch/input.png".replace("input", "detail")).convert("L"))).resize((512, 512))
    curr_detail = 255 - np.expand_dims(np.array(curr_detail_image).astype(np.uint8), axis=-1).repeat(3, axis=-1)
    curr_detail = cv2.dilate(curr_detail, np.ones((2, 2), np.uint8), iterations=1)
    curr_detail = cv2.medianBlur(curr_detail, 3)
    curr_detail = (curr_detail + cv2.GaussianBlur(curr_detail, (3, 3), 10))[:, :, 0]
    curr_detail_image = Image.fromarray((255 - curr_detail).astype(np.uint8)).convert("L")
    curr_detail_image.resize((512, 512)).save(f"./static/examples/{subdir}/detail.png")
    curr_detail_image.resize((512, 512)).save(f"./examples/{subdir}/detail.png")
    if construction:
        dilation_mask = Image.fromarray(255 - np.array(curr_construction_image)).filter(ImageFilter.MaxFilter(dilation))
        dilation_mask = dilation_mask.point( lambda p: 256 if p > 0 else 25).filter(ImageFilter.GaussianBlur(radius = 5))

        neg_dilation_mask = Image.fromarray(255 - np.array(curr_detail_image)).filter(ImageFilter.MaxFilter(contour_dilation)) 
        neg_dilation_mask = np.array(neg_dilation_mask.point( lambda p: 256 if p > 0 else 0))
        neg_dilation_mask_img = Image.fromarray(neg_dilation_mask).filter(ImageFilter.GaussianBlur(radius = 5))
        neg_dilation_mask_img.save(f"./static/examples/{subdir}/contour_dilation.png")
        neg_dilation_mask_img.save(f"./static/sketch/contour_dilation.png")
        dilation_mask = np.array(dilation_mask)
        dilation_mask[neg_dilation_mask > 0] = 25
        dilation_mask = Image.fromarray(dilation_mask).filter(ImageFilter.GaussianBlur(radius = 5))
        dilation_mask.save(f"./static/examples/{subdir}/dilation.png")
        dilation_mask.save(f"./static/sketch/dilation.png")

    # generate images
    out_sketch, images, reshape_images2 = run_sketching(construction, curr_sketch_image, dilation_mask, prompt,"./static/sketch/input.png", strength=strength, guidance_scale=guidance_scale, controlnet_conditioning_scale = controlnet_conditioning_scale, num_steps=num_steps)

    for i in range(len(images)):
        images[i].save(f"./static/generated/{i}.png")
        images[i].resize((512, 512)).save(f"{curr_out_dir}/gen_{i}.png")
        reshape_images2[i].resize((512, 512)).save(f"{curr_out_dir}/reshape2_{i}.png")

    subdir = "_".join(prompt.split(" "))

    Image.open(f"./examples/{subdir}/input.png").resize((512, 512)).save(f"{curr_out_dir}/input.png")

    return_dict = {"generating": 0}
    return_dict[f"cluster0"] = [sorted(glob.glob(f"{out_dir}/gen_*.png"))]
    print("gen took", time.time() - start)
    generating = False
    return jsonify(return_dict)

@app.route('/update-selected-ref-img', methods=['POST'])
def update_selected_ref_img():
    global selected_reference_image_path
    selected_reference_image_path = request.get_json()['path']
    selected_reference_image_path = selected_reference_image_path.split("?")[0]
    
    print("Selected reference image path:", selected_reference_image_path)
    return jsonify({"success": 1})

@app.route("/save-overlay", methods=["POST"])
def save_overlay():
    prompt = os.path.basename(request.get_json()['prompt'])

    # reconstruct colored input sketch
    try:
        sketch = Image.open(glob.glob("./static/previous_uploads/detail*.png")[0]).convert("RGBA")
        sketch = np.array(sketch)
    except:
        sketch = (np.zeros((512, 512, 4)) * 255).astype(np.uint8)
    construction = Image.open(glob.glob("./static/previous_uploads/construction*.png")[0]).convert("RGBA")
    save_sketch = np.array(construction)
    save_sketch[:, :, 0] = 0
    save_sketch[:, :, 2] = 0
    save_sketch[:, :, 1][save_sketch[:, :, -1] != 0] = 255
    save_sketch[:, :, 1][sketch[:, :, -1] != 0] = 0
    save_sketch[:, :, 2][sketch[:, :, -1] != 0] = 0
    save_sketch[:, :, 0][sketch[:, :, -1] != 0] = 0
    save_sketch[:, :, -1][sketch[:, :, -1] != 0] = sketch[:, :, -1][sketch[:, :, -1] != 0]
    
    # create save folder
    os.makedirs(app.config['SAVE_OVERLAY_FOLDER'], exist_ok=True)
    subdirs = os.listdir(app.config['SAVE_OVERLAY_FOLDER'])
    num = len(subdirs)
    os.makedirs(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}"), exist_ok=True)
    save_sketch1 = Image.alpha_composite(Image.fromarray(255 * np.ones((512, 512)).astype(np.uint8)).convert("RGBA"), Image.fromarray(save_sketch).point( lambda p: 256 if p > 0 else 0).convert("RGBA"))
    save_sketch1.convert("RGB").save(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}", "sketch.png"))
    
    # reconstruct input sketch
    try:
        sketch = Image.open(glob.glob("./static/previous_uploads/detail*.png")[0]).filter(ImageFilter.MaxFilter(size = 5)).point( lambda p: 256 if p > 0 else 0).convert("RGBA")
        sketch = np.array(sketch)
    except:
        sketch = (np.zeros((512, 512, 4)) * 255).astype(np.uint8)
    construction = Image.open(glob.glob("./static/previous_uploads/construction*.png")[0]).filter(ImageFilter.MaxFilter(size = 5)).point( lambda p: 256 if p > 0 else 0).convert("RGBA")
    save_sketch = np.array(construction)
    save_sketch[:, :, 0] = 0
    save_sketch[:, :, 2] = 0
    save_sketch[:, :, 1][save_sketch[:, :, -1] != 0] = 255
    save_sketch[:, :, 1][sketch[:, :, -1] != 0] = 0
    save_sketch[:, :, 2][sketch[:, :, -1] != 0] = 0
    save_sketch[:, :, 0][sketch[:, :, -1] != 0] = 0
    save_sketch[:, :, -1][sketch[:, :, -1] != 0] = sketch[:, :, -1][sketch[:, :, -1] != 0]
    
    # save sketch with thicker strokes for better visualization
    save_sketch1 = Image.alpha_composite(Image.fromarray(255 * np.ones((512, 512)).astype(np.uint8)).convert("RGBA"), Image.fromarray(save_sketch).point( lambda p: 256 if p > 0 else 0).convert("RGBA"))
    save_sketch1.convert("RGB").save(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}", "thick_sketch.png"))

    save_sketch = np.array(Image.open(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}", "thick_sketch.png")).convert("RGBA"))
    save_sketch[:, :, 3] = 255 - save_sketch[:, :, 0]

    p = {"prompt": prompt}

    # save prompt
    with open(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}", "prompt.json"), "w") as f:
        json.dump(p, f)

    # overlay sketch on generated images from current and previous steps
    paths = glob.glob(f"{curr_out_dir}/reshape2*")
    for idx in range(len(paths)):
        p = paths[idx]
        _idx = p.split("_")[-1].split(".")[0]
        background = Image.open(p)
        background.putalpha(80)
        background = Image.alpha_composite(Image.fromarray(255 * np.ones((512, 512)).astype(np.uint8)).convert("RGBA"), background)
        overlay = Image.alpha_composite(background.resize((512, 512)), Image.fromarray(save_sketch).convert("RGBA"))
        overlay.convert("RGB").save(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}", f"overlay_curr_{_idx}.png"))

    if num >= 1: 
        for p in glob.glob(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num-1}", "all_vis/partialSketchCNet*.png")):
            _idx = p.split("_")[-1].split(".")[0]
            background = Image.open(p)

            background.putalpha(80)
            background = Image.alpha_composite(Image.fromarray(255 * np.ones((512, 512)).astype(np.uint8)).convert("RGBA"), background)
            overlay = Image.alpha_composite(background.resize((512, 512)), Image.fromarray(save_sketch).convert("RGBA"))
            overlay.convert("RGB").save(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}", f"overlay_prev_{_idx}.png"))
    
    # save all generated images
    os.makedirs(os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}/all_vis"), exist_ok=True)
    paths = glob.glob(f"{curr_out_dir}/reshape2*") + glob.glob(f"{curr_out_dir}/gen*") + glob.glob(f"{curr_out_dir}/dilation*")

    for p in paths:
        Image.open(p).save(p.replace(f"{curr_out_dir}", os.path.join(app.config['SAVE_OVERLAY_FOLDER'], f"{num}/all_vis")).replace("reshape2", "partialSketchCNet"))

    return jsonify({"name":f"{os.path.basename(curr_out_dir)}_{num}.zip", "link": os.path.join(os.getcwd(), app.config['SAVE_OVERLAY_FOLDER'], f"{os.path.basename(curr_out_dir)}_{num}.zip")})

if __name__ == '__main__':
    app.run(debug=True, port=args.port, host="0.0.0.0")#, use_reloader=False)
