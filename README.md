# Block and Detail: Scaffolding Sketch-to-Image Generation

[Live Demo](https://huggingface.co/spaces/BlockDetail/Block-and-Detail)

We introduce a novel sketch-to-image tool that aligns with the iterative refinement process of artists. 
Our tool lets users sketch **blocking** strokes to coarsely represent the placement and form of objects and **detail** strokes to refine their shape and silhouettes. 

![alt text](https://github.com/BlockDetail/Block-and-Detail/blob/main/images/teaser.png?raw=true)

# Setup

With conda installed, create a new environment with the following commands:

```
cd Block-and-Detail
conda env create -f environment.yml
```

# Download Pretrained PartialSketchControlNet

Download the pretrained PartialSketchControlNet from [here](https://drive.google.com/file/d/1CiqGXn9UOhLS9N_Lu_MtGzzkUa92pZu5/view?usp=sharing) and place it in the `Block-and-Detail/partialsketchcontrolnet` folder.

# Usage - Interface

To run the interface, simply run the following commands (note that number of examples generated may be limited due to GPU memory constraints):

```
cd Block-and-Detail/interface
CUDA_VISIBLE_DEVICES=<available_gpu_ids> python app.py --port=<port_id>
```

![alt text](https://github.com/BlockDetail/Block-and-Detail/blob/main/images/interface.png?raw=true)

# Usage - Gradio Demo

We also offer a gradio demo, but this demo is limited in functionality compared to our interface.  
To run the gradio demo, run the following commands:

```
cd Block-and-Detail
python run_gradio.py
```

![alt text](https://github.com/BlockDetail/Block-and-Detail/blob/main/images/gradio.png?raw=true)

# Data Generation
To generate training data for partial sketch ControlNet, we need foreground object masks and svg files of the extracted synthetic "sketches" of the target images.
We used scripts from [CLIPascene]{https://github.com/yael-vinker/SceneSketch} for foreground object mask extraction and [VirtualSketching]{https://github.com/MarkMoHR/virtual_sketching} for svg file generation.
Then run our data generation script for training partial sketch generation. 
```
python make_data.py --mask_path=MASK_PATH --svg_path=SVG_PATH
```

# Acknowledgements

This repository is partially based on [Diffusers](https://github.com/huggingface/diffusers) and [Collage Diffusion](https://github.com/VSAnimator/collage-diffusion).