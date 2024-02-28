# Block and Detail: Scaffolding Sketch-to-Image Generation

- [Online Demo]()
- [Paper]()

We introduce a novel sketch-to-image tool that aligns with the iterative refinement process of artists. 
Our tool lets users sketch **blocking** strokes to coarsely represent the placement and form of objects and **detail** strokes to refine their shape and silhouettes. 

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

# Usage - Gradio Demo

To run the gradio demo with limited functionality, run the following commands:

```
cd Block-and-Detail
python run_gradio.py
```


# Citation

```
@misc{}
```

# Acknowledgements

This repository is partially based on [Diffusers](https://github.com/huggingface/diffusers) and [Collage Diffusion](https://github.com/VSAnimator/collage-diffusion).