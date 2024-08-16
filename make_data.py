import cv2
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
from xml.dom import minidom
from svgpathtools import wsvg, svg2paths2
from cairosvg import svg2png
import random

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--num_batch", type=int, default=1)
parser.add_argument("--svg_path", type=str)
parser.add_argument("--mask_path", type=str)
parser.add_argument("--save_path", type=str, default="./sketch")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

paths = glob.glob(os.path.join(args.svg_path, "*.svg"))
paths.sort()

batch = len(paths)//args.num_batch

for path in tqdm(paths[batch*(args.split):batch*(args.split+1)]):
    name = path.split("/")[-1].split(".")[-2]
    
    doc = minidom.parse(path)
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()

    lines = []

    paths, attributes, svg_attributes = svg2paths2(path)

    mask = os.path.join(args.mask_path, f"{name}_mask.png")
    mask = np.array(Image.open(mask).convert("L").resize((256, 256), resample=0))
    contour = np.zeros_like(mask)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(contour, [c], -1, (255, 255, 255), thickness=1)
    x, y = np.where(contour == 255)

    all_lines = []
    all_dists = []
    for i in range(len(paths)):
        attributes[i]["stroke"] = "#FFFFFF"
        attributes[i]["stroke-width"] = 5.0
        wsvg(paths[i:i+1], attributes=attributes[i:i+1], svg_attributes=svg_attributes, filename=f"check_{args.split}.svg")
        svg_code = open(f"check_{args.split}.svg", 'rt').read()
        svg2png(bytestring=svg_code, write_to=f"check_{args.split}.png")
        sketch = Image.fromarray(np.array(Image.open(f"check_{args.split}.png").convert("L").resize((256, 256))))
        sketch = 255 - np.array(sketch)
        sketch = cv2.medianBlur(sketch, 3)
        sketch = cv2.dilate(sketch, np.ones((4, 4), np.uint8), iterations=1)
        sketch = (sketch + cv2.GaussianBlur(sketch, (3, 3), 10))/255.
        sketch[sketch > 0.9] = 1.0
        if np.sum(1.0 - sketch) > 50:
            x0, y0 = np.where(sketch < 0.5)
            try:
                distx, disty = np.abs(x.reshape(-1, 1) - x0.reshape(1, -1)), np.abs(y.reshape(-1, 1) - y0.reshape(1, -1))
                dist = np.mean(np.min(distx ** 2 + disty ** 2, axis=0))
            except:
                continue
            if np.sum(mask/255. * (1.-sketch)) == 0:
                continue
            all_lines.append((1 - sketch) * 255.)
            all_dists.append(dist)
    idx = np.argsort(all_dists)
    all_lines = np.array(all_lines)[list(idx)]

    if len(all_lines) == 0:
        continue

    length = random.randint(1, max(2, len(all_lines)//2))
    Image.fromarray(255-np.sum(all_lines[:length], axis=0)).convert("RGB").resize((512, 512)).save(f"check_{args.split}.png")
    img = cv2.imread(f"check_{args.split}.png", 0)
    img = cv2.dilate(img, np.ones((1, 1), np.uint8), iterations=1) 
    img = np.expand_dims(cv2.resize(img, (256, 256)), axis=-1).repeat(3, axis=-1)
    cv2.imwrite(os.path.join(args.save_path, f"{name}_0.png"), img)