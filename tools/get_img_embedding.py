import torch
import clip
from PIL import Image
import json
import numpy as np
import os
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img_root = ""  # path to the folder containing images
save_root = ""  # path to the folder to save the features

for img in os.listdir(img_root):
    if "color" not in img:
        continue

    img_path = os.path.join(img_root, img)
    im = Image.open(img_path)
    arr = np.array(im)

    tmp_result = {}
    feat_save_path = img.replace("_color.jpg", "")
    feat_save_path = os.path.join(save_root, feat_save_path + ".pkl")

    # stride is 8 and patch size is 16 here
    imgs = []
    for row in range(1, 60):
        for col in range(1, 80):
            row_pos = row * 8
            col_pos = col * 8
            arr_patch = arr[row_pos-8:row_pos+8, col_pos-8:col_pos+8, :]
            pil_img = Image.fromarray(arr_patch)
            imgs.append(preprocess(pil_img).unsqueeze(0))

    clip_input = torch.cat(imgs).cuda()
    with torch.no_grad():
        image_features = model.encode_image(clip_input)

    tmp_result[f"patch_clip_embedding"] = image_features.detach().cpu().numpy()

    with open(feat_save_path, "wb") as handle:
        pickle.dump(tmp_result, handle)
        print("wrote to", feat_save_path)
