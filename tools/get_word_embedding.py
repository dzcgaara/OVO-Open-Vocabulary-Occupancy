import torch
import clip
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# NYU
prompt_list = ["empty", "ceiling", "floor", "wall", "window", "chair",
               "bed", "sofa", "table", "tv", "furniture", "other"]

# kitti
prompt_list = ["empty", "car", "bicycle", "motocycle", "truck", "strange-vehicle", "person", "bicyclist", "motocyclist", "road",
               "parking", "sidewalk", "strange-ground", "building", "fence", "vegatation", "trunk", "terrian", "pole", "traffic_sign"]

save_path = ""  # save path

text = clip.tokenize(prompt_list).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

text_features = text_features.cpu().numpy()

text_embedding = {}
for c, i in enumerate(prompt_list):
    text_embedding[i] = text_features[c].tolist()

with open(save_path, "w") as f:
    json.dump(text_embedding, f)
