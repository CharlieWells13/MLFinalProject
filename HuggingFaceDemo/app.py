#Note that this is a copy of app.py, the actual one that drives the demo can be found in the RepoLink.txt file
import random

import gradio as gr
import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw

from model_pretrained import build_model

CHECKPOINT_PATH = "model/best.pt"
RF_PATH = "model/random_forest_pipeline.joblib"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
dl_model = build_model(pretrained=False, freeze_backbone=False, apply_sigmoid=True).to(device)
dl_model.load_state_dict(state_dict)
dl_model.eval()

rf_model = joblib.load(RF_PATH)

EXAMPLE_PATHS = [
    "examples/Abyssinian_225.jpg",
    "examples/Bengal_66.jpg",
    "examples/Birman_167.jpg",
    "examples/Bombay_213.jpg",
    "examples/British_Shorthair_38.jpg",
    "examples/Egyptian_Mau_167.jpg",
    "examples/Maine_Coon_38.jpg",
    "examples/Persian_221.jpg",
    "examples/Ragdoll_189.jpg",
    "examples/Russian_Blue_253.jpg",
    "examples/Siamese_193.jpg",
    "examples/Sphynx_241.jpg",
    "examples/american_bulldog_196.jpg",
    "examples/american_pit_bull_terrier_66.jpg",
    "examples/basset_hound_112.jpg",
    "examples/beagle_138.jpg",
    "examples/boxer_147.jpg",
    "examples/chihuahua_63.jpg",
    "examples/english_cocker_spaniel_11.jpg",
    "examples/english_setter_170.jpg",
    "examples/german_shorthaired_131.jpg",
    "examples/great_pyrenees_167.jpg",
    "examples/havanese_113.jpg",
    "examples/japanese_chin_12.jpg",
    "examples/keeshond_23.jpg",
    "examples/leonberger_6.jpg",
    "examples/miniature_pinscher_200.jpg",
    "examples/newfoundland_137.jpg",
    "examples/pomeranian_189.jpg",
    "examples/pug_52.jpg",
    "examples/saint_bernard_139.jpg",
    "examples/samoyed_59.jpg",
    "examples/scottish_terrier_132.jpg",
    "examples/shiba_inu_122.jpg",
    "examples/staffordshire_bull_terrier_39.jpg",
    "examples/wheaten_terrier_49.jpg",
    "examples/yorkshire_terrier_20.jpg",
]

BREED_NAMES = [path.split("/")[-1].rsplit("_", 1)[0].replace("_", " ").title() for path in EXAMPLE_PATHS]
BREED_TO_PATH = dict(zip(BREED_NAMES, EXAMPLE_PATHS))


def preprocess_dl(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def preprocess_rf(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)  # 224*224*3 = 150528 features


def predict(image: Image.Image) -> Image.Image:
    orig_w, orig_h = image.size
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    # Deep learning prediction (normalized xywh center coords)
    with torch.no_grad():
        dl_pred = dl_model(preprocess_dl(image))[0].cpu()
    cx, cy, bw, bh = dl_pred.tolist()
    dl_box = [
        max(0, int(round((cx - bw / 2) * orig_w))),
        max(0, int(round((cy - bh / 2) * orig_h))),
        min(orig_w, int(round((cx + bw / 2) * orig_w))),
        min(orig_h, int(round((cy + bh / 2) * orig_h))),
    ]
    draw.rectangle(dl_box, outline="red", width=3)

    # Random forest prediction (pixel xywh top-left coords in 224x224 space)
    rf_pred = rf_model.predict(preprocess_rf(image))[0]
    x, y, w, h = rf_pred
    rf_box = [
        max(0, int(round(x * orig_w / 224))),
        max(0, int(round(y * orig_h / 224))),
        min(orig_w, int(round((x + w) * orig_w / 224))),
        min(orig_h, int(round((y + h) * orig_h / 224))),
    ]
    draw.rectangle(rf_box, outline="blue", width=3)

    return result


def load_from_dropdown(breed: str):
    return Image.open(BREED_TO_PATH[breed])


def load_random():
    breed = random.choice(BREED_NAMES)
    return Image.open(BREED_TO_PATH[breed]), breed


with gr.Blocks(title="ML Final Project Demo") as demo:
    gr.Markdown("# ML Final Project Demo")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")

            gr.Markdown("**Try an example:**")
            with gr.Row():
                dropdown = gr.Dropdown(choices=BREED_NAMES, label="Select a breed", scale=3)
                random_btn = gr.Button("Random", scale=1)

        with gr.Column():
            output_image = gr.Image(label="Output")
            run_btn = gr.Button("Run", variant="primary")
            gr.Markdown("**Legend**\n\n🟥 ResNet-18 (Deep Learning)\n\n🟦 Random Forest (Traditional ML)")

    dropdown.change(load_from_dropdown, inputs=dropdown, outputs=input_image)
    random_btn.click(load_random, outputs=[input_image, dropdown])
    run_btn.click(predict, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    demo.launch()