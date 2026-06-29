import pandas as pd
import os
import torch
from model import get_model_and_processor
from peft import PeftModel
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import json

from paths_and_constants import *
from prompt import *
from text_manipulations import *


USE_CROP = True
ATOMIZE = False

def run_fill_missing(input_csv, model_path, show=False):
    # 1. Filename logic
    if 'pre' not in input_csv:
        raise ValueError("Input filename must contain 'post'")
    output_csv = input_csv.replace('pre', 'filled')

    # 2. Load Model & Refined LoRA
    # Using your existing get_model_and_processor function
    model, processor = get_model_and_processor(light_quant=True)
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    df = pd.read_csv(input_csv)
    results = []

    print(f"Starting correction. Processing {len(df)} rows...")

    use_images_dir = images_dir.replace('images', 'for_specimen') if USE_CROP else images_dir

    for _, row in tqdm.tqdm(df.iterrows()):
        # Prepare Prompt
        img_path = os.path.join(use_images_dir, row['image_file'])
        input_loc = str(row['verbatimLocality'])
        input_date = str(row['verbatimDate'])

        if (input_loc.upper() != 'MISSING') or (input_date.upper() != 'MISSING'):
            results.append(row)
            continue

        # Consistent with training prompt
        prompt_text = get_missing_fill_messages(img_path=img_path)
        prompt_text = processor.apply_chat_template(
            prompt_text["messages"],
            tokenize=False,
            add_generation_prompt=True
        )

        # Inference
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50*2)
            # Slice to get only the generated part
            generated_text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:],
                                                    skip_special_tokens=True)[0]
            #generated_text = unatomize(generated_text)
        generated_dict = json.loads(generated_text)
        for key in generated_dict:
            row[key] = generated_dict[key]

        results.append(row)

        # display
        if show and (generated_text.find('MISSING') == -1):
            plt.figure(figsize=(12, 10))
            plt.imshow(image)
            plt.title('{}, {}\n{}'.format(input_date, input_loc,
                                          generated_text.replace('"verbatimDate_confidence": 0.95', " ").replace('"verbatimLocality_confidence": 0.95', " ")))
            plt.tight_layout()
            plt.show()

    # 3. Save
    preds_df = pd.DataFrame(results)
    preds_df["verbatimDate"] = preds_df["verbatimDate"].apply(postprocess)
    preds_df["verbatimLocality"] = preds_df["verbatimLocality"].apply(postprocess)
    output_fname = input_csv.replace('pre', 'filled')
    preds_df.to_csv(output_fname, index=False)

    print(f"Done! Saved to {output_fname}")


if __name__ == "__main__":
    # Point these to your paths
    run_fill_missing(
        input_csv="/home/soffer/kaggle/MuseumSCAT/working/submission_pre_test.csv",
        model_path="/home/soffer/kaggle/MuseumSCAT/working/fill_missing_lora_10",
        show=False
    )