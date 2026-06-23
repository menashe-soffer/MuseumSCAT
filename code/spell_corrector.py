import pandas as pd
import os
import torch
from model import get_model_and_processor
from peft import PeftModel
from PIL import Image
import tqdm

from paths_and_constants import *
from prompt import *


def run_spell_corrector(input_csv, model_path):
    # 1. Filename logic
    if 'post' not in input_csv:
        raise ValueError("Input filename must contain 'post'")
    output_csv = input_csv.replace('post', 'spelling')

    # 2. Load Model & Refined LoRA
    # Using your existing get_model_and_processor function
    model, processor = get_model_and_processor(light_quant=True)
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    df = pd.read_csv(input_csv)
    results = []

    print(f"Starting correction. Processing {len(df)} rows...")

    for _, row in tqdm.tqdm(df.iterrows()):
        # Prepare Prompt
        img_path = os.path.join("/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/images", row['image_file'])
        input_loc = str(row['verbatimLocality'])

        if input_loc.upper() == 'MISSING':
            results.append(row)
            continue

        # Consistent with training prompt
        prompt_text = get_spell_fix_messages(img_path=img_path, input_loc=input_loc)
        prompt_text = processor.apply_chat_template(
            prompt_text["messages"],
            tokenize=False,
            add_generation_prompt=True
        )

        # Inference
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50)
            # Slice to get only the generated part
            generated_text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:],
                                                    skip_special_tokens=True)[0]

        row['verbatimLocality'] = generated_text.strip()
        results.append(row)

    # 3. Save
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Done! Saved to {output_csv}")


if __name__ == "__main__":
    # Point these to your paths
    run_spell_corrector(
        input_csv="/home/soffer/kaggle/MuseumSCAT/working/submission_post_test.csv",
        model_path="/home/soffer/kaggle/MuseumSCAT/working/refinement_lora"
    )