import numpy as np
import pandas as pd
import os
import sys
import gc
import torch
import json
import re

from qwen_vl_utils import process_vision_info

from paths_and_constants import *
from model import get_model_and_processor#, wrap_with_peft
from prompt import propmt


train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


model, processor = get_model_and_processor(light_quant=True, checkpoint_path='/home/soffer/kaggle/MuseumSCAT/working/lora_adapter')
# ── Raw output ─────────────────────────────────────────────────────────────
# FastVisionModel.for_inference(model)
model.eval()

test_row = test_df.iloc[0]
img_path = os.path.join(dest_dir, test_row["image_file"])
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img_path},
    {"type": "text", "text": propmt},
]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_info, _ = process_vision_info(msgs)
inputs = processor(text=[text], images=img_info, return_tensors="pt").to(model.device)
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

# raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=== RAW OUTPUT ===");
print(raw);
print("==================")

# ── Batch inference ──────────────────────────────────────────────────────────
def clean_json(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()



def run_inference(df, prompt):
    results = []

    # Force evaluation mode
    model.eval()

    print(f"Starting batch-by-batch inference on {len(df)} samples...")

    # Process exactly 1 row at a time to minimize memory footprints
    for idx, row in df.iterrows():
        # if idx > 25:
        #     continue   # for quick debug

        # Clear VRAM cache aggressively between rows
        gc.collect()
        torch.cuda.empty_cache()

        # text = row["text"] # update to match your prompt string column name if different
        img_file = os.path.join(dest_dir, row["image_file"])

        # 1. Load image safely
        from PIL import Image
        try:
            img_info = Image.open(img_file).convert("RGB")
        except Exception:
            img_info = None

        # ─── THE FIX: Format the prompt string for multi-modal inference ───
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_file},
                    {"type": "text", "text": propmt}
                ]
            }
        ]
        # This inserts the critical <|image_pad|> tags Qwen needs to align the features
        formatted_text = processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)  # 2. Process input with an explicit max token pixel boundary

        inputs = processor(
            text=[formatted_text],
            images=[img_info] if img_info else None,
            return_tensors="pt",
        ).to(model.device)

        # Cast vision tensors to half precision
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # 3. Generate with strict constraints
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,  # Tightened to save generation step VRAM
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True  # Speeds up sequential token generation memory reuse
                )

        # 4. Decode text
        prompt_len = inputs["input_ids"].shape[1]
        #decoded_text = processor.decode(out_ids[0][prompt_len:], skip_special_tokens=True)

        for ids in out_ids:
            text_out = processor.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
            text_out = clean_json(text_out)
            try:
                parsed = json.loads(text_out)
                date_val = str(parsed.get("verbatimDate", "missing")).strip().lower()
                loc_val = str(parsed.get("verbatimLocality", "missing")).strip().lower()
            except json.JSONDecodeError:
                # Fallback values if the JSON structure is broken
                date_val = "missing"
                loc_val = "missing"
            date_conf = 1.0 if date_val in ["missing", "nan", ""] else 0.95
            loc_conf = 1.0 if loc_val in ["missing", "nan", ""] else 0.95

            # Construct the clean dictionary matching your evaluation schema
            final_row = {
                "image_file": row["image_file"],
                "verbatimDate": date_val,
                "verbatimDate_confidence": date_conf,
                "verbatimLocality": loc_val,
                "verbatimLocality_confidence": loc_conf
            }

            results.append(final_row)

    # # Append target key format to list (assuming standard sample submission headers)
    #     results.append({
    #         "image_file": row["image_file"],
    #         "prediction": decoded_text.strip()
    #     })


        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows successfully.")

    return pd.DataFrame(results)


# ── Execution Block ─────────────────────────────────────────────────────────
TEST = True
gc.collect()
torch.cuda.empty_cache()
infer_df = test_df if TEST else train_df
infer_df = infer_df.head(5) if SMOKE else infer_df
output_fname = os.path.join("/home/soffer/kaggle/MuseumSCAT/working/", 'submission_pre_{}.csv'.format('test' if TEST else 'train'))
preds_df = run_inference(infer_df, prompt=propmt)
preds_df.to_csv(output_fname, index=False)
print(preds_df)
# Clear out lingering baseline training states first
gc.collect()
torch.cuda.empty_cache()

###

def postprocess(text):
    if not isinstance(text, str) or text.upper() == "MISSING":
        return "MISSING" if text.upper() == "MISSING" else text
    for old, new in {"ö":"ø","Ö":"Ø","ä":"æ","Ä":"Æ","ü":"y","Ü":"Y","ÿ":"y","Ÿ":"Y"}.items():
        text = text.replace(old, new)
    return text

preds_df["verbatimDate"]     = preds_df["verbatimDate"].apply(postprocess)
preds_df["verbatimLocality"] = preds_df["verbatimLocality"].apply(postprocess)
preds_df.to_csv(output_fname.replace('pre', 'post'), index=False)

