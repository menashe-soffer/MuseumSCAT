import os
import gc
import json
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from model import get_model_and_processor
from utils import clear_gpu_zombies

from paths_and_constants import *
from train_lora import my_trainer, finalize_dataset_for_training  # Reuse your existing function
from prompt import *

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
clear_gpu_zombies()

# 1. Distort function for training
import random

import pandas as pd
import random
import os


def distort_locality(text):
    text = str(text)
    # 60% chance to mess with casing
    if random.random() < 0.6:
        text = text.lower() if random.random() < 0.5 else text.upper()
    # 20% chance to drop a character
    if random.random() < 0.2 and len(text) > 2:
        chars = list(text)
        chars.pop(random.randint(0, len(chars) - 1))
        text = "".join(chars)
    return text


def build_dataset(train_csv, pred_csv, image_dir):
    df_gt = pd.read_csv(train_csv)
    df_pred = pd.read_csv(pred_csv)

    # Merge
    df = pd.merge(df_gt, df_pred, on="image_file", suffixes=('_gt', '_pred'))

    # (1) Discard MISSING
    df = df[df['verbatimLocality_gt'].str.lower() != 'missing']

    dataset_rows = []

    for _, row in df.iterrows():
        gt = str(row['verbatimLocality_gt'])
        pred = str(row['verbatimLocality_pred'])

        if gt.upper() == 'MISSING':
            continue

        # generate one synthetic example
        dataset_rows.append({
            "image_path": os.path.join(image_dir, row['image_file']),
            "input_loc": distort_locality(gt),
            "target_loc": gt
        })

        # generate one genuine example
        dataset_rows.append({
            "image_path": os.path.join(image_dir, row['image_file']),
            "input_loc": gt,
            "target_loc": gt
        })

    return dataset_rows



# 2. Refinement Data Preparation
def make_refinement_row(row, method="synthetic"):
    gt_loc = str(row["verbatimLocality"])

    # Input is either the model's bad output OR a distorted GT
    if method == "synthetic":
        input_loc = distort_locality(gt_loc)
    else:
        input_loc = str(row.get("pred_locality", "MISSING"))

    return get_spell_fix_messages(os.path.join(dest_dir, row["image_file"]), input_loc, gt_loc)
    # return {
    #     "messages": [
    #         {"role": "user", "content": [
    #             {"type": "image", "image": os.path.join(dest_dir, row["image_file"])},
    #             {"type": "text", "text": f"Input Locality: {input_loc}\n" +
    #                                      "Instructions: Correct the spelling of the locality based on the image. " +
    #                                      "Maintain capitalization and special characters exactly."}
    #         ]},
    #         {"role": "assistant", "content": [
    #             {"type": "text", "text": gt_loc}
    #         ]},
    #     ]
    # }


# --- Initialization Flow ---
model, processor = get_model_and_processor(light_quant=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=8,  # Lower rank for refinement behavior
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# --- Wrapping to initialize Trainer (Optional, keeps your pipeline consistent) ---
# We use SFTTrainer to ensure the model internal buffers are set for Causal LM
dummy_dataset = Dataset.from_list([
    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
])
dummy_trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=dummy_dataset,  # Dummy
    args=SFTConfig(output_dir="./temp_init")  # Ensure this is set to avoid validation errors)
)
del dummy_trainer
gc.collect()

# --- Dataset Generation ---
train_df = pd.read_csv(train_file)

raw_dataset = build_dataset(train_file, pred_csv="/home/soffer/kaggle/MuseumSCAT/working/submission_post_train.csv", image_dir=images_dir)

# Generate dataset mixing synthetic distortion and your real model predictions
data = []
for _, row in train_df.iterrows():
    data.append(make_refinement_row(row, method="synthetic"))
    # Optionally: if row['pred_locality'] exists, add it too!

hf_dataset = Dataset.from_list(data)
processed_dataset = finalize_dataset_for_training(hf_dataset, processor, min_pixels=256 * 28 * 28,
                                                  max_pixels=256 * 28 * 28)

# --- Custom Training Loop ---

my_trainer(model, processed_dataset, save_path="/home/soffer/kaggle/MuseumSCAT/working/refinement_lora", device='cuda')





# optional code to focus loss on completion text
#
# def make_refinement_row(row, processor, method="synthetic"):
#     # ... (your existing logic for input_loc and gt_loc) ...
#
#     # 1. Create the message structure
#     messages = get_messages(input_loc, gt_loc)
#
#     # 2. Tokenize the full conversation
#     # Use the processor's chat template to get the full formatted text
#     formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
#
#     # 3. MANUALLY MASK THE PROMPT
#     # We create a labels tensor that is a copy of input_ids
#     labels = inputs["input_ids"].clone()
#
#     # Find the boundary of the assistant's turn
#     # This varies by model, but for Qwen it's usually after the assistant token
#     assistant_token_id = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[-1]
#
#     # Find index of assistant token
#     # (Simplified: find where the assistant turn starts)
#     # Everything before the assistant response gets masked with -100
#     mask_idx = (labels == assistant_token_id).nonzero()[0][0]
#     labels[:, :mask_idx + 1] = -100
#
#     return {
#         "input_ids": inputs["input_ids"][0],
#         "attention_mask": inputs["attention_mask"][0],
#         "pixel_values": inputs["pixel_values"][0],
#         "labels": labels[0]
#     }


