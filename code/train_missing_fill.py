import os
import gc
import json
import torch
import pandas as pd
import numpy as np
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



USE_CROP = True
ATOMIZE = False


def atomize(text):
    # Split into words, then atomize each word, then join with triple spaces
    words = text.split(' ')
    atomized_words = [" ".join(list(word)) for word in words]
    return "   ".join(atomized_words)


def unatomize(atomized_text):

    words = atomized_text.split("   ")
    clean_words = [word.replace(" ", "") for word in words]
    return " ".join(clean_words)




def build_dataset(train_csv, pred_csv, image_dir):

    df_gt = pd.read_csv(train_csv)
    df_pred = pd.read_csv(pred_csv)

    # Merge
    df = pd.merge(df_gt, df_pred, on="image_file", suffixes=('_gt', '_pred'))

    dataset_rows = []

    for _, row in df.iterrows():
        gt_locality = str(row['verbatimLocality_gt'])
        pred_locality = str(row['verbatimLocality_pred'])
        gt_date = str(row['verbatimDate_gt'])
        pred_date = str(row['verbatimDate_pred'])

        if (pred_locality.upper() != 'MISSING') or (pred_date.upper() != 'MISSING'):
            continue


        # generate one genuine example
        dataset_rows.append({
            "image_path": os.path.join(image_dir, row['image_file']),
            "target_loc": gt_locality,
            "target_date": gt_date
        })

    return dataset_rows



# 2. Refinement Data Preparation
def make_refinement_row(row, method="synthetic"):

    gt_loc = str(row["target_loc"])
    pred_loc = row["input_loc"]

    # Input is either the model's bad output OR a distorted GT
    if method == "synthetic":
        input_loc = distort_locality(gt_loc)
    else:
        input_loc = pred_loc

    return get_spell_fix_messages(os.path.join(dest_dir, row["image_path"]), input_loc, gt_loc)


if __name__ == "__main__":

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
    #

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

    use_images_dir = images_dir.replace('images', 'for_specimen') if USE_CROP else images_dir

    raw_dataset = build_dataset(train_file, pred_csv="/home/soffer/kaggle/MuseumSCAT/working/submission_post_train.csv",
                                image_dir=use_images_dir)

    # Generate dataset mixing synthetic distortion and your real model predictions
    data = []
    for row in raw_dataset:
        data.append(
            get_missing_fill_messages(img_path=row['image_path'], gt_loc=row['target_loc'], gt_date=row['target_date']))

    hf_dataset = Dataset.from_list(data)
    processed_dataset = finalize_dataset_for_training(hf_dataset, processor, min_pixels=256 * 28 * 28,
                                                      max_pixels=256 * 28 * 28)


    save_path = "/home/soffer/kaggle/MuseumSCAT/working/fill_missing_lora"
    model, optimizer = my_trainer(model, processed_dataset, save_path=save_path,
                               device='cuda', num_epochs=10, optimizer=None)





