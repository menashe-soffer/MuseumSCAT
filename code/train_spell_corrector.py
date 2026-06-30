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

from text_manipulations import *


USE_CROP = True
ATOMIZE = True




def distort_locality(text):

    text = str(text)

    SIMILARITY_MAP = {
        'h': ['b', 'n', 'g'],
        'b': ['h', 'v', 'p'],
        'a': ['o', 'e', 'å', 'ä'],
        'o': ['a', '0', 'u', 'ø', 'ö'],
        'æ': ['a', 'e', 'æ'],
        'ø': ['o', 'u', 'ø'],
        # ... add more as needed
    }
    RAMDON_CHARS = 'abcdefghijklmnopqratuvwxyzåäæöø'
    RAMDON_CHARS = RAMDON_CHARS + RAMDON_CHARS.upper()
    RAMDON_CHARS = list(RAMDON_CHARS)

    words = text.split()
    for i_word, word in enumerate(words):

        chars = list(word)
        # lower case first letter
        if random.random() < 0.7:
            chars[0] = chars[0].lower()
        # next letters
        for i in range(1, len(chars)):
            if random.random() < 0.05:
                chars[i] = chars[i].lower()
            if random.random() < 0.05:
                chars[i] = chars[i].upper()
        # drop letters
        drop_mask = np.array([random.random() for i_char in range(len(chars))]) < 0.1
        chars = [chars[i] for i in np.argwhere(np.logical_not(drop_mask)).flatten().astype(int)]
        # letter replacement
        for i_char, char in enumerate(chars):
            # replacement of similar letters
            if (char.lower() in SIMILARITY_MAP) and (random.random() < 0.15):
                candidates = SIMILARITY_MAP[char.lower()]
                new_char = random.choice(candidates)
                chars[i_char] = new_char.upper() if char.isupper() else new_char
            else:
                # completely randon replacemenr
                if random.random() < 0.01:
                    chars[i_char] = random.choice(RAMDON_CHARS)
        # add letter
        if (random.random() < 0.05) and (len(chars) > 3):
            place = random.randint(0, len(chars) - 1)
            new_char = random.choice(RAMDON_CHARS)
            chars = chars[:place] + [new_char] + chars[place:]

        words[i_word] = "".join(chars)

    text = " ".join(words)
    # add Dania
    if random.random() < 0.05:
        fake = [random.choice(RAMDON_CHARS) for i in range(random.randint(2, 6))]
        text = "Dania: " + "".join(fake) + " " + text


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

        if (gt.upper() == 'MISSING') or (pred.upper() == 'MISSING'):
            continue

        # generate one synthetic example
        if random.random() < 0.8:
            dataset_rows.append({
                "image_path": os.path.join(image_dir, row['image_file']),
                "input_loc": atomize(distort_locality(gt)),
                "target_loc": atomize(gt)
            })

        # generate one genuine example
        if random.random() < 0.9:
            for i in range(1):
                dataset_rows.append({
                    "image_path": os.path.join(image_dir, row['image_file']),
                    "input_loc": atomize(pred),
                    "target_loc": atomize(gt)
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

    use_images_dir = images_dir.replace('images', 'for_card') if USE_CROP else images_dir

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

    optimizer = None
    for major_epoch in range(10):
        print("Major epoch {}".format(major_epoch))
        raw_dataset = build_dataset(train_file, pred_csv="/home/soffer/kaggle/MuseumSCAT/working/submission_post_train.csv", image_dir=use_images_dir)

        # Generate dataset mixing synthetic distortion and your real model predictions
        data = []
        for row in raw_dataset:
            data.append(get_spell_fix_messages(img_path=row['image_path'], input_loc=row['input_loc'], gt_loc=row['target_loc']))

        hf_dataset = Dataset.from_list(data)
        processed_dataset = finalize_dataset_for_training(hf_dataset, processor, min_pixels=256 * 28 * 28,
                                                          max_pixels=256 * 28 * 28)

        # --- Custom Training Loop ---

        save_path = "/home/soffer/kaggle/MuseumSCAT/working/refinement_lora"
        model, optimizer = my_trainer(model, processed_dataset, save_path=save_path,
                               device='cuda', num_epochs=1, optimizer=optimizer)
        model.save_pretrained(f"{save_path}_{major_epoch + 1}")




