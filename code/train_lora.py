# =====================================================================
# STEP 1: CRITICAL ENVIRONMENT CONFIGURATION (Must run before ALL else)
# =====================================================================
import os
import tqdm
import sys
import torch
from PIL import Image

import bitsandbytes as bnb
from utils import clear_gpu_zombies
# Force the system to recognize the GPU before any library initializes
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Prevent multi-threading forks from breaking CUDA handshakes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

clear_gpu_zombies()
# try:
#     # Force device driver context wakeup
#     _ = torch.tensor([1.0]).cuda()
#     print(f"✅ CUDA handshaked successfully on: {torch.cuda.get_device_name(0)}")
#
#     # Print current overhead
#     free_mem, total_mem = torch.cuda.mem_get_info()
#     print(f"📊 Available VRAM: {free_mem / 1024 ** 3:.2f} GB / {total_mem / 1024 ** 3:.2f} GB")
#
#     if (free_mem / total_mem) < 0.75:
#         print("⚠️ WARNING: Less than 30% of your VRAM is free! A leaked process is likely still active.")
#         print("👉 Please run 'Kernel -> Restart Kernel' or check 'nvidia-smi' before training.")
#
# except Exception as e:
#     print(f"❌ CUDA is visible but failed to initialize! Error: {e}")
#     print("👉 Run 'kill -9 <PID>' on the process shown in nvidia-smi.")
#     assert False, "CUDA is not available to PyTorch!"

# =====================================================================
# STEP 2: STANDARD PYTHON UTILITIES (Safe to load now)
# =====================================================================
import gc
import json
import pandas as pd
import shutil

# =====================================================================
# STEP 3: CORE DEEP LEARNING FRAMEWORKS (Loads with clean environment)
# =====================================================================
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =====================================================================
# STEP 4: YOUR CUSTOM PROJECTS MODULAR IMPORTS
# =====================================================================
from paths_and_constants import *
from model import get_model_and_processor#, wrap_with_peft
from prompt import propmt

from transformers import Qwen2_5_VLProcessor
import torch

from transformers import Qwen2_5_VLProcessor
import torch


def finalize_dataset_for_training(partially_processed_dataset, processor, min_pixels, max_pixels):
    """
    Takes the partially processed dataset (containing 'messages'),
    runs it through the Qwen vision processor using a safe manual text layout,
    and applies a custom -100 masking array to the labels.
    """
    print('finalize_dataset_for_training')


    def process_and_mask_element(example):
        messages = example["messages"]
        image_path = messages[0]["content"][0]["image"]

        # 1. Extract the raw string values directly from your messages list
        user_prompt = messages[0]["content"][1]["text"]
        assistant_answer = messages[1]["content"][0]["text"]

        # 2. Manually construct Qwen's exact chat sequence layout.
        # This explicitly guarantees EXACTLY ONE <|image_pad|> token exists in the text block!
        custom_text_sequence = (
            "<|im_start|>user\n"
            "<|image_pad|>"  # Exactly one image token placement
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{assistant_answer}<|im_end|>\n"
        )

        # 3. Stream through the vision processor safely
        batch_processed = processor(
            text=custom_text_sequence,
            images=image_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding=False,
            return_tensors="pt"
        )

        # Pull arrays out of batch wrapping safely
        input_ids = batch_processed["input_ids"][0].tolist()
        attention_mask = batch_processed["attention_mask"][0].tolist()
        pixel_values = batch_processed["pixel_values"]
        image_grid_thw = batch_processed["image_grid_thw"][0].tolist()

        # 4. Apply custom masking
        labels = list(input_ids)

        # Get the tokenized representation of our exact assistant marker
        assistant_header_tokens = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        match_idx = None
        for i in range(len(input_ids) - len(assistant_header_tokens) + 1):
            if input_ids[i:i + len(assistant_header_tokens)] == assistant_header_tokens:
                match_idx = i + len(assistant_header_tokens)
                break

        if match_idx is not None:
            # Mask everything before the assistant's answer out with -100
            for idx in range(match_idx):
                labels[idx] = -100
        else:
            print(f"Warning: Boundary marker not found for image {image_path}!")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels
        }

    # Map across the entire dataset and drop the unneeded original "messages" column
    final_mapped_dataset = partially_processed_dataset.map(
        process_and_mask_element,
        remove_columns=partially_processed_dataset.column_names
    )
    return final_mapped_dataset




def my_trainer(model, processed_hf_dataset, save_path, device):

    # Ensure LoRA parameters are explicitly marked as trainable
    model.train()
    optimizer = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    # 2. Your Standard PyTorch DataLoader
    # Inside your custom collate function, ensure tensors are initialized on CPU first:
    def qwen_collate_fn(batch):

        try:
            # this works for the "manual" preperation of the dataset
            input_ids = torch.Tensor([item["input_ids"] for item in batch]).unsqueeze(0).to(torch.long)
            attention_mask = torch.Tensor([item["attention_mask"] for item in batch]).unsqueeze(0).to(torch.long)
            labels = torch.Tensor([item["labels"] for item in batch]).unsqueeze(0).to(torch.long)
            image_grid_thw = torch.Tensor([item["image_grid_thw"] for item in batch]).to(torch.long)
            # For pixel_values, since it's a cat operation:
            pixel_values = torch.Tensor([item["pixel_values"] for item in batch]).to(torch.bfloat16)
        except:
            # Using a list comprehension with .clone().detach()
            input_ids = torch.stack([item["input_ids"].clone().detach() for item in batch])
            attention_mask = torch.stack([item["attention_mask"].clone().detach() for item in batch])
            labels = torch.stack([item["labels"].clone().detach() for item in batch])
            image_grid_thw = torch.stack([item["image_grid_thw"].clone().detach() for item in batch])
            # For pixel_values, since it's a cat operation:
            pixel_values = torch.cat([item["pixel_values"].clone().detach().to(torch.bfloat16) for item in batch], dim=0)

        # print('input_ids:     \t', input_ids.shape)
        # print('attention_mask:\t', attention_mask.shape)
        # print('labels:        \t', labels.shape)
        # print('image_grid_thw:\t', image_grid_thw.shape)
        # print('pixel_values:  \t', pixel_values.shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }
        # Assumes your dataset outputs pre-processed 'input_ids', 'labels', and 'pixel_values'
    train_loader = torch.utils.data.DataLoader(processed_hf_dataset, batch_size=1, shuffle=True, collate_fn=qwen_collate_fn, pin_memory=True)

    num_epochs = 5
    gradient_accumulation_steps = 20
    # model.gradient_checkpointing_enable()
    # model.config.gradient_checkpointing = True

    print("Starting Custom Training Loop...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        # Progress bar for visibility
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Move inputs to your laptop's GPU
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            labels = batch["labels"].squeeze(1).to(device)

            # Qwen2.5-VL specific vision features (adjust keys based on your processor output)
            pixel_values = batch.get("pixel_values", None).squeeze(1)
            image_grid_thw = batch.get("image_grid_thw", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
                image_grid_thw = image_grid_thw.view(-1, 3).to(device)

            #model.config.use_cache = False
            model.loss_type = None
            # Mixed precision context manager for safety and VRAM efficiency
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,  # i will calculate loss "manually"
                    #return_dict=False,
                    use_cache=False
                )

                # # Extract logits and shift them IN-PLACE without duplicating memory
                # shift_logits = outputs.logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()
                #
                # # Filter out padding elements immediately
                # loss_mask = shift_labels != -100
                # active_logits = shift_logits[loss_mask]
                # active_labels = shift_labels[loss_mask]
                #
                # # # Clean up the massive raw output structure from VRAM immediately!
                # # del outputs
                # # import gc
                # # gc.collect()
                #
                # # Flatten the tokens for CrossEntropyLoss
                # # We stay strictly in bfloat16 to save that 880 MiB of VRAM!
                # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                #
                # raw_loss = loss_fct(
                #     shift_logits.view(-1, shift_logits.size(-1)),
                #     shift_labels.view(-1)
                # )
                #
                # # Scale loss for gradient accumulation
                # if active_labels.numel() > 0:
                #     raw_loss = loss_fct(active_logits, active_labels)
                # else:
                #     raw_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Backward pass
            raw_loss = outputs.loss
            raw_loss.backward()
            epoch_loss += raw_loss.item() * gradient_accumulation_steps

            # Optimizer Step (Accumulation execution)
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Optional: Add gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            disp_loss = raw_loss.detach().cpu().numpy()
            del input_ids, attention_mask, labels, pixel_values, image_grid_thw, outputs, raw_loss
            gc.collect()
            torch.cuda.empty_cache()  # Forces PyTorch to release unused memory memory pools back to your system GPU

            # Update progress bar view dynamically
            progress_bar.set_postfix({"loss": f"{disp_loss.item() * gradient_accumulation_steps:.4f}"})

        print(f"Epoch {epoch + 1} Complete. Average Loss: {epoch_loss / len(train_loader):.4f}")

        # Save a native model checkpoint manually per epoch
        model.save_pretrained(f"{save_path}_{epoch + 1}")



if __name__ == '__main__':

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # 1. Load the raw base architecture
    model, processor = get_model_and_processor(light_quant=True)

    # 2. Extract the TRUE inner model before PEFT wrappers hide it
    # For Qwen2.5-VL, this is the Qwen2_5_VLForConditionalGeneration instance
    raw_base_model = model

    # 3. Mandatorily prepare k-bit training hooks on the raw model
    print("Preparing quantized base model parameters...")
    raw_base_model = prepare_model_for_kbit_training(raw_base_model, use_gradient_checkpointing=True)

    # 4. Define and apply your exact LoRA configurations
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("Applying PEFT LoRA wrappers...")
    model = get_peft_model(raw_base_model, peft_config)

    # 5. REVERSE-ENGINEERED: Drill directly to the true core layers to enable checkpointing
    print("Enabling explicit gradient checkpointing on text layers...")
    model.gradient_checkpointing_enable()

    # 6. FORCE the true vision transformer module to stop caching forward patches
    model.model.model.gradient_checkpointing_enable()
    model.model.model.language_model.gradient_checkpointing_enable()
    model.model.model.visual.gradient_checkpointing_enable()

    print("Scanning model for trainable parameters (requires_grad=True)...\n")

    trainable_params_found, none_lora = 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            #print(f"🔥 TRAINABLE [{param.dtype}] -> {name} | Shape: {list(param.shape)}")
            trainable_params_found += 1
            if name.find('.lora_') == -1:
                none_lora += 1
    print(trainable_params_found, 'layers require grads', none_lora, 'not from lora')


    def make_row(row):
        img_path  = os.path.join(dest_dir, row["image_file"])
        date_val  = str(row["verbatimDate"])
        loc_val   = str(row["verbatimLocality"])
        # Training confidences are all 1.0 (ground truth artifact) — use realistic values instead
        date_conf = 1.0 if date_val == "MISSING" else 0.95
        loc_conf  = 1.0 if loc_val  == "MISSING" else 0.95
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text",  "text": propmt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": json.dumps({
                        "verbatimDate":                date_val,
                        "verbatimDate_confidence":     date_conf,
                        "verbatimLocality":            loc_val,
                        "verbatimLocality_confidence": loc_conf,
                    })}
                ]},
            ]
        }


    src = train_df.head(20) if SMOKE else train_df
    hf_dataset = Dataset.from_list([make_row(r) for _, r in src.iterrows()])
    processed_hf_dataset = finalize_dataset_for_training(partially_processed_dataset=hf_dataset, processor=processor, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)

    from model_parsing import custom_model_parser, ExecutionFlowWrapper
    #custom_model_parser(model)
    #ExecutionFlowWrapper(processor, model, hf_dataset[0])

    print(f"Training on {len(hf_dataset)} samples (smoke={SMOKE})")
    # Clean memory before launching trainer
    gc.collect()
    torch.cuda.empty_cache()

    #my_trainer(model, processed_hf_dataset=processed_hf_dataset, save_path="/home/soffer/kaggle/MuseumSCAT/working/lora_adapter", device='cuda')
    # Launch directly into the configured trainer (no placeholder double instantiations)
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=hf_dataset,
        args=SFTConfig(
            output_dir="/home/soffer/kaggle/MuseumSCAT/checkpoints",
            report_to="tensorboard",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=20,
            optim="adamw_8bit",
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=3e-4,#2e-4,
            warmup_steps=10,#100,
            lr_scheduler_type="cosine",

            # Enforce your hardware precision cleanly
            fp16=False,
            bf16=True,

            save_strategy="no",
            remove_unused_columns=False,

            # # ─── ADDED MODERN NATIVE MASKING STRATEGY ───
            # completion_only_loss=True,                  # Activate -100 parallel matrix masking
            # packing=False,                              # Required to keep masking calculations intact
            # # ────────────────────────────────────────────
            completion_only_loss=False,  # TURN THIS OFF! Stop relying on the broken mask tracker
            packing=False,               # Keep this False

            # ─── REMOVED dataset_text_field="text" HERE ───
            dataset_kwargs={"skip_prepare_dataset": False,
                            "response_template": "<|im_start|>assistant\n"}, # Passed safely here!

            ddp_find_unused_parameters=False,
            dataloader_num_workers=0,
        ),
    )


    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    my_trainer(model, processed_hf_dataset=processed_hf_dataset, save_path="/home/soffer/kaggle/MuseumSCAT/working/lora_adapter", device='cuda')

    # if TRAIN_EPOCHS > 0:
    #     print("Starting training loop...")
    #     trainer.train()
    #     model.save_pretrained("/home/soffer/kaggle/MuseumSCAT/working/lora_adapter")
    #     print("Training Complete!")



