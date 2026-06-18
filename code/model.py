import os
import json
import shutil
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from torchview import draw_graph
from peft import LoraConfig, get_peft_model
from paths_and_constants import *

def fix_patched_model_folder():
    os.makedirs(writable_model_path, exist_ok=True)
    for f in os.listdir(model_path_3b):
        src = os.path.realpath(os.path.join(model_path_3b, f))
        dst = os.path.join(writable_model_path, f)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if f.endswith((".json", ".txt", ".py", ".md")):
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)

    cfg_path = os.path.join(writable_model_path, "preprocessor_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    cfg["image_processor_type"] = "Qwen2VLImageProcessor"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

def get_model_and_processor(light_quant=False, checkpoint_path=None):
    # Always make sure the patched folder is ready
    fix_patched_model_folder()

    min_pixels = 256 * 28 * 28
    max_pixels = 256 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        writable_model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Use AutoModelForImageTextToText uniformly for both training and inference configurations
    if light_quant:
        model = AutoModelForImageTextToText.from_pretrained(
            writable_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            writable_model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )

    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Valid checkpoint_path must be provided. Got: {checkpoint_path}")
        print(f"Attaching LoRA adapter weights from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)

    return model, processor



# def wrap_with_peft(model):
#
#     peft_config = LoraConfig(
#         r=8,
#         lora_alpha=8,
#         target_modules=[
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ],
#         lora_dropout=0,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
#
#     print("Applying LoRA adapters...")
#     return get_peft_model(model, peft_config)



if __name__ == "__main__":


    # Pulling your active model function
    #from model import get_model_and_processor

    print("Booting up the model for structural mapping...")
    # light_quant=False gives the mapping engine clean access to full layers
    model, processor = get_model_and_processor(light_quant=False)
    #model.cpu()

    print("Tracing the live data streams...")
    model_device = next(model.parameters()).device
    # We use a micro-batch text token tensor to spark the forward pass blueprint
    mock_input_ids = torch.randint(0, 1000, (1, 16)).to(model_device)


    # Compile the high-fidelity flowchart
    model_graph = draw_graph(
        model,
        input_data={"input_ids": mock_input_ids},
        expand_nested=False,  # Set to True if you want a massive chart showing every single attention head
        depth=3,  # Depth 3 perfectly isolates the main blocks (Vision, Projector, Transformer Layers)
        device=model_device,
        save_graph=True,
        filename="qwen2_5_vl_true_layout",
        directory="/home/soffer/kaggle/MuseumSCAT/"
    )

    print("🎉 Success! Check your folder for 'qwen2_5_vl_true_layout.png'!")