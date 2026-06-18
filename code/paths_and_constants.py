# import pandas as pd
# #from unsloth import FastVisionModel
# #from unsloth.trainer import UnslothVisionDataCollator
# #from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from transformers import AutoProcessor, AutoModelForImageTextToText#AutoModelForVision2Seq
# from huggingface_hub import snapshot_download
# from qwen_vl_utils import process_vision_info
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from PIL import Image
# from tqdm.auto import tqdm
# import os
# import torch
# from datasets import Dataset
# from transformers import TrainingArguments
# from trl import SFTTrainer
# from datasets import load_dataset
# from transformers import TrainingArguments, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer
# import json
# import shutil
# from transformers import Qwen2_5_VLProcessor
# from trl import SFTConfig
# import re
# import gc

# ── Toggle ───────────────────────────────────────────────────────────────────
SMOKE      = False
BATCH_SIZE = 10 # inference
T_BATCH_SIZE = 1
TRAIN_EPOCHS = 5#+15
# ─────────────────────────────────────────────────────────────────────────────

images_dir = "/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/images"
test_file  = "/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/test.csv"
train_file = "/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/train.csv"
model_path_3b = "/home/soffer/kaggle/MuseumSCAT/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct"
dest_dir   = "/home/soffer/kaggle/MuseumSCAT/working/output"

writable_model_path = "/home/soffer/kaggle/working/model_patched"
