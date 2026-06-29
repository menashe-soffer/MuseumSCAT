import os
import tqdm
import cv2
import glob

from sklearn.datasets import images

from paths_and_constants import *

for_card_folder = images_dir.replace('images', 'for_card')
for_specimen_folder = images_dir.replace('images', 'for_specimen')
os.makedirs(for_specimen_folder, exist_ok=True)
os.makedirs(for_card_folder, exist_ok=True)

image_files = glob.glob(os.path.join(images_dir, "*.jpeg") )
#print(image_files)
for image_file in tqdm.tqdm(image_files):
    img = cv2.imread(image_file)
    h, w = img.shape[:2]
    # crop for card
    crop_img = img[:int(h*7/16), int(w*4/16):-int(w*1/16)]
    cv2.imwrite(os.path.join(for_card_folder, os.path.basename(image_file)), crop_img)
    # crop for specimen
    crop_img = img[int(h*5/16):-int(h*2/16), int(w*4/16):-int(w*3/16)]
    cv2.imwrite(os.path.join(for_specimen_folder, os.path.basename(image_file)), crop_img)
