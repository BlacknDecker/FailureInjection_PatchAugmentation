import random
from pathlib import Path

import cv2
import numpy as np

from AugmentedPatchInjector import AugmentedPatchInjector

# CONFIG
original_p = Path.cwd() / "000001.png"
out_folder = Path.cwd() / "augmented"
patches_folder = Path.cwd() / "patches"
SEED = 42069
SAMPLES = 100

# SETUP
random.seed(SEED)
np.random.seed(SEED)
original = cv2.imread(str(original_p), cv2.IMREAD_GRAYSCALE)
original_H, original_W = original.shape
injector = AugmentedPatchInjector(patches_folder, target_height=original_H, target_width=original_W)

# TEST
print("Start...")
for i in range(SAMPLES):
    print(f"{i}/{SAMPLES}")
    injected = injector.inject(original)
    # Save
    cv2.imwrite(str(out_folder/f"sample_{str(i).zfill(3)}.png"), injected)
print("Done!")




