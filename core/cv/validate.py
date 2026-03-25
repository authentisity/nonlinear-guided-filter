import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

# 1. Load your images (ensure they are the same size and type)
# Assuming they are loaded as standard 8-bit RGB images (0-255)
ground_truth = cv2.imread("img2.png")
filtered_output = cv2.imread("noisy_3_guided.png")

# 2. Calculate PSNR
# data_range=255 because it's an 8-bit image. If using float32 (0.0 to 1.0), set to 1.0
psnr_value = psnr(ground_truth, filtered_output, data_range=255)

# 3. Calculate SSIM
# channel_axis=-1 tells the function it is an RGB image, not grayscale
ssim_value = ssim(ground_truth, filtered_output, data_range=255, channel_axis=-1)

print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.4f}")
