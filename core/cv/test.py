import cv2
import numpy as np

# Load a clean, noise-free image
clean_img = cv2.imread('img.png')

# 1. Define the noise parameters
mean = 0
sigma = 25 # Increase this to make the noise heavier and more obvious

# 2. Generate the Gaussian noise (must match the image shape)
# We generate it as floats to avoid clipping issues during math
gauss_noise = np.random.normal(mean, sigma, clean_img.shape)

# 3. Add the noise to the image
# We cast the clean image to float, add the noise, clip it to valid 0-255 ranges, 
# and cast it back to standard 8-bit unsigned integers for OpenCV
noisy_img = clean_img.astype(np.float32) + gauss_noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Now you can feed 'noisy_img' into your Gaussian, Bilateral, and Guided filters!
cv2.imwrite('noisy_1.png', noisy_img)
