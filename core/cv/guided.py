import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.io import read_image

# 1. Load the target image
p = read_image("noisy_1.png").float() / 255.0

# 2. THE FIX: Create a 1-channel Grayscale guide image.
# Averaging the RGB channels removes pure color noise and isolates physical edges.
I = p.mean(dim=0, keepdim=True) 

r = 31
e = 0.1

def disp(img): 
    return (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

def cross_guided_filter(p, I, r, e):
    
    # We need separate kernels for 1-channel and 3-channel convolutions
    kernel_1 = torch.ones(1, 1, r, r).to(p.device) / (r**2)
    kernel_3 = torch.ones(3, 1, r, r).to(p.device) / (r**2)
    pad = r // 2

    # Averages
    mean_p = F.conv2d(p, kernel_3, padding=pad, groups=3)
    mean_I = F.conv2d(I, kernel_1, padding=pad)
    
    # Covariance & Variance (PyTorch automatically broadcasts the 1-channel I to match p)
    corr_I = F.conv2d(I * I, kernel_1, padding=pad)
    corr_Ip = F.conv2d(I * p, kernel_3, padding=pad, groups=3) 

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p 

    # Linear coefficients
    a = cov_Ip / (var_I + e) 
    b = mean_p - a * mean_I 
    
    mean_a = F.conv2d(a, kernel_3, padding=pad, groups=3)
    mean_b = F.conv2d(b, kernel_3, padding=pad, groups=3)
    
    # Final output
    q = mean_a * I + mean_b 
    return q

# Run the updated filter
q = cross_guided_filter(p, I, r, e)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.set_title("Original Noisy Image")
ax1.imshow(disp(p))
ax1.axis('off')

ax2.set_title("Grayscale-Guided Output")
ax2.imshow(disp(q))
ax2.axis('off')

plt.imsave("noisy_3_guided.png", disp(q))
plt.show()
