import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.io import read_image

# 1. Load the target image
p = read_image("noisy_1.png").float() / 255.0

# 2. Create a 1-channel Grayscale guide image
I = p.mean(dim=0, keepdim=True) 

# 3. Parameters
r = 31
e = 0.1

def disp(img): 
    return (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

# Safe, fast Box Blur with reflect padding to prevent dark edges
def box_blur(x, kernel, r):
    pad = r // 2
    x_padded = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)
    groups = kernel.shape[0]
    return F.conv2d(x_padded, kernel, padding=0, groups=groups)

@torch.no_grad()
def second_order_cross_guided_filter(p, I, r, e):
    c = p.shape[0] 
    
    # Kernels for 1-channel (Guide) and c-channels (Image)
    kernel_1 = torch.ones(1, 1, r, r, device=p.device) / (r**2)
    kernel_c = torch.ones(c, 1, r, r, device=p.device) / (r**2)

    # 1. Calculate all the powers of the Guide (I)
    I2 = I * I
    I3 = I2 * I
    I4 = I3 * I

    # 2. Calculate the Cross-products
    Ip = I * p
    I2p = I2 * p

    # 3. Calculate all Local Means
    mean_I  = box_blur(I, kernel_1, r)
    mean_I2 = box_blur(I2, kernel_1, r)
    mean_I3 = box_blur(I3, kernel_1, r)
    mean_I4 = box_blur(I4, kernel_1, r)
    
    mean_p   = box_blur(p, kernel_c, r)
    mean_Ip  = box_blur(Ip, kernel_c, r)
    mean_I2p = box_blur(I2p, kernel_c, r)

    # 4. Calculate Variances and Covariances
    cyx  = mean_Ip - mean_I * mean_p
    cyx2 = mean_I2p - mean_I2 * mean_p
    cxx2 = mean_I3 - mean_I2 * mean_I
    
    vx1  = mean_I2 - mean_I * mean_I
    vx2  = mean_I4 - mean_I2 * mean_I2

    # 5. Calculate Linear Coefficients (Beta 1, Beta 2, Alpha)
    vx1_reg = vx1 + e
    vx2_reg = vx2 + e
    
    # Determinant of the covariance matrix
    D = (vx1_reg * vx2_reg) - (cxx2 * cxx2) 
    D = D.clamp(min=1e-8) 

    beta1 = (cyx * vx2_reg - cyx2 * cxx2) / D
    beta2 = (cyx2 * vx1_reg - cyx * cxx2) / D
    alpha = mean_p - (beta1 * mean_I) - (beta2 * mean_I2)

    # 6. Smooth the coefficients
    mean_beta1 = box_blur(beta1, kernel_c, r)
    mean_beta2 = box_blur(beta2, kernel_c, r)
    mean_alpha = box_blur(alpha, kernel_c, r)

    # 7. Final Quadratic Output: q = (beta2 * I^2) + (beta1 * I) + alpha
    q = (mean_beta2 * I2) + (mean_beta1 * I) + mean_alpha
    
    return q

# Run the updated 2nd order filter
q = second_order_cross_guided_filter(p, I, r, e)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.set_title("Original Noisy Image")
ax1.imshow(disp(p))
ax1.axis('off')

ax2.set_title("2nd-Order Grayscale-Guided Output")
ax2.imshow(disp(q))
ax2.axis('off')

plt.imsave("noisy_1_2nd_order_guided.png", disp(q))
plt.show()
