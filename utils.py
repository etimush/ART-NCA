import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import colorsys
import torch.nn.functional as F


def get_n_distinct_colors(n, device='cpu'):
    """
    Generates N visually distinct RGB colors.
    Returns a PyTorch tensor of shape (N, 3) with values between [0.0, 1.0].
    """
    colors = []

    for i in range(n):
        # Evenly divide the 360-degree color wheel
        hue = i / n

        # High saturation and value to keep the colors bright and distinguishable
        saturation = 0.85
        value = 0.90

        # Convert to RGB (returns floats from 0.0 to 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return torch.tensor(colors, dtype=torch.float32, device=device)

def get_image(path,height=50, width=50, padding =0):
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = cv2.resize(base, (int(height), int(width)), interpolation=cv2.INTER_AREA)
    base_2 = base / 255

    #base_2[..., :3] *= base_2[..., 3:]
    base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).cuda()
    base_torch = torch.nn.functional.pad(base_torch, [padding,padding,padding,padding ])
    base_tt = base_torch.cpu().permute((1, 2, 0)).clone().detach().numpy()
    return base_torch,base_tt

def make_gene_pool(gene_location,pool_size = 250, height = 50, width= 50, channels = 12, device = "cuda:0", freq = 10):
    seed = torch.rand((pool_size,channels, height, width), device=device)
    seed[:,3:] = 0
    seed[:, 3] = 1
    seed[:,:3] = simple_rgb_perlin(pool_size,height,width,freq,device=device)
    for gene_loc in gene_location:
        seed[:,channels-1
             -gene_loc] = 1


    return seed

def make_gene_pool_static(gene_location,pool_size = 250, height = 50, width= 50, channels = 12, device = "cuda:0", gene_size = 3):
    seed = torch.rand((pool_size,channels, height, width), device=device)
    seed[:,3:] = 0
    seed[:, 3] = 1
    seed[:,:3] = simple_rgb_perlin(height,width,10,device=device)
    for gene_loc in gene_location:
        seed[:,channels-1
             -gene_loc] = 1


    return seed


def get_gene_pool(pools, partitions, seeds):
    idxs = []
    pool_tot = []
    for part, pool, seed in zip(partitions, pools, seeds):
        idx = np.random.choice(pool.shape[0], part, replace=False)
        idxs.append(idx)
        p = pool[idx]
        p[0:1] = seed.clone()
        pool_tot.append(p)
    return idxs, torch.cat(pool_tot,dim=0)



def udate_gene_pool(pools,results, idxs, partitions):
    pool_new =[]
    cum_idx = 0
    for pool, idx, part in zip(pools, idxs, partitions):
        pool[idx] = results[cum_idx:part+cum_idx]
        cum_idx+=part
        pool_new.append(pool)
    return pool_new

def update_problem_pool(pools, results, idxs, pool_id):
    pool_new = []
    for p in range(len(pools)):
        if p != pool_id:
            pool_new.append(pools[p])
        else:
            pool = pools[p]
            pool[idxs] = results
            pool_new.append(pool)
    return pool_new


def extra_features(I:torch.Tensor, n_levels, k):
    F = []
    Ii = I
    GB = torchvision.transforms.GaussianBlur(5)
    for _ in range(n_levels):
        Isharp = Ii + 2*(Ii-GB.forward(Ii))
        Fl = Isharp.unfold(1,3,1).unfold(2,k,1).unfold(3,k,1)

        F.append(Fl)
        transform = transforms.Resize(size=Ii.shape[-1]//2)
        Ii = transform(Ii)
    return F


vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
vgg16.to("cuda:0")


def calc_styles_vgg(imgs, vgg_model):
    style_layers = [1, 6, 11, 18, 25]
    device = imgs.device  # Make device dynamic instead of hardcoded

    # Reshape mean and std properly for broadcasting (B, C, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (imgs - mean) / std

    b, c, h, w = x.shape
    # Include raw pixel space as the first feature for exact color palette matching
    features = [x.reshape(b, c, h * w)]

    for i, layer in enumerate(vgg_model[:max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            features.append(x.reshape(b, c, h * w))

    return features


def project_sort(x, proj):
    # x: (Batch, Channels, N_pixels)
    # proj: (Channels, Proj_N)
    # Result sorted along the pixel dimension (dim=-1)
    projected = torch.einsum('bcn,cp->bpn', x, proj)
    return projected.sort(dim=-1)[0]


def ot_loss(source, target, proj_n=64):
    device = source.device
    ch, n_source = source.shape[-2:]
    _, n_target = target.shape[-2:]

    # Generate random projections
    projs = torch.randn(ch, proj_n, device=device)
    projs = F.normalize(projs, dim=0)

    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)

    # If NCA image and Target image are different sizes, align their quantiles
    if n_source != n_target:
        # Use 'linear' instead of 'nearest'. Linear interpolation of sorted arrays
        # gives the mathematically accurate Wasserstein distance for unequal sizes.
        target_proj = F.interpolate(target_proj, size=n_source, mode='linear', align_corners=False)

    # Use mean() instead of sum() to prevent exploding gradients on large images
    return (source_proj - target_proj).square().mean()


def create_vgg_loss(target_img):
    vgg_model = vgg16
    # Pre-calculate target features ONCE outside the training loop
    with torch.no_grad():
        target_features = calc_styles_vgg(target_img, vgg_model)

    def loss_f(imgs):
        source_features = calc_styles_vgg(imgs, vgg_model)

        # Calculate SWD for every layer
        loss = 0
        for s, t in zip(source_features, target_features):
            loss += ot_loss(s, t)

        return loss

    return loss_f

def show_batch(results, channels=4, fig_num = 3):
    x = results.cpu().clone().permute((0, 2, 3, 1)).detach().numpy()
    plt.figure(fig_num)
    plt.clf()
    num = results.shape[0]
    if num > 8:
        num = 8
    for i in range(num):
        img = np.clip(x[i, :, :, 0:channels],0,1)
        img = img[...,::-1]
        plt.figure(fig_num)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)


import math
import torch
import math

def simple_rgb_perlin(batch_size, height, width, frequency, device='cpu'):
    """
    Generates a batch of single-frequency RGB Perlin noise safely mapped to [0, 1].
    Outputs tensor of shape: (batch_size, 3, height, width)
    """
    # Create normalized grid coordinates based on frequency using arange
    y = (torch.arange(height, device=device, dtype=torch.float32) / height) * frequency
    x = (torch.arange(width, device=device, dtype=torch.float32) / width) * frequency
    y, x = torch.meshgrid(y, x, indexing='ij')

    # Get integer and fractional grid cell bounds
    y_int = y.long()
    x_int = x.long()
    y_frac = y - y_int
    x_frac = x - x_int

    # WRAP coordinates using modulo to guarantee we never go out of bounds
    y0 = y_int % frequency
    x0 = x_int % frequency
    y1 = (y0 + 1) % frequency
    x1 = (x0 + 1) % frequency

    # Smoothstep fade function for blending
    fade_y = y_frac**3 * (y_frac * (y_frac * 6 - 15) + 10)
    fade_x = x_frac**3 * (x_frac * (x_frac * 6 - 15) + 10)

    # Generate random gradient vectors (Now includes batch_size)
    # Shape: (batch_size, 3 channels, frequency, frequency)
    angles = 2 * math.pi * torch.rand((batch_size, 3, frequency, frequency), device=device)
    grad_y, grad_x = torch.sin(angles), torch.cos(angles)

    # Helper function to calculate dot products
    def dot(yi, xi, dy, dx):
        # grad_y[:, :, yi, xi] grabs the correct gradients for the entire batch and channel dims
        # Resulting shape before multiplication is (batch_size, 3, height, width)
        return grad_y[:, :, yi, xi] * dy + grad_x[:, :, yi, xi] * dx

    # Calculate dot products for the 4 corners of the grid cells
    n00 = dot(y0, x0, y_frac, x_frac)
    n10 = dot(y1, x0, y_frac - 1, x_frac)
    n01 = dot(y0, x1, y_frac, x_frac - 1)
    n11 = dot(y1, x1, y_frac - 1, x_frac - 1)

    # Interpolate along x, then y
    # The spatial fades (H, W) broadcast perfectly against the (B, 3, H, W) dot products
    nx0 = n00 + fade_x * (n01 - n00)
    nx1 = n10 + fade_x * (n11 - n10)
    noise = nx0 + fade_y * (nx1 - nx0)

    # Standard Perlin bounds mapped to[0.0, 1.0]
    noise = (noise * math.sqrt(2) + 1.0) / 2.0
    return torch.clamp(noise, 0.0, 1.0)





dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
dinov2.to("cuda:0")
for param in dinov2.parameters():
    param.requires_grad = False


def calc_styles_dino(imgs, dino_model):
    device = imgs.device
    b, c, h, w = imgs.shape

    # 1. Include raw pixel space as the first feature for exact color matching
    features = [imgs.reshape(b, c, h * w)]

    # 2. DINOv2 STRICTLY requires image dimensions to be multiples of 14.
    # We dynamically resize the image to the nearest multiple of 14.
    new_h = (h // 14) * 14
    new_w = (w // 14) * 14

    if new_h != h or new_w != w:
        x = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    else:
        x = imgs

    # 3. Standardize image for DINOv2 (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    # 4. Extract deep semantic features from DINOv2.
    # 'get_intermediate_layers' grabs the patch tokens from the last N transformer blocks.
    # n=4 extracts the last 4 blocks (capturing deep spatial shapes and motifs).
    intermediate_outputs = dino_model.get_intermediate_layers(x, n=4, return_class_token=False)

    for out in intermediate_outputs:
        # DINO outputs shape: (Batch, Num_Patches, Channels)
        # We MUST permute to (Batch, Channels, Num_Patches) for the SWD project_sort
        features.append(out.permute(0, 2, 1))

    return features


def project_sort(x, proj):
    # x: (Batch, Channels, N_pixels)
    # proj: (Channels, Proj_N)
    projected = torch.einsum('bcn,cp->bpn', x, proj)
    return projected.sort(dim=-1)[0]


def ot_loss(source, target, proj_n=64):
    device = source.device
    ch, n_source = source.shape[-2:]
    _, n_target = target.shape[-2:]

    # Generate random projections
    projs = torch.randn(ch, proj_n, device=device)
    projs = F.normalize(projs, dim=0)

    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)

    # Align quantiles if source and target have different numbers of patches
    if n_source != n_target:
        target_proj = F.interpolate(target_proj, size=n_source, mode='linear', align_corners=False)

    return (source_proj - target_proj).square().mean()


def create_dino_loss(target_img):
    # Pre-calculate target features ONCE outside the training loop
    with torch.no_grad():
        target_features = calc_styles_dino(target_img, dinov2)

    def loss_f(imgs):
        source_features = calc_styles_dino(imgs, dinov2)

        # Calculate SWD for every layer
        loss = 0

        # WEIGHTING TRICK FOR SHAPES:
        # [Raw Colors, DINO Block 1, Block 2, Block 3, Block 4]
        # We put massive weight on Block 4. This forces the NCA to prioritize
        # generating coherent motifs over just color-matching.
        layer_weights = [10.0, 5.0, 2.0, 1.0, 1.0]

        for w, s, t in zip(layer_weights, source_features, target_features):
            loss += w * ot_loss(s, t)

        return loss

    return loss_f


def create_hybrid_loss(target_img):
    # Pre-calculate target features ONCE outside the training loop
    with torch.no_grad():
        target_vgg = calc_styles_vgg(target_img, vgg16)  # Only use relu1_1 and relu2_1
        target_dino = calc_styles_dino(target_img, dinov2)  # Use deep blocks

    def loss_f(imgs):
        source_vgg = calc_styles_vgg(imgs, vgg16)
        source_dino = calc_styles_dino(imgs, dinov2)

        loss = 0

        # 1. VGG LOSS: Enforces crisp pixels, sharp edges, and exact colors
        # We only use the first two VGG layers (e.g., indices 0 and 1 from your list)
        vgg_weights = [1.0, 1.0]  # Adjust weights as needed
        for w, s, t in zip(vgg_weights, source_vgg[:2], target_vgg[:2]):
            loss += w * ot_loss(s, t)

        # 2. DINO LOSS: Enforces macro-motifs and coherent shapes
        # We only use the deepest DINO blocks, ignoring the raw pixels
        dino_weights = [2.0, 1.0, 1.0]
        for w, s, t in zip(dino_weights, source_dino[-3:], target_dino[-3:]):
            # Normalizing DINO tokens before SWD helps mitigate positional bias
            s_norm = F.normalize(s, dim=1)
            t_norm = F.normalize(t, dim=1)
            loss += w * ot_loss(s_norm, t_norm)

        return loss

    return loss_f


def rgb_color_loss(source, target, proj_n=64):
    # This applies SWD to the raw RGB channels (3 channels)
    # This forces the NCA to perfectly match the color histogram of the target
    return ot_loss(source.view(source.shape[0], 3, -1),
                   target.view(target.shape[0], 3, -1),
                   proj_n=proj_n)