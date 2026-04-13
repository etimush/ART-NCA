import cv2
import numpy as np
import torch
import itertools

import utils
from NCA import *
from utils import *

DEVICE = "cuda:0"
scale = 5
HEIGHT = 100 * scale
WIDTH = 100 * scale
CHANNELS = 24
GENESIZE = 4
noise_freq = 100

# Load NCA
with torch.no_grad():
    ca = GeneCA(CHANNELS, 256, gene_size=GENESIZE)
    ca.load_state_dict(torch.load("Trained_models/art_4.pth"))
    ca.to(DEVICE)
ca.eval()

# 1. Canvas starts blank
x = torch.zeros(1, CHANNELS, HEIGHT, WIDTH).to(DEVICE)
x[:, -GENESIZE:] = 0

# --- App State Variables ---
view_mode = "genes"  # 'genes' or 'nca'
paused = True  # Starts paused
eraser_mode = False  # Eraser toggle
brush_size = 15
gene_index = 0
is_drawing = False

# Automatically infer all binary permutations for the given GENESIZE (excluding all-zeros)
gene_combos = list(itertools.product([0, 1], repeat=GENESIZE))
gene_combos.remove((0,) * GENESIZE)
perms = len(gene_combos)
colors = utils.get_n_distinct_colors(perms, DEVICE)

# Cache a NumPy array of the distinct colors mapped to BGR (OpenCV format)
# for efficient drawing and previewing
colors_np_bgr = colors.cpu().numpy()[:, ::-1]

# --- Pre-compute Brush Previews ---
preview_cache = {}
print(f"Pre-computing {len(gene_combos)} paintbrush previews for 200 steps each. Please wait...")

for combo in gene_combos:
    p_size = 120  # Mini lattice size
    p_x = torch.rand(1, CHANNELS, p_size, p_size).to(DEVICE)

    print(combo)
    # Create mask for the initial seed

    noise = torch.rand((1, CHANNELS, p_size, p_size), device=DEVICE)
    noise[:, :3] = utils.simple_rgb_perlin(1,p_size, p_size, noise_freq, DEVICE)
    noise[:, 3:] = 0
    noise[:, 3] = 1
    for i in range(len(combo)):
        noise[:, -GENESIZE + i] = combo[-GENESIZE + i]

    p_x = noise

    for _ in range(400):
        p_x = ca(p_x, 0.5)
        p_x = p_x.detach()

    # Extract RGB channels, clip, convert to BGR and resize
    img = p_x[0, :3].clone().detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)

    preview_cache[combo] = img

print("Previews generated successfully!")


# --- Mouse Callback for Drawing ---
def draw_callback(event, mouse_x, mouse_y, flags, param):
    global x, is_drawing, brush_size, gene_index, gene_combos, eraser_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

    if is_drawing and (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN):
        cx = int(mouse_x * (WIDTH / 1000))
        cy = int(mouse_y * (HEIGHT / 1000))

        mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        cv2.circle(mask, (cx, cy), brush_size, 1.0, -1)
        mask_t = torch.tensor(mask, device=DEVICE, dtype=torch.float32)
        combo = gene_combos[gene_index]

        with torch.no_grad():
            if eraser_mode:
                # Eraser: Set all channels to 0
                for c in range(CHANNELS):
                    x[0, c] = torch.where(mask_t > 0, torch.tensor(0.0, device=DEVICE), x[0, c])
            else:
                # Paintbrush: Set RGB to noise, Alpha to 1, assign genes
                # noise = torch.rand((3, HEIGHT, WIDTH), device=DEVICE)
                noise = utils.simple_rgb_perlin(1,HEIGHT, WIDTH, noise_freq, DEVICE)[0]
                for c in range(CHANNELS):
                    if c < 3:
                        x[0, c] = torch.where(mask_t > 0, noise[c], x[0, c])
                    elif c == 3:
                        x[0, c] = torch.where(mask_t > 0, torch.tensor(1.0, device=DEVICE), x[0, c])
                    elif c >= CHANNELS - GENESIZE:
                        idx = c - (CHANNELS - GENESIZE)
                        x[0, c] = torch.where(mask_t > 0, torch.tensor(float(combo[idx]), device=DEVICE), x[0, c])
                    else:
                        x[0, c] = torch.where(mask_t > 0, torch.tensor(0.0, device=DEVICE), x[0, c])


# --- Setup Windows ---
cv2.namedWindow("NCA Canvas", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("NCA Canvas", draw_callback)
cv2.namedWindow("Info", cv2.WINDOW_AUTOSIZE)

# --- Main Loop ---
while True:
    if not paused:
        with torch.no_grad():
            x = ca(x, 0.5)
            mask = x[:, -GENESIZE:].sum(1, keepdim=True) > 0
            x = x * mask
            x = x.detach()

    x_np = x.clone().detach().cpu().permute(0, 2, 3, 1).numpy()[0]

    # --- Canvas Visualization ---
    if view_mode == "genes":
        # Start with a black canvas
        vis_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        # Round the gene channels to easily match combo patterns
        # (since NCAs might shift values slightly during growth)
        last_channels = np.round(x_np[:, :, -GENESIZE:])

        for i, c_combo in enumerate(gene_combos):
            combo_arr = np.array(c_combo, dtype=np.float32)
            # Find which pixels match this exact gene permutation
            match = np.all(last_channels == combo_arr, axis=-1)
            # Paint matching pixels with the corresponding distinct BGR color
            vis_img[match] = colors_np_bgr[i]

    else:
        # Show actual NCA (RGB channels)
        vis_img = x_np[:, :, :3]
        # vis_img[:,:,3] = 1
        # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGRA)

    vis_img = np.clip(vis_img, 0.0, 1.0)
    display_img = cv2.resize(vis_img, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("NCA Canvas", display_img)

    # --- Info Window Visualization ---
    info_img = np.ones((350, 500, 3), dtype=np.float32) * 0.15
    combo = gene_combos[gene_index]

    # Text States
    cv2.putText(info_img, f"View : {'DRAWING (Genes)' if view_mode == 'genes' else 'NCA'} [SPACE]", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1, 1, 1), 2)
    cv2.putText(info_img, f"State: {'PAUSED' if paused else 'RUNNING'} ['p']", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 1) if paused else (0, 1, 0), 2)
    cv2.putText(info_img, f"Brush: {'ERASER' if eraser_mode else 'PAINT'} ['e']", (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0.5, 0.5, 1) if eraser_mode else (1, 1, 1), 2)
    cv2.putText(info_img, f"Size : {brush_size} [Left/Right]", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1, 1, 1), 2)
    cv2.putText(info_img, f"Combo: {combo} [Up/Down]", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1, 1, 1), 2)

    # Color Block Preview using the generated distinct colors
    color_bgr = colors_np_bgr[gene_index]
    preview_color = (float(color_bgr[0]), float(color_bgr[1]), float(color_bgr[2]))  # Open CV expects BGR float tuple
    cv2.rectangle(info_img, (30, 220), (90, 280), preview_color, -1)
    cv2.rectangle(info_img, (30, 220), (90, 280), (1, 1, 1), 1)
    cv2.putText(info_img, "Color", (40, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    # Generated 200-step Brush Image Preview
    p_img = preview_cache[combo]
    info_img[200:300, 130:230] = p_img
    cv2.rectangle(info_img, (130, 200), (230, 300), (1, 1, 1), 1)
    cv2.putText(info_img, "200-Step Brush Result", (105, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    cv2.imshow("Info", info_img)

    # --- Keyboard Controls ---
    key = cv2.waitKeyEx(1)

    if key == 27:  # ESC key
        break
    elif key == 32:  # SPACE key: Toggle View Mode
        if view_mode == "genes":
            view_mode = "nca"
            paused = False
        else:
            view_mode = "genes"
            paused = True
    elif key in [ord('p'), ord('P')]:  # P key: Toggle Pause safely
        paused = not paused
    elif key in [ord('e'), ord('E')]:  # E key: Toggle Eraser
        eraser_mode = not eraser_mode

    # Left arrow
    elif key in [2424832, 65361, 81]:
        brush_size = max(1, brush_size - 1)
    # Right arrow
    elif key in [2555904, 65363, 83]:
        brush_size += 1
    # Up arrow
    elif key in [2490368, 65362, 82]:
        gene_index = (gene_index + 1) % len(gene_combos)
    # Down arrow
    elif key in [2621440, 65364, 84]:
        gene_index = (gene_index - 1) % len(gene_combos)

cv2.destroyAllWindows()