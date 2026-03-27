import torch
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt

# ===================== MODEL =====================

class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def exp(c):
            return torch.nn.Sequential(
                torch.nn.Conv2d(c, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, 3, padding=1),
                torch.nn.ReLU()
            )

        self.exp_s2 = exp(12)
        self.exp_list = torch.nn.ModuleList([exp(1) for _ in range(8)])

        self.gate = torch.nn.Sequential(
            torch.nn.Conv2d(20, 12, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(12, 9, 1),
            torch.nn.Softmax(dim=1)
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(8,1,1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        parts = [
            self.exp_s2(x[:,0:12]),
            *[self.exp_list[i](x[:,12+i:13+i]) for i in range(8)]
        ]
        w = self.gate(x)
        return self.final(sum(w[:,i:i+1]*parts[i] for i in range(9)))

# ===================== LOAD MODEL =====================

model = MoE()
model.load_state_dict(torch.load("C:/Users/pulis/replicate/landslide_model.pth", weights_only=True))
model.eval()

print("Model loaded!")

# ===================== LOAD SENTINEL-2 =====================

s2_folder = "C:/Users/pulis/replicate/2024-12-11-20260327T062215Z-3-001/2024-12-11/Sentinel-2"

bands = []
for file in sorted(os.listdir(s2_folder)):
    if file.endswith(".tif"):
        with rasterio.open(os.path.join(s2_folder, file)) as src:
            bands.append(src.read())

s2 = np.concatenate(bands, axis=0)
s2 = np.nan_to_num(s2)

# normalize
s2 = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-8)

# ===================== LOAD SENTINEL-1 =====================

s1_path = "C:/Users/pulis/replicate/2024-12-11-20260327T062215Z-3-001/2024-12-11/Sentinel-1/Sentinel-1_2024-12-11_SAR.tif"

with rasterio.open(s1_path) as src:
    s1 = src.read()

s1 = np.nan_to_num(s1)
s1 = np.mean(s1, axis=0, keepdims=True)

# normalize
s1 = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-8)

# ===================== CREATE INPUT =====================

H, W = s2.shape[1], s2.shape[2]

X = np.zeros((20, H, W), dtype=np.float32)

# fill Sentinel-2
X[0:12] = s2

# fill Sentinel-1
X[15:16] = s1

X_t = torch.tensor(X).unsqueeze(0)

# ===================== PREDICT =====================

with torch.no_grad():
    out = model(X_t)

prob_map = out.squeeze().numpy()

# ===================== CONTRAST =====================

p_min = np.percentile(prob_map, 5)
p_max = np.percentile(prob_map, 95)
prob_vis = np.clip((prob_map - p_min)/(p_max - p_min + 1e-8), 0, 1)

# ===================== RGB BASE =====================

rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1)

# ===================== OVERLAY =====================

plt.imshow(rgb)
plt.imshow(prob_vis, cmap='jet', alpha=0.5)
plt.axis("off")
plt.title("S1 + S2 Landslide Prediction")

save_path = "C:/Users/pulis/replicate/s1_s2_overlay.png"
plt.savefig(save_path)
plt.close()

print(f"Saved final overlay at: {save_path}")