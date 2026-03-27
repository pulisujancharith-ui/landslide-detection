import os
import rasterio
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import matplotlib.pyplot as plt

# ===================== FUNCTIONS =====================

def load_raster(path):
    with rasterio.open(path) as src:
        return np.nan_to_num(src.read(), nan=0.0)

def resize_to_match(data, target_shape):
    return np.array([
        cv2.resize(b, (target_shape[2], target_shape[1]))
        for b in data
    ])

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def create_patches(X, y, p=64):
    px, py = [], []
    _, H, W = X.shape
    for i in range(0, H-p, p):
        for j in range(0, W-p, p):
            px.append(X[:, i:i+p, j:j+p])
            py.append(y[i:i+p, j:j+p])
    return np.array(px), np.array(py)

def reconstruct(patches, H, W, p=64):
    img = np.zeros((H, W))
    cnt = np.zeros((H, W))
    k = 0
    for i in range(0, H-p, p):
        for j in range(0, W-p, p):
            img[i:i+p, j:j+p] += patches[k]
            cnt[i:i+p, j:j+p] += 1
            k += 1
    return img / (cnt + 1e-8)

# ===================== PATH =====================

base = "C:/Users/pulis/replicate/2024-12-11-20260327T062215Z-3-001/2024-12-11"
save = "C:/Users/pulis/replicate/"

# ===================== LOAD DATA =====================

s2 = np.concatenate([
    load_raster(os.path.join(base, "Sentinel-2", f))
    for f in sorted(os.listdir(base + "/Sentinel-2")) if f.endswith(".tif")
], axis=0)

soil = np.concatenate([
    load_raster(os.path.join(base, "Soil_moisture/Soil_Mositure/Soil_Mositure", f))
    for f in os.listdir(base + "/Soil_moisture/Soil_Mositure/Soil_Mositure") if f.endswith(".tif")
], axis=0)
soil = np.mean(soil, axis=0, keepdims=True)

dem = load_raster(base + "/DEM/Copernicus_DEM_30m.tif")

s1 = load_raster(base + "/Sentinel-1/Sentinel-1_2024-12-11_SAR.tif")
s1 = np.mean(s1, axis=0, keepdims=True)

ds = xr.open_dataset(base + "/Rainfall Data/kerala_rainfall_data.nc")
rain = np.expand_dims(np.nan_to_num(ds["rain"].isel(time=0).values), axis=0)

soil = resize_to_match(soil, s2.shape)
rain = resize_to_match(rain, s2.shape)
dem = resize_to_match(dem, s2.shape)
s1 = resize_to_match(s1, s2.shape)

s2, soil, rain, dem, s1 = map(normalize, [s2, soil, rain, dem, s1])

# ===================== FEATURES =====================

red, nir = s2[3], s2[7]
ndvi = normalize(np.expand_dims((nir-red)/(nir+red+1e-8),0))

gy, gx = np.gradient(dem[0])
slope = normalize(np.expand_dims(np.sqrt(gx**2 + gy**2),0))

relief = normalize(np.expand_dims(dem[0] - np.min(dem[0]),0))
brightness = normalize(np.expand_dims(np.mean(s2, axis=0),0))

# ===================== LABEL =====================

risk = 0.2*soil + 0.2*rain + 0.3*brightness + 0.15*slope + 0.1*s1 + 0.05*relief
label = (risk > np.mean(risk)*0.9).astype(int)
if label.ndim == 3:
    label = label.squeeze(0)

# ===================== INPUT =====================

X = np.concatenate([s2, soil, rain, dem, s1, ndvi, slope, relief, brightness], axis=0)
px, py = create_patches(X, label)

X_t = torch.tensor(px, dtype=torch.float32)
y_t = torch.tensor(py, dtype=torch.float32).unsqueeze(1)

# ===================== MODEL =====================

class MoE(nn.Module):
    def __init__(self):
        super().__init__()

        def exp(c):
            return nn.Sequential(
                nn.Conv2d(c, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU()
            )

        self.exp_s2 = exp(12)
        self.exp_list = nn.ModuleList([exp(1) for _ in range(8)])

        self.gate = nn.Sequential(
            nn.Conv2d(20, 12, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 9, 1),
            nn.Softmax(dim=1)
        )

        self.final = nn.Sequential(
            nn.Conv2d(8,1,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        parts = [
            self.exp_s2(x[:,0:12]),
            *[self.exp_list[i](x[:,12+i:13+i]) for i in range(8)]
        ]
        w = self.gate(x)
        return self.final(sum(w[:,i:i+1]*parts[i] for i in range(9)))

model = MoE()
opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# ===================== TRAIN =====================

for e in range(20):
    opt.zero_grad()
    out = model(X_t)
    loss = loss_fn(out, y_t)
    loss.backward()
    opt.step()
    if e % 2 == 0:
        print(f"Epoch {e+1}, Loss: {loss.item()}")

# ===================== PREDICT =====================

with torch.no_grad():
    out = model(X_t)

prob = reconstruct(out.squeeze().numpy(), *label.shape)

print("Accuracy:", np.mean((prob>0.5)==label))

# ===================== VISUAL FIXED =====================

p_min = np.percentile(prob, 5)
p_max = np.percentile(prob, 95)
prob_vis = np.clip((prob - p_min)/(p_max - p_min + 1e-8), 0, 1)

# Heatmap
plt.imshow(prob_vis, cmap='jet')
plt.colorbar()
plt.title("Risk Map")
plt.savefig(save+"risk_map.png")
plt.close()

# Overlay
rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1)
rgb = normalize(rgb)

plt.imshow(rgb)
plt.imshow(prob_vis, cmap='jet', alpha=0.5)
plt.axis("off")
plt.savefig(save+"overlay.png")
plt.close()

# ===================== SAVE =====================

torch.save(model.state_dict(), save+"landslide_model.pth")
np.save(save+"risk_map.npy", prob)

print("FINAL MODEL COMPLETE")