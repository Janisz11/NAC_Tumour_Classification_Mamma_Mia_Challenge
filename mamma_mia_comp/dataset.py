# dataset.py
import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

TARGET_SHAPE = (128, 128, 128)  # finalny rozmiar po resize
MAX_PHASES = 5  # maksymalnie 5 faz DCE-MRI
MARGIN = 10  # margines dookoła maski w voxelach
EPS = 1e-8


# -----------------------
# Normalizacja wspólna dla wszystkich faz pacjenta
# -----------------------
def _normalize_stack(phase_list_np):
    """
    Z-score dla całego stosu faz: μ i σ liczone z fazy 0,
    a następnie stosowane do KAŻDEJ fazy.
    """
    baseline = phase_list_np[0].astype(np.float32)
    mu = baseline.mean()
    std = baseline.std() + EPS
    return [(ph.astype(np.float32) - mu) / std for ph in phase_list_np]


# -----------------------
# Pomocnicze funkcje TIC
# -----------------------
def _generate_tic_curves(image_stack: np.ndarray, mask: np.ndarray):
    """Zwraca słownik {voxel_index: [I/I0 w kolejnych fazach]}"""
    tic_curves = {}
    I0 = image_stack[0].astype(np.float32) + EPS
    nz = np.where(mask > 0)
    for idx in zip(*nz):
        curve = image_stack[:, idx[0], idx[1], idx[2]] / I0[idx]
        tic_curves[idx] = curve.tolist()
    return tic_curves


def _compute_voxel_tic_features(tic):
    tic = np.array(tic, dtype=np.float32)
    n_phases = len(tic)
    baseline = tic[0]
    peak_idx = int(np.argmax(tic))
    peak_val = tic[peak_idx]
    last_val = tic[-1]

    wash_in_rate = (peak_val - baseline) / (peak_idx + EPS)
    wash_out_enh = (last_val - peak_val) / (peak_val + EPS)

    if peak_idx < n_phases - 1:
        x = np.arange(peak_idx, n_phases).reshape(-1, 1)
        y = tic[peak_idx:]
        mdl = LinearRegression().fit(x, y)
        y_pred = mdl.predict(x)
        rss = np.sum((y - y_pred) ** 2)
        wash_out_stab = rss / ((baseline + EPS) * (n_phases - peak_idx))
    else:
        wash_out_stab = 0.0

    return wash_in_rate, wash_out_enh, wash_out_stab


def _aggregate_tic_features(image_stack: np.ndarray, mask: np.ndarray):
    """Zwraca tensor [log_voxel_count, avg_wash_in, avg_wash_out_enh, avg_wash_out_stab]"""
    tic_curves = _generate_tic_curves(image_stack, mask)
    voxel_count = len(tic_curves)
    if voxel_count == 0:
        return torch.zeros(4)

    total_in, total_out, total_stab = 0.0, 0.0, 0.0
    for tic in tic_curves.values():
        wi, wo_enh, wo_stab = _compute_voxel_tic_features(tic)
        total_in += wi
        total_out += wo_enh
        total_stab += wo_stab

    avg_wi = total_in / voxel_count
    avg_wo_e = total_out / voxel_count
    avg_wo_s = total_stab / voxel_count

    log_voxel_count = torch.log10(torch.tensor([voxel_count + EPS]))
    normed_features = torch.tensor([avg_wi, avg_wo_e, avg_wo_s], dtype=torch.float32)
    return torch.cat([log_voxel_count, normed_features], dim=0)


# -----------------------
# Dataset
# -----------------------
class MammaMiaCompetitionDataset(Dataset):
    def __init__(self, patient_ids, images_root, clinical_xlsx, segmentation_root):
        self.patient_ids = patient_ids
        self.images_root = images_root
        self.segmentation_root = segmentation_root
        clin_df = pd.read_excel(clinical_xlsx).dropna(subset=["pcr"])
        self.labels = dict(zip(clin_df["patient_id"], clin_df["pcr"].astype(int)))

    # ---------- utils ----------
    def _torch_from_np(self, vol_np):
        return torch.tensor(vol_np[None, ...], dtype=torch.float32)  # [1,D,H,W]

    def _resize(self, vol_t, target_shape):
        return F.interpolate(
            vol_t.unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False
        ).squeeze(0)

    # ---------- main ----------
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pid_dir = os.path.join(self.images_root, pid)

        # ---- Wczytanie faz ----
        phase_paths = sorted(
            [os.path.join(pid_dir, f) for f in os.listdir(pid_dir) if f.endswith(".nii.gz")]
        )[:MAX_PHASES]

        phase_vols_np = [
            nib.load(p).get_fdata().astype(np.float32) for p in phase_paths
        ]
        while len(phase_vols_np) < MAX_PHASES:
            phase_vols_np.append(np.zeros_like(phase_vols_np[0]))

        # ---- Normalizacja wspólna (μ, σ z fazy 0) ----
        phase_vols_np = _normalize_stack(phase_vols_np)

        # ---- Wczytanie maski ----
        mask_path = os.path.join(self.segmentation_root, f"{pid}.nii.gz")
        if not os.path.exists(mask_path):
            mask_np = np.zeros_like(phase_vols_np[0])
        else:
            mask_np = (nib.load(mask_path).get_fdata() > 0).astype(np.uint8)

        # ---- Crop bbox ----
        nz = np.where(mask_np > 0)
        if len(nz[0]) > 0:
            zmin, ymin, xmin = np.min(nz[0]), np.min(nz[1]), np.min(nz[2])
            zmax, ymax, xmax = np.max(nz[0]), np.max(nz[1]), np.max(nz[2])
            zmin = max(zmin - MARGIN, 0)
            ymin = max(ymin - MARGIN, 0)
            xmin = max(xmin - MARGIN, 0)
            zmax = min(zmax + MARGIN + 1, mask_np.shape[0])
            ymax = min(ymax + MARGIN + 1, mask_np.shape[1])
            xmax = min(xmax + MARGIN + 1, mask_np.shape[2])

            phase_vols_np = [v[zmin:zmax, ymin:ymax, xmin:xmax] for v in phase_vols_np]
            mask_np = mask_np[zmin:zmax, ymin:ymax, xmin:xmax]

        image_stack_np = np.stack(phase_vols_np, axis=0)  # [T,D,H,W]
        tic_features = _aggregate_tic_features(image_stack_np, mask_np)  # tensor(4,)

        phase_vols_t = [self._torch_from_np(v) for v in phase_vols_np]
        mask_t = self._torch_from_np(mask_np.astype(np.float32))

        phase_vols_t = [self._resize(v, TARGET_SHAPE) for v in phase_vols_t]
        mask_t = self._resize(mask_t, TARGET_SHAPE)

        while len(phase_vols_t) < MAX_PHASES:
            phase_vols_t.append(torch.zeros((1, *TARGET_SHAPE)))

        x_img = torch.cat(phase_vols_t, dim=0)  # [MAX_PHASES,D,H,W]
        x = torch.cat([x_img, mask_t], dim=0)  # +1 kanał maski

        y = torch.tensor(self.labels[pid], dtype=torch.long)
        return x, y, tic_features
