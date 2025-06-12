import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
import csv
import nibabel as nib


# Generate TIC curve (contrast intensity over time) for each voxel
def generate_voxelwise_tic_curves(image_stack, mask):

    tic_curves = {}
    I0 = image_stack[0].astype(np.float32) + 1e-8  # pre-contrast
    num_phases = image_stack.shape[0]  # number of phases
    print(num_phases)

    for idx in zip(*np.where(mask > 0)):  # voxel coordinates with tumor (3D)
        curve = []
        for t in range(num_phases):
            I_t = image_stack[t][idx]
            normalized = I_t / I0[idx]  # normalization to pre-contrast
            curve.append(normalized)
        tic_curves[idx] = curve
    return tic_curves

# Extract TIC curve features
def compute_tic_features(tic):
    tic = np.array(tic, dtype=np.float32)
    eps = 1e-8
    n_phases = len(tic)  # number of phases

    baseline = tic[0]  # intensity before contrast agent
    peak_phase_index = np.argmax(tic)  # index of maximum enhancement
    peak_value = tic[peak_phase_index]  # peak intensity
    last_value = tic[-1]  # final phase intensity

    # Rate of contrast uptake
    wash_in_rate = (peak_value - baseline) / (peak_phase_index + eps)
    print(baseline)
    print(peak_value)
    print(peak_phase_index)
    print(wash_in_rate)

    # Degree of contrast wash-out
    wash_out_enhancement = (last_value - peak_value) / (peak_value + eps)

    # Stability of wash-out
    if peak_phase_index < n_phases - 1:  # check if peak is not the last phase
        x = np.arange(peak_phase_index, n_phases).reshape(-1, 1)
        y = tic[peak_phase_index:]  # linear trend after peak
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        rss = np.sum((y - y_pred)**2)  # residual sum of squares
        wash_out_stability = rss / ((baseline + eps) * (n_phases - peak_phase_index))  # normalized to pre-contrast
    else:
        wash_out_stability = 0.0

    return {
        "wash_in_rate": wash_in_rate,
        "wash_out_enhancement": wash_out_enhancement,
        "wash_out_stability": wash_out_stability
    }

# Classify TIC pattern into Nottingham type for a single voxel
def classify_nottingham_type(features):
    wash_in = features["wash_in_rate"]
    wash_out = features["wash_out_enhancement"]
    stability = features["wash_out_stability"]

    # Classification of wash-in rate
    if wash_in < 0.1:
        wash_in_type = "nonenhanced"
    elif 0.1 <= wash_in < 0.5:
        wash_in_type = "slow"
    elif 0.5 <= wash_in < 1.0:
        wash_in_type = "medium"
    else:
        wash_in_type = "fast"

    # Classification of wash-out enhancement
    if wash_out > 0.05:
        wash_out_type = "persistent"
        overall_type = "Type I"
    elif -0.05 <= wash_out <= 0.05:
        wash_out_type = "plateau"
        overall_type = "Type II"
    else:  # wash_out < -0.05
        wash_out_type = "decline"
        overall_type = "Type III"

    # Classification of wash-out stability
    if stability < 0.1:
        stability_type = "steady"
    else:
        stability_type = "non-steady"

    subtype_label = f"{wash_in_type}_{wash_out_type}_{stability_type}"

    return overall_type, subtype_label

# Classify all voxels in TIC curves
def classify_all_voxels_nottingham(tic_curves):
    type_counts = {"Type I": 0, "Type II": 0, "Type III": 0}
    total_features = {"wash_in_rate": 0.0, "wash_out_enhancement": 0.0, "wash_out_stability": 0.0}
    voxel_count = 0

    for voxel, tic in tic_curves.items():
        features = compute_tic_features(tic)
        overall_type, _ = classify_nottingham_type(features)
        type_counts[overall_type] += 1
        for k in total_features:
            total_features[k] += features[k]
        voxel_count += 1

    if voxel_count > 0:
        avg_features = {k: v / voxel_count for k, v in total_features.items()}
    else:
        avg_features = {k: 0.0 for k in total_features}

    return type_counts, avg_features, voxel_count


# Loop through all patients
which = "full_breast"
NOTTINGHAM_CSV = os.path.join("/home/rozmaz2565", "nottingham_summary.csv")
SRC_ROOT = "/lustre/pd01/hpc-ljelen-1692966897/mamma_mia"
IMG_DIR = os.path.join(SRC_ROOT, "images")
SEG_EXPERT_DIR = os.path.join(SRC_ROOT, "segmentations", "expert")

# CSV header
csv_header = [
    "patient_id", "type_I_count", "type_II_count", "type_III_count", "dominant_type",
    "avg_wash_in_rate", "avg_wash_out_enhancement", "avg_wash_out_stability", "voxel_count"
]

with open(NOTTINGHAM_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

    for pid in os.listdir(IMG_DIR):
        patient_path = os.path.join(IMG_DIR, pid)
        mask_path=os.path.join(SRC_ROOT,pid)
        if not os.path.isdir(patient_path):
            continue

        pid_lower = pid.lower()

        # Wczytanie wszystkich klatek
        image_array_list = []
        phase_index = 0
        while True:
            fname = f"{pid_lower}_000{phase_index}.nii.gz"
            full_path = os.path.join(patient_path, fname)
            if not os.path.exists(full_path):
                break
            img = nib.load(full_path).get_fdata()
            image_array_list.append(img)
            phase_index += 1

        if len(image_array_list) == 0:
            continue

        # Maska - ekspert
        seg_path  = os.path.join(SEG_EXPERT_DIR, f"{pid_lower}.nii.gz")
        if not os.path.exists(seg_path):
            continue

        seg_np = nib.load(seg_path).get_fdata()
        seg_np = (seg_np > 0).astype(np.uint8)
        if seg_np.sum() == 0:
            continue

        # Stack kanałów
        image_stack = np.stack(image_array_list, axis=0) 
        
        tic_curves = generate_voxelwise_tic_curves(image_stack,seg_np)

        type_counts, avg_features, voxel_count = classify_all_voxels_nottingham(tic_curves)

        count_I = type_counts.get("Type I", 0)
        count_II = type_counts.get("Type II", 0)
        count_III = type_counts.get("Type III", 0)

        dominant_type = max(type_counts, key=type_counts.get)

        writer.writerow([
            pid, count_I, count_II, count_III, dominant_type,
            avg_features["wash_in_rate"],
            avg_features["wash_out_enhancement"],
            avg_features["wash_out_stability"],
            voxel_count
        ])
