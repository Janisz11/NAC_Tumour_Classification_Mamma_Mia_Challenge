import os
import json
import numpy as np
import nibabel as nib
import pandas as pd

SRC_ROOT = "/lustre/pd01/hpc-ljelen-1692966897/mamma_mia"
IMG_DIR = os.path.join(SRC_ROOT, "images")
SEG_EXPERT_DIR = os.path.join(SRC_ROOT, "segmentations", "expert")
CLINICAL_XLSX = os.path.join(SRC_ROOT, "clinical_and_imaging_info.xlsx")

DEST_ROOT = os.path.join(SRC_ROOT, "processed_dataset")
DEST_JSON = os.path.join(SRC_ROOT, "train.json")

CROP_WITH_MARGIN = True
PROCESS_FULL_BREAST = True
MARGIN_SIZE = 10
USE_DIFFS = True

# funkcja do przycinania
def crop_to_bbox(image, mask, margin=10):
    coords = np.array(np.where(mask))
    minc = np.maximum(coords.min(axis=1) - margin, 0)
    maxc = np.minimum(coords.max(axis=1) + margin, mask.shape)
    slices = tuple(slice(minc[i], maxc[i]) for i in range(3))
    return image[:, slices[0], slices[1], slices[2]], mask[slices[0], slices[1], slices[2]]

# funkcja do przyciania do jednej piersi
def crop_breast_containing_mask(image_stack, mask):
    
    # środek guza w osi X (indeks 2)
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return image_stack, mask # Pacjentka bez raka

    center_x = coords[1].mean() 
    
    # całkowity rozmiar obrazu w osi X
    full_x_dim = image_stack.shape[2]
    
    # środek obrazu w osi X
    mid_x = full_x_dim / 2
    
    # Sprawdzam czy guz jest w lewej czy w prawej piersi
    if center_x < mid_x:
        slices_x = slice(0, int(mid_x))
    else:
        slices_x = slice(int(mid_x), full_x_dim)
    
    # Przycinanie dla wszystkich faz
    cropped_image_stack = image_stack[:, :, :, slices_x]
    cropped_mask = mask[:, :, slices_x] # WAŻNE: Maska też musi być przycięta!
    
    return cropped_image_stack, cropped_mask


# Mapowanie pacjent -> pcr
clin_df = pd.read_excel(CLINICAL_XLSX)
pcr_map = dict(zip(clin_df['patient_id'], clin_df['pcr']))

datalist = []

for pid in os.listdir(IMG_DIR):
    patient_path = os.path.join(IMG_DIR, pid)
    if not os.path.isdir(patient_path):
        continue
        
    pid_lower = pid.lower()
    
    # Sprawdzenie czy to duke
    is_duke_patient = "duke" in pid_lower 
    
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
    image_stack_original = np.stack(image_array_list, axis=0) 
    
    # Label PCR
    label = pcr_map.get(pid)
    if label not in [0, 1]:
        continue

     # Tworzenie katalogu docelowego dla pacjenta i zapis etykiety
    out_dir_patient = os.path.join(DEST_ROOT, pid)
    os.makedirs(out_dir_patient, exist_ok=True)
    with open(os.path.join(out_dir_patient, "label.txt"), "w") as f:
        f.write(str(int(label)))
        
    #Do JSON
    data_entry = {
        "patient_id": pid,
        "label": int(label),
    }

    #Dodanie folderu z całymi zdjeciami (ucinam do jednej piersi dla duke)
    if PROCESS_FULL_BREAST:
        full_breast_image_to_save = image_stack_original.copy()
        full_breast_mask_to_save = seg_np.copy()
        
        #Jezeli duke to przycinamy do jednej piersi
        if is_duke_patient:
            full_breast_image_to_save, full_breast_mask_to_save = crop_breast_containing_mask(image_stack_original.copy(), seg_np.copy())
            
            
        # Zapis do podfolderu
        out_dir_full_breast = os.path.join(out_dir_patient, "full_breast")
        os.makedirs(out_dir_full_breast, exist_ok=True)
        np.save(os.path.join(out_dir_full_breast, "images.npy"), full_breast_image_to_save)
        np.save(os.path.join(out_dir_full_breast, "mask.npy"), full_breast_mask_to_save)
        
        # Dodanie info do JSON
        data_entry["full_breast_path"] = os.path.join(pid, "full_breast")
        data_entry["is_duke"] = is_duke_patient
        data_entry["full_breast_num_phases"] = full_breast_image_to_save.shape[0]
        data_entry["full_breast_shape"] = full_breast_image_to_save.shape[1:]
        data_entry["full_breast_mask_shape"] = full_breast_mask_to_save.shape

    # Dodanie podfolderu ze zdjeciami przycietymi do obszaru guza
    if CROP_WITH_MARGIN:
        cropped_image_stack, cropped_seg_np = crop_to_bbox(image_stack_original.copy(), seg_np.copy(),margin=MARGIN_SIZE)
        
        # Różnice między fazami
        if USE_DIFFS and cropped_image_stack.shape[0] > 1:
            diffs_cropped = [cropped_image_stack[i+1] - cropped_image_stack[i] for i in range(cropped_image_stack.shape[0]-1)]
            cropped_image_stack = np.concatenate((cropped_image_stack, np.stack(diffs_cropped, axis=0)), axis=0)

        # Zapis do podfolderu
        out_dir_cropped = os.path.join(out_dir_patient, "cropped")
        os.makedirs(out_dir_cropped, exist_ok=True)
        np.save(os.path.join(out_dir_cropped, "images.npy"), cropped_image_stack)
        np.save(os.path.join(out_dir_cropped, "mask.npy"), cropped_seg_np)
        
        # Dodanie info do JSON
        data_entry["cropped_path"] = os.path.join(pid, "cropped")
        data_entry["cropped_num_phases"] = cropped_image_stack.shape[0]
        data_entry["cropped_shape"] = cropped_image_stack.shape[1:]
        data_entry["cropped_mask_shape"] = cropped_seg_np.shape
        data_entry["cropped_with_diffs"] = USE_DIFFS 

    datalist.append(data_entry)


# Zapis finalnego pliku JSON
with open(DEST_JSON, "w") as f:
    json.dump(datalist, f, indent=2)

