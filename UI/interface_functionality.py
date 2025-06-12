from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk
import os
from tkinter import messagebox, simpledialog
import constants
import pandas as pd
from pathlib import Path
import requests
from tkinter import END

API_URL = "http://127.0.0.1:8000/predict_pcr"


def load_data_file(self, file_path=None):

    if file_path is None:
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii.gz *.nii")])
        if not file_path:
            return

        for widget in self.info_frame.winfo_children():
            widget.destroy()

    img = nib.load(file_path)
    self.loaded_image_data = img.get_fdata()
    shape = self.loaded_image_data.shape

    # Obraz
    for widget in self.image_display_frame.winfo_children():
        widget.destroy()

    self.current_slice = shape[2] // 2

    self.fig, self.ax = plt.subplots(figsize=(3, 3))
    self.image_obj = self.ax.imshow(self.loaded_image_data[:, :, self.current_slice].T, cmap="gray", origin="lower")
    self.ax.axis("off")

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_display_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack()

    self.slider.config(to=shape[2]-1)
    self.slider_value.set(self.current_slice)
    self.slider_value_showcase.config(text=str(self.current_slice))

    self.data_beginning_label.forget()


def load_folder(self, extension, folder_path=None):

    if folder_path is None:
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

    folder = os.fsencode(folder_path)
    files = []

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(extension):
            full_path = os.path.join(folder_path, filename)
            files.append(full_path)

    if len(files) > 0:

        self.current_file_group_index = 0
        self.current_data_file_group = files

        self.selector_label.config(text=f"1 / {len(files)}")

        load_data_file(self, files[0])

        return files

    else:
        messagebox.showinfo("Warning", "No files with specified extension found in the folder")


def browse_patient_data(self):
    patient_id = simpledialog.askstring("Patient ID", "Input ID of the patient whose data you want to browse.")
    load_patient_data(self, patient_id)


def load_patient_data(self, patient_id):
    df = pd.read_excel(constants.clinical_and_imaging_info_path, dtype=str).fillna('')

    df.columns = df.columns.str.strip()
    row = df[df['patient_id'] == patient_id]

    if row.empty:
        messagebox.showinfo("Patient not found" ,"Patient not found")
        return

    binary_cols = {
        'bilateral_breast_cancer', 'multifocal_cancer', 'endocrine_therapy', 'anti_her2_neu_therapy', 'pcr',
        'mastectomy_post_nac', 'hr', 'er', 'pr', 'her2', 'has_implant', 'bilateral_mri', 'fat_suppressed'
    }

    row_data = row.iloc[0]

    for widget in self.info_frame.winfo_children():
        widget.destroy()

    for col, val in row_data.items():
        col_clean = col.strip()

        if col_clean in binary_cols:
            if val == '1':
                display_val = True
            elif val == '0':
                display_val = False
            else:
                display_val = ''
        else:
            display_val = val

        ttk.Label(self.info_frame, text=f"{col_clean}: {display_val}", wraplength=300).pack(anchor="w", padx=10)

    patient_images_folder_path = str(constants.images_folder_path) + "\\" + patient_id
    images_folder_children = [str(p) for p in Path(constants.images_folder_path).iterdir()]
    if patient_images_folder_path in images_folder_children:
        load_folder(self, ".nii.gz", patient_images_folder_path)

    self.data_beginning_label.forget()


def update_slider_value(self, val):
    int_value = int(float(val))

    if not hasattr(self, "loaded_image_data"):
        return

    self.image_obj.set_data(self.loaded_image_data[:, :, int_value].T)
    self.canvas.draw()

    self.slider_value.set(int_value)
    self.slider_value_showcase.config(text=f"{int_value}")


def next_image(self):
    if self.current_file_group_index + 1 < len(self.current_data_file_group):
        self.current_file_group_index += 1
        load_data_file(self, self.current_data_file_group[self.current_file_group_index])
        self.selector_label.config(text=f"{self.current_file_group_index+1} / {len(self.current_data_file_group)}")


def previous_image(self):
    if self.current_file_group_index - 1 >= 0:
        self.current_file_group_index -= 1
        load_data_file(self, self.current_data_file_group[self.current_file_group_index])
        self.selector_label.config(text=f"{self.current_file_group_index + 1} / {len(self.current_data_file_group)}")


def send_request(self):
    user_text = self.user_input.get().strip()
    if not user_text:
        return

    self.chat_display.insert("end", f"TY: {user_text}\n\n")

    self.user_input.delete(0, END)

    payload = {"raw_text": user_text}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        prediction = data.get("prediction", "N/A")
        extracted = data.get("extracted_data", {})
        missing = data.get("warning", {}).get("missing_fields", [])

        result = f"Prediction (pCR): {prediction}\n\nExtracted data:\n"
        for k, v in extracted.items():
            result += f"{k}: {v}\n"

        if missing:
            result += f"\n Missing fields: {', '.join(missing)}"

        self.chat_display.insert("end", f"ASYSTENT: {result}\n\n")

    except Exception as e:
        messagebox.showerror("Error", f"Request failed:\n{str(e)}")
