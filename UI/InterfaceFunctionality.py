from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk
import os
import warnings
from tkinter import messagebox, simpledialog
import Values
import pandas as pd


def load_data_file(self, file_path=None):

    if file_path is None:
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii.gz *.nii")])
        if not file_path:
            return

    img = nib.load(file_path)
    self.loaded_image_data = img.get_fdata()  # zapisujemy dane
    shape = self.loaded_image_data.shape
    spacing = img.header.get_zooms()
    filename = file_path.split("/")[-1]

    # Info o pliku
    for widget in self.info_frame.winfo_children():
        widget.destroy()

    ttk.Label(self.info_frame, text=f"Nazwa pliku: {filename}").pack(anchor="w", padx=10)
    ttk.Label(self.info_frame, text=f"Wymiary obrazu: {shape}").pack(anchor="w", padx=10)
    ttk.Label(self.info_frame, text=f"Spacing: {spacing}").pack(anchor="w", padx=10)

    # Obraz
    for widget in self.image_display_frame.winfo_children():
        widget.destroy()

    self.current_slice = shape[2] // 2  # startowy slice

    # Miejsce na figurę matplotliba
    self.fig, self.ax = plt.subplots(figsize=(3, 3))
    self.image_obj = self.ax.imshow(self.loaded_image_data[:, :, self.current_slice].T, cmap="gray", origin="lower")
    self.ax.axis("off")

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_display_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack()

    # Aktualizujemy zakres suwaka
    self.slider.config(to=shape[2]-1)
    self.slider_value.set(self.current_slice)
    self.slider_value_showcase.config(text=str(self.current_slice))

    self.data_beginning_label.forget()


def load_model_file(self, file_path=None):
    if file_path is None:
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if not file_path:
            return

    self.training_beginning_label.forget()


def load_folder(self, extension, load_first_file=False):

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

        match extension:

            case ".nii.gz":

                self.current_file_group_index = 0
                self.current_data_file_group = files

                self.selector_label.config(text=f"1 / {len(files)}")

                if load_first_file:
                    load_data_file(self, files[0])

            case ".pkl":

                self.current_model_file_group = files

                if load_first_file:
                    load_model_file(self, files[0])

            case _:
                warnings.warn("Nieoczekiwane rozszerzenie")

        return files

    else:
        messagebox.showinfo("Warning", "No files with specified extension found in the folder")


def browse_patient_data(self):
    patient_id = simpledialog.askstring("ID Pacjenta", "Podaj ID pacjenta, którego dane chcesz przeglądać:")
    load_patient_data(self, patient_id)


def load_patient_data(self, patient_id):
    df = pd.read_excel(Values.clinical_and_imaging_info_path, dtype=str).fillna('')

    df.columns = df.columns.str.strip()
    row = df[df['patient_id'] == patient_id]

    if row.empty:
        ttk.Label(self.info_frame, text="Pacjent nie znaleziony").pack(anchor="w", padx=10)
        return

    binary_cols = {
        'bilateral_breast_cancer', 'multifocal_cancer', 'endocrine_therapy', 'anti_her2_neu_therapy', 'pcr',
        'mastectomy_post_nac', 'hr', 'er', 'pr', 'her2', 'has_implant', 'bilateral_mri', 'fat_suppressed'
    }

    row_data = row.iloc[0]

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

        ttk.Label(self.info_frame, text=f"{col_clean}: {display_val}").pack(anchor="w", padx=10)

    patient_images_folder_path = str(Values.images_folder_path) + patient_id
    if patient_images_folder_path in load_folder(self, ".nii.gz", False):
        load_data_file(self, patient_images_folder_path)


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
