import ttkbootstrap as ttk
from ttkbootstrap.constants import *


class BreastCancerApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Breast Cancer AI Interface")
        self.root.geometry("1000x700")

        self.init_ui()

        self.current_file = None
        self.current_file_group = None
        self.current_file_group_index = 0

    from InterfaceFunctionality import (
        load_data_file,
        load_model_file,
        load_folder,
        update_slider_value,
        next_image,
        previous_image,
        browse_patient_data
    )

    def init_ui(self):

        # --// Top frame with section buttons \\--

        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=10)

        ttk.Button(top_frame, text="Dataset", command=self.show_dataset_section).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Training", command=self.show_training_section).pack(side="left", padx=5)

        # --// Dataset section \\--

        self.dataset_frame = ttk.Frame(self.root)

        self.load_data_buttons_frame = ttk.Frame(self.dataset_frame)
        self.load_data_buttons_frame.pack(pady=10)

        self.load_data_file_button = ttk.Button(self.load_data_buttons_frame, text="Load .nii.gz file", command=self.load_data_file)
        self.load_data_folder_button = ttk.Button(self.load_data_buttons_frame, text="Load data folder",
                                                command=lambda: self.load_folder(".nii.gz"))

        self.browse_patient_data_button = ttk.Button(self.load_data_buttons_frame, text="Browse patient data",
                                                command=self.browse_patient_data)

        self.data_beginning_label = ttk.Label(self.dataset_frame, text="Wczytaj dane, aby rozpocząć",
                                        font=("Segoe UI", 16))
        self.data_beginning_label.pack(pady=20)

        self.load_data_file_button.pack(side="left", padx=5)
        self.load_data_folder_button.pack(side="left", padx=5)
        self.browse_patient_data_button.pack(side="left", padx=5)

        self.info_frame = ttk.LabelFrame(self.dataset_frame, text="DANE")
        self.info_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.image_frame = ttk.LabelFrame(self.dataset_frame, text="ZDJĘCIE - PRZEKRÓJ", width=500, height=500)
        self.image_frame.pack(side="right", padx=10, pady=10, fill="both")
        self.image_frame.pack_propagate(False)

        self.image_display_frame = ttk.Frame(self.image_frame)
        self.image_display_frame.pack(side="top", fill="both", expand=True, pady=35)
        self.image_frame.pack_propagate(False)

        self.slider_frame = ttk.Frame(self.image_frame)
        self.slider_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.slider_value = ttk.IntVar(value=0)
        self.slider = ttk.Scale(self.slider_frame, variable=self.slider_value, from_=1, to=100, orient=HORIZONTAL,
                                command=self.update_slider_value)

        self.slider.pack(side="left", fill="x", expand=True, padx=10)

        self.slider_value_showcase = ttk.Label(self.slider_frame, text="0")
        self.slider_value_showcase.pack(side="right", padx=10)

        self.selector_frame = ttk.Frame(self.image_frame)
        self.selector_frame.pack(anchor='center', padx=10, pady=10)

        self.selector_left_button = ttk.Button(self.selector_frame, text='<<', command=self.previous_image)
        self.selector_left_button.pack(side="left", padx=5)

        self.selector_label = ttk.Label(self.selector_frame, text="0 / 0")
        self.selector_label.pack(side="left", padx=5)

        self.selector_right_button = ttk.Button(self.selector_frame, text='>>', command=self.next_image)
        self.selector_right_button.pack(side="left", padx=5)


        # --// Training section \\--

        self.training_frame = ttk.Frame(self.root)

        self.load_training_buttons_frame = ttk.Frame(self.training_frame)
        self.load_training_buttons_frame.pack(pady=10)

        self.load_model_file_button = ttk.Button(self.load_training_buttons_frame, text="Load .pkl file",
                                                 command=self.load_model_file)
        # self.load_model_folder_button = ttk.Button(self.load_training_buttons_frame, text="Load model folder",
        #                                           command= lambda: self.load_folder(".pkl"))

        self.load_model_file_button.pack(side="left", padx=5)
        # self.load_model_folder_button.pack(side="left", padx=5)

        self.training_beginning_label = ttk.Label(self.training_frame, text="Wczytaj model, aby rozpocząć", font=("Segoe UI", 16))
        self.training_beginning_label.pack(pady=20)

        self.selected_model_frmae = ttk.Frame(self.training_frame)
        self.training_beginning_label = ttk.Label(self.selected_model_frmae, text="nazwa_modelu", font=("Segoe UI", 16))


        self.show_dataset_section()

    def show_dataset_section(self):
        self.training_frame.forget()
        self.dataset_frame.pack(fill="both", expand=True)

    def show_training_section(self):
        self.dataset_frame.forget()
        self.training_frame.pack(fill="both", expand=True)



