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
        self.assistant_messages = []

    from interface_functionality import (
        load_data_file,
        load_folder,
        update_slider_value,
        next_image,
        previous_image,
        browse_patient_data,
        send_request
    )

    def init_ui(self):

        # --// Top frame with section buttons \\--

        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x", pady=10)

        ttk.Button(top_frame, text="Talk to an AI assistant", command=self.show_assistant_section).pack(side="right", padx=10)
        self.ai_assistant_close_button = ttk.Button(top_frame, text="Close", command=self.show_dataset_section)

        # --// Dataset section \\--

        self.dataset_frame = ttk.Frame(self.root)

        self.load_data_buttons_frame = ttk.Frame(self.dataset_frame)
        self.load_data_buttons_frame.pack(pady=10)

        self.load_data_file_button = ttk.Button(self.load_data_buttons_frame, text="Load .nii.gz file", command=self.load_data_file)
        self.load_data_folder_button = ttk.Button(self.load_data_buttons_frame, text="Load data folder",
                                                command=lambda: self.load_folder(".nii.gz"))

        self.browse_patient_data_button = ttk.Button(self.load_data_buttons_frame, text="Browse patient data",
                                                command=self.browse_patient_data)

        self.data_beginning_label = ttk.Label(self.dataset_frame, text="Load in data to begin",
                                        font=("Segoe UI", 16))
        self.data_beginning_label.pack(pady=20)

        self.load_data_file_button.pack(side="left", padx=5)
        self.load_data_folder_button.pack(side="left", padx=5)
        self.browse_patient_data_button.pack(side="left", padx=5)

        info_container = ttk.LabelFrame(self.dataset_frame, text="DATA")
        info_container.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        canvas = ttk.Canvas(info_container)
        scrollbar = ttk.Scrollbar(info_container, orient="vertical", command=canvas.yview)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.info_frame = ttk.Frame(canvas)
        self.info_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.info_frame, anchor="nw")

        self.image_frame = ttk.LabelFrame(self.dataset_frame, text="IMAGE - CROSS-SECTION", width=500, height=500)
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

        # --// AI Assistant section \\--

        self.assistant_frame = ttk.Frame(self.root)

        self.chat_display = ttk.ScrolledText(self.assistant_frame, height=20, wrap="word", font=("Segoe UI", 10))
        self.chat_display.pack(padx=10, pady=10, fill="both", expand=True)

        bottom_frame = ttk.Frame(self.assistant_frame)
        bottom_frame.pack(fill="x", padx=10)

        self.user_input = ttk.Entry(bottom_frame)
        self.user_input.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(bottom_frame, text="Send", command=self.send_request).pack(side="right")

        self.show_dataset_section()

    def show_dataset_section(self):
        self.assistant_frame.forget()
        self.dataset_frame.pack(fill="both", expand=True)
        self.ai_assistant_close_button.forget()

    def show_assistant_section(self):
        self.dataset_frame.forget()
        self.assistant_frame.pack(fill="both", expand=True, padx=15, pady=15)
        self.ai_assistant_close_button.pack(side="right", padx=0)