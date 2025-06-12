import ttkbootstrap as ttk

from interface_rendering import BreastCancerApp

if __name__ == "__main__":

    root = ttk.Window(themename="darkly")

    app = BreastCancerApp(root)

    root.mainloop()

"""

I have a patient that is 38 years old. Her HER2 is 1, ER is 1, and PR is 0. She is premenopausal. 
The tumor subtype is HER2-enriched. She received neoadjuvant therapy with HER2-targeted drugs. 
She's hispanic and does not have breast implants.

"""