import ttkbootstrap as ttk

from InterfaceRendering import BreastCancerApp

if __name__ == "__main__":

    root = ttk.Window(themename="darkly")

    app = BreastCancerApp(root)

    root.mainloop()
