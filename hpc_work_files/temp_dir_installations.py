import sys
import subprocess
import os
import tempfile
import pkg_resources

def install_packages_tmp(packages):
    tmp_dir = os.environ.get('TMPDIR') or tempfile.gettempdir()
    install_dir = os.path.join(tmp_dir, 'python_tmp_lib')

    os.makedirs(install_dir, exist_ok=True)
    print(f"Instaluję pakiety {packages} w katalogu: {install_dir}")


    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    packages_to_install = [pkg for pkg in packages if pkg.lower() not in installed_packages]

    if not packages_to_install:
        print("Wszystkie pakiety są już zainstalowane.")
    else:

        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir",
            "--target", install_dir
        ] + packages_to_install)


    if install_dir not in sys.path:
        sys.path.insert(0, install_dir)

    print("Instalacja zakończona.")


install_packages_tmp(['nibabel', 'pandas', 'openpyxl', 'matplotlib', 'fsspec','scikit-learn'])
