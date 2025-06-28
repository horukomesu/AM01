import os
import sys
import subprocess
import site

DIST_DIR = os.path.join(os.path.dirname(__file__), "dist")
GET_PIP_LOCAL = os.path.join(DIST_DIR, "get-pip.py")

# Хардкод соответствий ABI -> нужные .whl
WHEEL_FILES = {
    "cp39": {
        "numpy": "numpy-1.26.4-cp39-cp39-win_amd64.whl",
        "pillow": "pillow-11.2.1-cp39-cp39-win_amd64.whl",
        "pycolmap": "pycolmap-3.11.1-cp39-cp39-win_amd64.whl"
    },
    "cp311": {
        "numpy": "numpy-1.26.4-cp311-cp311-win_amd64.whl",
        "pillow": "pillow-11.2.1-cp311-cp311-win_amd64.whl",
        "pycolmap": "pycolmap-3.11.1-cp311-cp311-win_amd64.whl"
    }
}


def ensure_user_site_in_path():
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
        print(f"[INFO] Добавлен в sys.path: {user_site}")


def is_pip_available() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print(f"[INFO] pip найден: {result.stdout.strip()}")
            return True
        else:
            print(f"[INFO] pip не работает: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"[INFO] pip не найден или вызвал ошибку: {e}")
        return False


def install_pip_offline():
    if is_pip_available():
        return True

    print("[INFO] pip не найден. Установка из локального get-pip.py...")

    if not os.path.exists(GET_PIP_LOCAL):
        print(f"[ERROR] get-pip.py не найден по пути: {GET_PIP_LOCAL}")
        return False

    try:
        subprocess.check_call([
            sys.executable,
            GET_PIP_LOCAL,
            "--user"
        ])
        print("[INFO] pip успешно установлен из dist.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка при установке pip из dist: {e}")
        return False


def pip_install_from_whl(whl_filename: str) -> bool:
    ensure_user_site_in_path()
    whl_path = os.path.join(DIST_DIR, whl_filename)
    print(f"[INFO] Установка {whl_filename} ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", whl_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] pip install не удался: {whl_filename}")
        print(f"[stderr]\n{e.stderr}")
        return False


def detect_python_tag() -> str:
    major = sys.version_info.major
    minor = sys.version_info.minor
    tag = f"cp{major}{minor}"
    print(f"[INFO] Определён Python ABI: {tag}")
    return tag


def main():
    print(f"[INFO] Используется Python: {sys.executable}")
    print(f"[INFO] Установка будет из папки: {DIST_DIR}")

    if not os.path.isdir(DIST_DIR):
        print(f"[ERROR] Папка dist/ не найдена по пути: {DIST_DIR}")
        return

    ensure_user_site_in_path()

    tag = detect_python_tag()
    if tag not in WHEEL_FILES:
        print(f"[ERROR] Нет .whl файлов для Python ABI: {tag}")
        return

    if install_pip_offline():
        pip_install_from_whl(WHEEL_FILES[tag]["numpy"])
        pip_install_from_whl(WHEEL_FILES[tag]["pillow"])
        pip_install_from_whl(WHEEL_FILES[tag]["pycolmap"])


if __name__ == "__main__":
    main()
