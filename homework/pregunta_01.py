import shutil
import zipfile
import pandas as pd
from pathlib import Path

# Paths
INPUT_ZIP = Path("files/input.zip")
EXTRACT_DIR = Path("files/input")
OUTPUT_DIR = Path("files/output")

def extract_input_zip(zip_path: Path = INPUT_ZIP, extract_to: Path = EXTRACT_DIR) -> None:
    """
    Extrae el ZIP de entrada en una carpeta limpia,
    eliminando antes cualquier extracción previa.
    """
    if extract_to.exists():
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

def load_dataset(extract_dir: Path, subset: str) -> pd.DataFrame:
    """
    Recorre la carpeta extraída y carga los archivos .txt
    cuyo path contenga el nombre del subset ('train' o 'test'),
    devolviendo un DataFrame con columnas 'phrase' y 'target'.
    """
    data = {"phrase": [], "target": []}
    for txt_file in extract_dir.rglob("*.txt"):
        # Solo archivo si pertenece al subset correcto
        if subset in txt_file.parts:
            text = txt_file.read_text(encoding="utf-8").strip()
            label = txt_file.parent.name
            data["phrase"].append(text)
            data["target"].append(label)
    return pd.DataFrame(data)

def pregunta_01():
    """
    Orquesta la extracción del ZIP, la carga de los datasets
    de entrenamiento y prueba, y guarda ambos como CSV.
    """
    # 1) Extraer ZIP de entrada
    extract_input_zip()

    # 2) Cargar DataFrames de train y test
    train_df = load_dataset(EXTRACT_DIR, subset="train")
    test_df  = load_dataset(EXTRACT_DIR, subset="test")

    # 3) Crear carpeta de salida y guardar CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train_dataset.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_dataset.csv",  index=False)
