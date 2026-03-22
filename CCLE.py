import os
import pandas as pd

#same code as CTPR+GDSC
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_CCLE_DATA_FOLDER = r"C:\Users\Book\Downloads\CCLE_data"
CCLE_SCREENING_DATA_FOLDER =RAW_CCLE_DATA_FOLDER

DATA_MODALITIES = {
    "table": "drug_screening.tsv",                  
    "matrix": "drug_screening_matrix_ccle.tsv",       
    "mutation": "cell_mut_raw.tsv",
    "copy_number": "cell_CN_raw.tsv",
    "expression": "cell_exp_raw.tsv",                       
}

#last 3 files are the same

def get_file_path(base_folder, filename):

    path = os.path.join(base_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found!: {path}")
    return path

def load_tsv(path):
    return pd.read_csv(path, sep="\t",low_memory=False)



def list_available_files(folder):

    if not os.path.exists(folder):
        print(f"Folder does not exist!: {folder}")
        return []
    return [f for f in os.listdir(folder) if f.endswith(".tsv")]


def check_dataset_integrity():
    print("checking CCLE dataset files...\n")
    missing =[]
    for modality, filename in DATA_MODALITIES.items():
        path = os.path.join(RAW_CCLE_DATA_FOLDER, filename)
        if not os.path.exists(path):
            print(f"Missing {modality}: {filename}!")
            missing.append(filename)
        else:
            size_mb = os.path.getsize(path) /(1024 * 1024)
            print(f"Found {modality}: {filename} ({size_mb:.2f} MB)")
    if not missing:
        print("\n All CCLE files found successfully!\n")
    return missing


test_file= "table"
if __name__ == "__main__":

    print(f"!!! Data folder: {RAW_CCLE_DATA_FOLDER}")
    print(f"Modalities: {DATA_MODALITIES}\n")

    try:
        missing = check_dataset_integrity()
        if not missing:
            exp_path = get_file_path(RAW_CCLE_DATA_FOLDER, DATA_MODALITIES[test_file])
            df = load_tsv(exp_path)
            print(f"{test_file} data as table ({df.shape[0]} × {df.shape[1]}):")
            print(df.head(5))
    except FileNotFoundError as e:
        print(str(e))
