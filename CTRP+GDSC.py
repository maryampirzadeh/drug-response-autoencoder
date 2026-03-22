import os
import pandas as pd

#path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# find dataset (folder)
RAW_BOTH_DATA_FOLDER = r"C:\Users\Book\Downloads\CTRP_GDSC_data"
BOTH_SCREENING_DATA_FOLDER = RAW_BOTH_DATA_FOLDER

#file names dict
DATA_MODALITIES= {
    "table": "drug_screening_table_both.tsv",
    "matrix": "drug_screening_matrix_gdsc_ctrp.tsv",
    "both_matrix":"drug_screening_matrix_both.tsv",
    "mutation": "cell_mut_raw.tsv",
    "copy_number": "cell_CN_raw.tsv",
    "expression": "cell_exp_raw.tsv",  # largest file
}

#check!
def get_file_path(base_folder,filename):
    path = os.path.join(base_folder,filename)
    if not os.path.exists(path):
        raise FileNotFoundError (f"File not found!: {path}")
    return path

#reformat tsv to show 
def load_tsv(path):
    return pd.read_csv(path, sep="\t",low_memory=False)

#check if folder is loaded

def list_available_files(folder):

    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return []
    return [f for f in os.listdir(folder) if f.endswith(".tsv")]

#check if all files are loaded
def check_dataset_integrity():
    print("Checking CTRP+GDSC dataset files...\n")
    missing =[]
    
    for modality, filename in DATA_MODALITIES.items():
        path = os.path.join(RAW_BOTH_DATA_FOLDER, filename)
        if not os.path.exists(path):
            print(f"Missing {modality}: {filename}!")
            missing.append(filename)
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Found {modality}: {filename} ({size_mb:.2f} MB)")
    if not missing:
        print("\nAll files found!\n")
    return missing



test_file = "expression"
#if __name__ == "__main__":
if __name__ == "__main__":
    print(f"!!! Data folder: {RAW_BOTH_DATA_FOLDER}")
    print(f"Modalities: {DATA_MODALITIES}\n")

    try:
        #error if not loaded
        missing = check_dataset_integrity()
        if not missing:
            # Show quick preview of mutation data
            mut_path = get_file_path(RAW_BOTH_DATA_FOLDER,DATA_MODALITIES[test_file])
            df = load_tsv(mut_path)
            #show dataset as table
            print(f"{test_file} data as table ({df.shape[0]} × {df.shape[1]}):")
            #examples
            print(df.head(5))
    except FileNotFoundError as e:
        print(str(e))

#test_file = "expression"