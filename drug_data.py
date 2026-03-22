import os
import pandas as pd

#pyhton code file and folder path
PROJECT_ROOT= os.path.dirname(os.path.abspath(__file__))
DRUG_DATA_FOLDER = r"C:\Users\Book\Downloads\drug_data"   #  change if needed

#all files (drug desc and comp are the same) ans SMILES is csv
DRUG_DATASET_FILES = {
    "drug_desc": "drug_desc_raw.tsv",
    "drug_comp": "drug_comp_raw.tsv",
    "drug_finger": "drug_finger_raw.tsv",
    "drug_target": "drug_DT_raw.tsv",
    "drug_names": "drug_names.tsv",
    "drug_smiles": "drug_names_with_smiles.csv",
}


#create path for the file
def get_file_path(base_folder, filename):
    path = os.path.join(base_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f" File not found!!: {path}")
    return path

#separate tsv
def load_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)

#already seperated
def load_csv(path):
    return pd.read_csv(path)

#create list for all the files in the folder
def list_available_files(folder):
    if not os.path.exists(folder):
        
        print(f"Folder does not exist!!: {folder}")
        return []
    #print([f for f in os.listdir(folder) if f.endswith((".tsv", ".csv"))])
    return [f for f in os.listdir(folder) if f.endswith((".tsv", ".csv"))]


#check files and print sizes
def check_drug_dataset_integrity():
    print("checking drug dataset files\n")
    missing =[]

    for modality,filename in DRUG_DATASET_FILES.items():
        path=os.path.join(DRUG_DATA_FOLDER, filename)
        if not os.path.exists(path):
            print(f"missing {modality}:{filename}")
            missing.append(filename)
        else:
            size_mb = os.path.getsize(path)/(1024*1024)
            print(f"found {modality}: {filename} ({size_mb:.2f} MB)")
    if not missing:
        print("\nAll drug data files found!!\n")
    return missing


#preview this file as table
TEST_FILE = "drug_target"   

if __name__ == "__main__":
    print(f"drug data folder: {DRUG_DATA_FOLDER}")
    print(f"drug modalities: {DRUG_DATASET_FILES}\n")
    try:
        missing = check_drug_dataset_integrity()

        if not missing:
            file_to_load=DRUG_DATASET_FILES[TEST_FILE]
            file_path = get_file_path(DRUG_DATA_FOLDER, file_to_load)
            #csv smiles + tsv
            if file_to_load.endswith(".tsv"):
                df = load_tsv(file_path)
            else:
                df = load_csv(file_path)

            print(f"\n preview of {TEST_FILE} ({df.shape[0]} × {df.shape[1]}):")
            #the number of rows can be changed
            print(df.head(5))

    except FileNotFoundError as e:
        print(str(e))
