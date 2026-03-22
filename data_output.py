from final_data import load_all, train_data

DATA_MODALITIES = [
    "cell_CN",
    "cell_exp",
    "cell_methy",
    "cell_mut",
    "drug_DT",
    "drug_comp",
    "drug_desc",
    "drug_finger",
]

RAW_BOTH_DATA_FOLDER = r"C:\Users\Book\Downloads\CTRP_GDSC_data"
DRUG_DATA_FOLDER     = r"C:\Users\Book\Downloads\drug_data"
SCREEN_FILE          = r"C:\Users\Book\Downloads\CTRP_GDSC_data\drug_screening_matrix_gdsc_ctrp.tsv"


def main():
    print("loading raw dataset")
    data, screening = load_all(
        DATA_MODALITIES,
        RAW_BOTH_DATA_FOLDER,
        SCREEN_FILE,
        DRUG_DATA_FOLDER,
    )

    print("loaded modalities:")
    for key in data:
        print(f"  {key} : {data[key].shape}")

    print("screening shape:", screening.shape)
    print("train data results:")

    main_cell, main_drug, label, cell_sizes, drug_sizes, complete_table = train_data(data, screening)

    print("\n FINAL OUTPUT")


    print("main_cell shape:", main_cell.shape)
    print("main_drug shape:", main_drug.shape)
    print("label len: (",len(label),")")
    print("main table shape:", complete_table.shape)

    print("\nmain cell sizes:", cell_sizes)
    print("main drug sizes:", drug_sizes)

    print("\nfirst 10 main_cell rows:")
    print(main_cell.head(10))

    print("\nfirst 10 main_drug rows:")
    print(main_drug.head(10))

    print("\nfirst 20 labels:")
    print(label[:20])
    
    print("\nfirst 10 in main table:")
    print(complete_table.head(10))
    print("\nDONE\n")


if __name__ == "__main__":
    main()
