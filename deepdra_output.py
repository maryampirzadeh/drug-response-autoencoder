# deepdra_output.py
#
# Joint training of DeepDRA (cell AE + drug AE + MLP)
# using your own final_deepdra, final_data, evaluation, etc.

import numpy as np
from sklearn.model_selection import train_test_split

from final_data import load_all, train_data
from final_deepdra import train_deepdra, test_deepdra

# same config style as your other files
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
    # ==============================
    # 1. LOAD AND MATCH DATA
    # ==============================
    print("\nloading dataset...")
    data, screening = load_all(
        data_modalities=DATA_MODALITIES,
        cell=RAW_BOTH_DATA_FOLDER,
        screen=SCREEN_FILE,
        drug=DRUG_DATA_FOLDER,
    )

    print("building matched cell + drug + labels...")
    main_cell, main_drug, label, cell_sizes, drug_sizes, complete_table = train_data(
        data, screening
    )

    print("\n=== RAW INPUT FOR DEEPDRA ===")
    print("main_cell shape:", main_cell.shape)
    print("main_drug shape:", main_drug.shape)
    print("labels length:", len(label))

    # convert labels to numpy
    y = np.array(label, dtype=float)

    # ==============================
    # 2. TRAIN / VAL / TEST SPLIT
    # ==============================
    # First: train+val vs test
    (
        X_cell_trainval,
        X_cell_test,
        X_drug_trainval,
        X_drug_test,
        y_trainval,
        y_test,
    ) = train_test_split(
        main_cell,
        main_drug,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # Second: train vs val (from trainval)
    (
        X_cell_train,
        X_cell_val,
        X_drug_train,
        X_drug_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_cell_trainval,
        X_drug_trainval,
        y_trainval,
        test_size=0.2,          # 20% of 80% -> 16% total as val
        random_state=42,
        shuffle=True,
    )

    print("\n=== SPLIT SHAPES ===")
    print("X_cell_train shape:", X_cell_train.shape)
    print("X_drug_train shape:", X_drug_train.shape)
    print("y_train len:", len(y_train))

    print("X_cell_val shape:", X_cell_val.shape)
    print("X_drug_val shape:", X_drug_val.shape)
    print("y_val len:", len(y_val))

    print("X_cell_test shape:", X_cell_test.shape)
    print("X_drug_test shape:", X_drug_test.shape)
    print("y_test len:", len(y_test))

    # ==============================
    # 3. TRAIN DEEPDRA (JOINT)
    # ==============================
    print("\nTRAINING DeepDRA (joint AE + MLP)...\n")

    model, history = train_deepdra(
        X_cell_train=X_cell_train,
        X_drug_train=X_drug_train,
        y_train=y_train,
        X_cell_val=X_cell_val,
        X_drug_val=X_drug_val,
        y_val=y_val,
        num_epochs=50,          # you can tune this
        batch_size=64,
        lr=0.0005,              # joint training is usually more stable with smaller lr
        cell_ae_weight=1.0,
        drug_ae_weight=1.0,
        mlp_weight=1.0,
    )

    # ==============================
    # 4. TEST DEEPDRA
    # ==============================
    print("\nEVALUATING DeepDRA ON TEST SET...")
    results = test_deepdra(
        model=model,
        X_cell_test=X_cell_test,
        X_drug_test=X_drug_test,
        y_test=y_test,
    )

    print("\nFINAL TEST METRICS (FROM evaluation.evaluate):")
    print(results)
    print("\nDONE (DeepDRA joint training + test)\n")


if __name__ == "__main__":
    main()
