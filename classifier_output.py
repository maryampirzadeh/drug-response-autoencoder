from final_data import load_all, train_data
from final_autoencoder import train_autoencoder
from final_classifier import train_classifier, result_classifier
import numpy as np
from sklearn.model_selection import train_test_split
from metrics import plot_auc_auprc
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
    # LOADING DATASET
    print("LOADING DATASET")
    data, screening = load_all(
        data_modalities=DATA_MODALITIES,
        cell=RAW_BOTH_DATA_FOLDER,
        screen=SCREEN_FILE,
        drug=DRUG_DATA_FOLDER,
    )

    main_cell, main_drug, label, cell_sizes, drug_sizes, complete_table = train_data(
        data, screening
    )

    print("\nDATA (BEFORE AUTOENCODER)")
    print(f"Main cell features shape: {main_cell.shape}")
    print(f"Main drug features shape: {main_drug.shape}")
    print(f"Labels length: {len(label)}")
    print("="*60)
    # CELL AUTOENCODER
    print("TRAINING CELL AUTOENCODER")
    cell_model, cell_latent = train_autoencoder(
        X=main_cell,
        epochs=25,
        batch_size=64,
        lr=0.001,
        model_name="cell_autoencoder"
    )
    print(f"Cell latent shape: {cell_latent.shape}")

    # DRUG AUTOENCODER
    print("="*60)
    print("TRAINING DRUG AUTOENCODER")
    drug_model, drug_latent = train_autoencoder(
        X=main_drug,
        epochs=25,
        batch_size=64,
        lr=0.001,
        model_name="drug_autoencoder"
    )
    print(f"Drug latent shape: {drug_latent.shape}")

    # CONCAT LATENT SPACE
    print("CONCAT LATENT SPACES")
    X_latent = np.concatenate([cell_latent, drug_latent], axis=1)
    print(f"final latent space shape: {X_latent.shape}")
    print(f"cell latent dim: {cell_latent.shape[1]}")
    print(f"drug latent dim: {drug_latent.shape[1]}")
    print(f"total latent dim: {X_latent.shape[1]}")


    #CREATE 70% TRAIN / 30% TEMP SPLIT
    # we can't separate data in 3 parts directly. only two parts is possible 
    #we do it in two steps. 1.train and val+test 2.val and test
    print("\n"+"="*60)
    print("CREATING TRAIN/VAL/TEST SPLITS (70/10/20)")
    #first split: 70% train and 30% val+test
    #X is data y are labels, train is the 70%, temp is 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_latent,   #input
        label,   #output
        test_size=0.3,  #30% for val+test
        random_state=42, #random seed
        shuffle=True, #cause label is [0,0,0,...,1,1,1]
        stratify=label  #same ratio
    )
    
    # second split:temp into 10% val and 20% test
    # 1/3 = 0.3333, 2/3 = 0.6667
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.6667,  
        random_state=42,
        shuffle=True,
        stratify=y_temp
    )
    
    print(f"\nFinal dataset splits:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Calculate percentages
    total_samples = len(label)

    
    print(f"\nratio:")
    print(f"Training: {np.bincount(y_train.astype(int))}")
    print(f"Validation: {np.bincount(y_val.astype(int))}")
    print(f"Test: {np.bincount(y_test.astype(int))}")


    #TRAIN MLP CLASSIFIER
    print("\n" + "="*60)
    print("7. TRAINING MLP CLASSIFIER")
    print("="*60)
    print("\nTraining MLP classifier on combined latent space...\n")
    
    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,      
        y_val=y_val,
        num_epochs=25,
        batch_size=64,
        lr=0.01,
    )

    # CLASSIFIER RESULTS WITH TEST DATA
    print("\n" + "="*60)
    print("MODEL ON TEST SET")
    #this func prints the metrics too
    results = result_classifier(model, X_test, y_test)

    #for plot
    import torch
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        predicted = outputs.numpy()
    plot_auc_auprc(y_test, predicted)



if __name__ == "__main__":
    main()
