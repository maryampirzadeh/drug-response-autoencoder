from final_data import load_all, train_data
from final_autoencoder import train_autoencoder
import torch

DATA_MODALITIES = [
    "cell_CN", "cell_exp", "cell_methy", "cell_mut",
    "drug_DT", "drug_comp", "drug_desc", "drug_finger"
]
RAW_BOTH_DATA_FOLDER = r"C:\Users\Book\Downloads\CTRP_GDSC_data"
DRUG_DATA_FOLDER     = r"C:\Users\Book\Downloads\drug_data"
SCREEN_FILE          = r"C:\Users\Book\Downloads\CTRP_GDSC_data\drug_screening_matrix_gdsc_ctrp.tsv"

def main():
    print("\nloading dataset...")
    data, screening = load_all(
        data_modalities=DATA_MODALITIES,
        cell=RAW_BOTH_DATA_FOLDER,
        screen=SCREEN_FILE,
        drug=DRUG_DATA_FOLDER
    )
    X_cell, _, Y, _, _, _ = train_data(data, screening)

    print("\ndata loaded successfully!")
    print(f"   cell shape: {X_cell.shape}")
    print(" TRAINING AUTOENCODER ON CELL FEATURES ")

    model, latent_vectors = train_autoencoder(
        X=X_cell,
        epochs=25,
        model_name="cell_autoencoder"
    )

    sample = X_cell.iloc[0].values
    sample_tensor = torch.tensor(sample, dtype=torch.float32)

    with torch.no_grad():
        recon = model(sample_tensor).numpy()
    sample_flat = sample.flatten()
    recon_flat = recon.flatten()

    print("\ntrained successfully!")
    print(f"latent vector shape: {latent_vectors.shape}")

    print("\noriginal (first 10 features):", sample_flat[:10])
    print("reconstructed (first 10):", recon_flat[:10])

if __name__ == "__main__":
    main()

