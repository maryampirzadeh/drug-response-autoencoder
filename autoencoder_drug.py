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
    _, X_drug, Y, _, _, _ = train_data(data, screening)
    print(f"\ndata loaded successfully!")
    print(f"    drug shape: {X_drug.shape}")
    print("\n TRAINING AUTOENCODER ON DRUG FEATURES ")
    model, latent_vectors = train_autoencoder(
        X=X_drug, 
        epochs=25,
        model_name="drug_autoencoder" 
    )
    sample = X_drug.iloc[0].values  
    sample_tensor = torch.tensor(sample, dtype=torch.float32)
    
    with torch.no_grad():
        recon = model(sample_tensor).numpy()
    sample_flat = sample.flatten()
    recon_flat = recon.flatten()
    print(sample)
    print(recon)
    print(f"\ntrained successfully!")
    print(f"latent vector shape: {latent_vectors.shape}")
    
    print("\noriginal (first 10 features):", sample_flat[:10])
    print("reconstructed (first 10):", recon_flat[:10])


if __name__ == "__main__":
    main()
