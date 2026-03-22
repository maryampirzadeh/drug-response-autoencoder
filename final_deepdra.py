# final_deepdra.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from final_autoencoder import create_autoencoder
from final_classifier import create_mlp
from evaluation import Evaluation


# ============================================================
# DeepDRA-style joint model (cell AE + drug AE + MLP)
# ============================================================

class DeepDRA:
    """
    Joint model:
      - Autoencoder for cell features
      - Autoencoder for drug features
      - MLP on concatenated latent space

    Style matches your Autoencoder & MLP (no nn.Module inheritance).
    """

    def __init__(self, cell_input_dim, drug_input_dim,
                 cell_latent_dim=50, drug_latent_dim=50,
                 hidden_dim=128, output_dim=1):

        # autoencoders use your fixed architecture: input -> 256 -> 50 -> 256 -> input
        # so latent_dim is effectively 50 already; the args here are just for clarity.
        self.cell_ae = create_autoencoder(cell_input_dim)
        self.drug_ae = create_autoencoder(drug_input_dim)

        # MLP on concatenated latent space: (cell_latent_dim + drug_latent_dim) -> 128 -> 1
        mlp_input_dim = cell_latent_dim + drug_latent_dim
        self.mlp = create_mlp(mlp_input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        # collect all parameters into a single list for optimizer
        self._params = (
            self.cell_ae.parameters()
            + self.drug_ae.parameters()
            + self.mlp.parameters()
        )

    def parameters(self):
        return self._params

    def __call__(self, cell_x, drug_x):
        """
        cell_x: [batch, cell_input_dim]
        drug_x: [batch, drug_input_dim]
        returns:
          - cell_recon: [batch, cell_input_dim]
          - drug_recon: [batch, drug_input_dim]
          - mlp_output: [batch, 1] probabilities
        """
        # encode both modalities
        z_cell = self.cell_ae.encode(cell_x)      # [B, 50]
        z_drug = self.drug_ae.encode(drug_x)      # [B, 50]

        # reconstruct (decode) using AE call
        cell_recon = self.cell_ae(cell_x)         # [B, cell_dim]
        drug_recon = self.drug_ae(drug_x)         # [B, drug_dim]

        # concat latent spaces
        z = torch.cat([z_cell, z_drug], dim=1)    # [B, 100]

        # MLP prediction
        mlp_output = self.mlp(z)                  # [B, 1] with sigmoid

        return cell_recon, drug_recon, mlp_output


# ============================================================
# TRAINING FUNCTION FOR DEEPDRA (joint AE + MLP)
# ============================================================

def train_deepdra(
    X_cell_train,
    X_drug_train,
    y_train,
    X_cell_val,
    X_drug_val,
    y_val,
    num_epochs=50,
    batch_size=64,
    lr=0.0005,
    cell_ae_weight=1.0,
    drug_ae_weight=1.0,
    mlp_weight=1.0,
):
    """
    Joint training:
      L_total = cell_ae_weight * MSE(cell_recon, cell_input)
              + drug_ae_weight * MSE(drug_recon, drug_input)
              + mlp_weight * BCE(mlp_output, labels)

    Inputs can be numpy arrays or pandas DataFrames.
    """

    # convert pandas to numpy if needed
    if hasattr(X_cell_train, "values"):
        X_cell_train = X_cell_train.values
    if hasattr(X_drug_train, "values"):
        X_drug_train = X_drug_train.values
    if hasattr(X_cell_val, "values"):
        X_cell_val = X_cell_val.values
    if hasattr(X_drug_val, "values"):
        X_drug_val = X_drug_val.values

    # tensors
    X_cell_train_t = torch.tensor(X_cell_train, dtype=torch.float32)
    X_drug_train_t = torch.tensor(X_drug_train, dtype=torch.float32)
    y_train_t      = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_cell_val_t = torch.tensor(X_cell_val, dtype=torch.float32)
    X_drug_val_t = torch.tensor(X_drug_val, dtype=torch.float32)
    y_val_t      = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # loaders
    train_loader = DataLoader(
        TensorDataset(X_cell_train_t, X_drug_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_cell_val_t, X_drug_val_t, y_val_t),
        batch_size=len(y_val_t),
        shuffle=False,
    )

    # model
    cell_input_dim = X_cell_train_t.shape[1]
    drug_input_dim = X_drug_train_t.shape[1]

    model = DeepDRA(cell_input_dim=cell_input_dim, drug_input_dim=drug_input_dim)

    optimizer = Adam(model.parameters(), lr=lr)
    mse_loss  = nn.MSELoss()
    bce_loss  = nn.BCELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print("TRAINING DeepDRA (joint AE + MLP)")

    for epoch in range(1, num_epochs + 1):
        # ---------------- TRAIN ----------------
        model_train_loss = 0.0
        correct = 0
        total = 0

        for xb_cell, xb_drug, yb in train_loader:
            cell_recon, drug_recon, mlp_out = model(xb_cell, xb_drug)

            # reconstruction losses
            cell_loss = mse_loss(cell_recon, xb_cell)
            drug_loss = mse_loss(drug_recon, xb_drug)

            # classification loss
            cls_loss = bce_loss(mlp_out, yb)

            loss = (
                cell_ae_weight * cell_loss
                + drug_ae_weight * drug_loss
                + mlp_weight * cls_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_train_loss += loss.item() * xb_cell.size(0)

            # training accuracy on this batch
            preds = (mlp_out >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        avg_train_loss = model_train_loss / total
        train_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)

        # ---------------- VAL ----------------
        with torch.no_grad():
            val_total_loss = 0.0
            val_correct = 0
            val_total = 0

            all_val_targets = []
            all_val_outputs = []

            for xb_cell, xb_drug, yb in val_loader:
                cell_recon, drug_recon, mlp_out = model(xb_cell, xb_drug)

                cell_loss = mse_loss(cell_recon, xb_cell)
                drug_loss = mse_loss(drug_recon, xb_drug)
                cls_loss  = bce_loss(mlp_out, yb)

                val_loss = (
                    cell_ae_weight * cell_loss
                    + drug_ae_weight * drug_loss
                    + mlp_weight * cls_loss
                )

                val_total_loss += val_loss.item() * xb_cell.size(0)

                preds = (mlp_out >= 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

                all_val_targets.append(yb.squeeze())
                all_val_outputs.append(mlp_out.squeeze())

            avg_val_loss = val_total_loss / val_total
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

    return model, history


# ============================================================
# TEST FUNCTION FOR DEEPDRA
# ============================================================

def test_deepdra(model, X_cell_test, X_drug_test, y_test):
    """
    Evaluate a trained DeepDRA model on test data.
    Uses your Evaluation.evaluate (ROC-AUC, AUPRC, etc.).
    """

    if hasattr(X_cell_test, "values"):
        X_cell_test = X_cell_test.values
    if hasattr(X_drug_test, "values"):
        X_drug_test = X_drug_test.values

    X_cell_test_t = torch.tensor(X_cell_test, dtype=torch.float32)
    X_drug_test_t = torch.tensor(X_drug_test, dtype=torch.float32)
    y_test_t      = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        _, _, mlp_out = model(X_cell_test_t, X_drug_test_t)

    # Evaluation.evaluate expects 1D tensors
    return Evaluation.evaluate(
        all_targets=y_test_t.squeeze(),
        mlp_output=mlp_out.squeeze(),
        show_plot=False,
    )
