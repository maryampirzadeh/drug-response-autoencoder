import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from final_autoencoder import init_layer   # init layer is the same as autoencoder
from metrics import all_metrics


# ============================================================
# forward: x - hidden(128 with relu) - output(1 with sigmoid)
# ============================================================

def classifier_forward(x, params):
    """
    x: [64, 100]
    params: [W1, b1, W2, b2]
    W1: [hidden = 128, input =  100]
    b1: [hidden = 128 ]
    W2: [output =1, hidden = 128]
    b2: [output = 1]
    so output is 64 x 1
    """
    W1, b1, W2, b2 = params
    h = torch.relu(x @ W1.T + b1)
    out = h @ W2.T + b2
    #so out is a probability 
    out = torch.sigmoid(out)
    return out


# ============================================================
# simple class to build a model at the end
# ============================================================

class Classifier:
    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        return classifier_forward(x, self.params)
    
    #for optimizer
    def parameters(self):
        return self.params


# ============================================================
# create classifier (it's the same as autoencoder)
# ============================================================

def create_classifier(input_dim, hidden_dim=128, output_dim=1):
    #first layer
    W1, b1 = init_layer(input_dim, hidden_dim)
    #second layer
    W2, b2 = init_layer(hidden_dim, output_dim)
    params = [W1, b1, W2, b2]
    model = Classifier(params)
    return model


# ============================================================
# Train classifier on latent space
# X_train, y_train: arrays or DataFrames
# X_val, y_val: optional validation data
# ============================================================

from tqdm import tqdm

def train_classifier(
    #latent vector which is [input, 100]
    X_train,
    # 0/1 labels for samples
    y_train,
    #for validation data
    X_val=None,
    y_val=None,
    num_epochs=25,
    batch_size=64,
    lr=0.01,
):
    #convert to numpy if pandas
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if X_val is not None and hasattr(X_val, "values"):
        X_val = X_val.values

    #create tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    #load data in pytorch format
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        #we turn [x] to [x,1]
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            #entire validation is one batch
            batch_size=len(y_val_tensor),
            #validation shouldn't be random
            shuffle=False,
        )
    else:
        #if no validation data
        val_loader = None

    input_dim = X_train_tensor.shape[1]
    model = create_classifier(input_dim)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    print("TRAINING CLASSIFIER")
    #to check loss and accuracy
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    #main training part
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for xb, yb in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False
        ):
            classifier_output = model(xb)                 # forward
            loss = loss_fn(classifier_output, yb)
            #update gradiant
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #loss is a tensor. .item() converts it to a number
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # for validation data (it's the same as train data)
        if val_loader is not None:
            total_val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    classifier_output = model(xb)
                    loss = loss_fn(classifier_output, yb)
                    total_val_loss += loss.item()

            val_loss = total_val_loss / len(val_loader)
            history["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}"
            )

    #print(history)        
    return model


# ============================================================
# using test data (validation and train were used in the previous func)
# {'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC', 'AUPRC'} are calculated
# ============================================================
def result_classifier(model, X_test, y_test):
    if hasattr(X_test, "values"):
        X_test = X_test.values
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        outputs = model(X_test_tensor)
    # your Evaluation expects tensors (1D)
    return all_metrics(
        real_labels=y_test_tensor.squeeze(),
        classifier_output=outputs.squeeze(),
    )
