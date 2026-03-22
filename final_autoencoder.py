
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ============================================================
# initialize layer (weight and bias)
# ============================================================
def init_layer(in_dim, out_dim):
    #in_dim = number of input features, out_dim = number of neurons of this layer
    #randn = normal distribution (mean 0, variance 1) also noise is 0.02 (0.01 to 0.05)
    W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)    
    #1D vector full of zeros with length out_dim
    b = nn.Parameter(torch.zeros(out_dim))
    print(W.shape,b.shape)
    return W, b


# ============================================================
# weight and bias calculation + activation func (forward)
# x: the input data [batch_size, input_dim].
# params: a list of all encoder parameters (weights & biases).
# ============================================================
def encoder_forward(x, params):
    #W1, b1: first layer (hidden), W2, b2: second layer (final)
    W1, b1, W2, b2 = params
    #this is for hidden layer. x shape is [batch_size, input_dim], W1 shape is [hidden_dim, input_dim]
    h1 = torch.relu(x @ W1.T + b1)
    #same for latent which is 50
    z = h1 @ W2.T + b2
    return z

#same as encoder but for decoder
def decoder_forward(z, params):
    W3, b3, W4, b4 = params
    h2 = torch.relu(z @ W3.T + b3)
    out = h2 @ W4.T + b4
    return out


# ============================================================
# to represnt the autoencoder as a model
# ============================================================
class Autoencoder:
    def __init__(self, encoder_params, decoder_params):
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        # concat both lists into a single list
        self._params = encoder_params + decoder_params

    def __call__(self, x):
        z = encoder_forward(x, self.encoder_params)
        out = decoder_forward(z, self.decoder_params)
        return out

    def encode(self, x):
        return encoder_forward(x, self.encoder_params)

    def parameters(self):
        return self._params


# ============================================================
# autoencoder creation (we return model at the end)
# ============================================================
def create_autoencoder(input_dim):
    # encoder
    W1, b1 = init_layer(input_dim, 256)
    W2, b2 = init_layer(256, 50)

    # decoder
    W3, b3 = init_layer(50, 256)
    W4, b4 = init_layer(256, input_dim)

    encoder_params = [W1, b1, W2, b2]
    decoder_params = [W3, b3, W4, b4]

    model = Autoencoder(encoder_params, decoder_params)
    return model


# ============================================================
# main part to train autoencoder (we only need latent space)
# x is dataframe
# epochs/lr are generated from paper, model name = drug or cell
# ============================================================
def train_autoencoder(
    X,
    epochs=25,
    batch_size=64,
    lr=0.001,
    model_name="drug_autoencoder"
):

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    loader = DataLoader(
        #the model tries to predict its own input so inp = target
        TensorDataset(X_tensor, X_tensor),
        batch_size=batch_size,
        shuffle=True
    )
    
    
    #number of features
    input_dim = X.shape[1]
    model = create_autoencoder(input_dim)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.00000001)
    loss_fn = nn.MSELoss()

    print(f"TRAINING AUTOENCODER ({model_name})")

    # training
    for epoch in tqdm(range(1, epochs + 1)):
        # this is for calculating total loss
        total = 0
        for inp, tar in loader:
            #forward input to get predictions
            pred = model(inp)
            loss = loss_fn(pred, tar)
            # clear old gradients
            optimizer.zero_grad()
            # calculate new gradients. (back propagation)
            loss.backward()
            # update values(w and b)
            optimizer.step()
            # loss for this batch
            total += loss.item()

        tqdm.write(f"Epoch {epoch}/{epochs} | Loss = {total:.5f}")



    # we dont need to calculate gardients anymore
    with torch.no_grad():
        #we only need encoder part, [num of samples, 50 latent features]
        latent = model.encode(X_tensor).numpy()
        #print(latent)

    return model, latent

