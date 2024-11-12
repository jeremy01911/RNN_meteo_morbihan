import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # Si d_model est impair, on ajuste pour éviter l'erreur
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # (d_model//2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # Sinus pour les positions paires
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)  # Cosinus pour les positions impaires
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # Ajuster pour d_model impair

        pe = pe.unsqueeze(0).transpose(0, 1)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # Ajouter le codage positionnel
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, length_features )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x
    

def entrainement_transformer(model, optimizer, criterion, train_loader, test_loader, scheduler, EPOCHS):

    """
    Boucle d'entrainement du modèle Transformer
    arrêt de l'entrainement au bout de 3 epochs successives où le loss ne diminue plus
    réduction progressive du learning rate

    """


    all_train_losses=[]
    all_val_losses=[]
    early_stop = 0
    min_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_losses=[]
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss_mean= np.mean(train_losses)
        all_train_losses.append(train_loss_mean)

        # Validation
        model.eval()
        val_losses = []
        val_outputs = []
        val_targets = []

        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                val_outputs.append(outputs) #on conserve tous les outputs de l'epoch pour calcul MSE
                val_targets.append(y_batch) #on conserve tous les targets de l'epoch pour calcul MSE
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item()) #on conserve toutes les loss pour calculer la moyenne à l'échelle de l'epoch
        val_loss_mean = np.mean(val_losses) #on prend la loss moyenne sur tous les batchs sinon ce n'est pas significatif
        scheduler.step(val_loss_mean)
        all_val_losses.append(val_loss_mean)

        mse = mean_squared_error(torch.cat(val_targets).cpu(), torch.cat(val_outputs).cpu())
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss_mean:.4f},  Validation Loss: {val_loss_mean:.4f}, mse:{mse:.4f}")
    

        if val_loss_mean < min_val_loss :
            min_val_loss = val_loss_mean
            early_stop = 0 
        else :
            early_stop +=1

        if early_stop == 3:
            print("EARLY STOPPING")
            break


              
    return mse, all_train_losses, all_val_losses



def transformer_predict(model, test_loader):
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.squeeze().tolist())
        return predictions
    

if __name__ == "__main__":


    device = torch.device("mps")  
    model = model.to(device)

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 32  


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    EPOCHS = 25


    param_grid = {

        "d_model": [length_features * 4, length_features * 10],
        "nhead": [2, 6, 12],
        "num_layers": [1, 2, 3],
        "lr" : [0.001, 0.0001]

    }

    best_mse = float("inf")
    best_params = {}
    best_model = None


    for d_model in param_grid["d_model"]:
        for nhead in param_grid["nhead"]:
            for num_layers in param_grid["num_layers"]:
                model = TransformerModel(input_dim=length_features, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=0.2)
                model = model.to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
                
                print(f'entrainement avec paramètres : d_model : {d_model}, nhead : {nhead}, num_layers : {num_layers}, lr : {lr}')
                mse, all_train_losses, all_val_losses =  entrainement_transformer(model, optimizer, criterion, train_loader, test_loader, scheduler, EPOCHS=EPOCHS) #retourne uniquement le mse à la fin de la dernière epoch ou au moment de la condition d'arrêt


                plt.plot(all_train_losses, label='Training Loss')
                plt.plot(all_val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'entrainement avec paramètres : d_model : {d_model}, nhead : {nhead}, num_layers : {num_layers}, , lr : {lr}')
                plt.legend(f'entrainement avec paramètres : d_model : {d_model}, nhead : {nhead}, num_layers : {num_layers}, , lr : {lr}')
            
                save_dir = "/Users/jeremytournellec/Desktop/figuresTransformer/"
                legend_name = f'dmodel{d_model}_nhead{nhead}_layers{num_layers}_lr{lr}'
                save_path = os.path.join(save_dir, f'loss_curve_{legend_name}.png')
                plt.savefig(save_path)
                plt.show()
                plt.close()
                    
                if mse < best_mse: #meilleur mse de tous les modèles avant leur condition d'arrêt
                    best_mse = mse
                    best_model = model
                    best_params = {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    }


        
