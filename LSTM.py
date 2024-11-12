

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



class neural_network(nn.Module):
    def __init__(self, hidden_size, num_layers  ):
        super(neural_network, self).__init__() 
        self.lstm = nn.LSTM(input_size=length_features,hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=length_features)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = output[:,-1,:] #récupère tout batcg_size, tout hidden_size mais uniquement le dernier terme de la séquence prédit par le LSTM
        output = self.fc1(torch.relu(output))
        return output

model = neural_network()


def entrainement_lstm(model, optimizer, criterion, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scheduler, EPOCHS):

    """
    Boucle d'entrainement du modèle LSTM
    arrêt de l'entrainement au bout de 3 epochs successives où le loss ne diminue plus
    réduction progressive du learning rate

    """

    train_losses = []
    val_losses = []
    min_val_loss = float("inf")
    early_stop = 0

    for epoch in range(EPOCHS):

        train_outputs = model(X_train_tensor)
        optimizer.zero_grad()
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            scheduler.step(val_loss)
            val_losses.append(val_loss.item())

        mse = mean_squared_error(y_test_tensor.cpu(), val_outputs.cpu())


        print(f' { epoch }/{ EPOCHS },  average_train_loss = {train_loss}, average_val_loss = {val_loss}, MSE = {mse}')
              
        if val_loss < min_val_loss :
            min_val_loss = val_loss
            early_stop = 0 
        else :
            early_stop +=1

        if early_stop == 3:
            print("EARLY STOPPING")
            break


              
    return mse, train_losses, val_losses


def lstm_predict(model, X_test_tensor):
    with torch.no_grad():
        prediction = model(X_test_tensor).detach().numpy() #item c'est pour avoir un numpy et pas un tensor
        return prediction
prediction = lstm_predict(model, X_test_tensor)



if __name__ == "__main__":

    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    y_test_tensor = torch.from_numpy(y_test).float()



    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    EPOCHS = 25
    length_features = 9



    param_grid = {

        "hidden_size": [10, 50, 100, 100],
        "num_layers": [1, 2, 3, 4],
        "lr" : [0.001, 0.0001]

    }

    best_mse = float("inf")
    best_params = {}
    best_model = None


    for hidden_size in param_grid["hidden_size"]:
        for lr in param_grid["lr"]:
            for num_layers in param_grid["num_layers"]:
                model = model = neural_network(hidden_size=hidden_size, num_layers=num_layers)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
                
                print(f'entrainement avec paramètres : hidden_size : {hidden_size}, num_layers : {num_layers}, lr : {lr}')
                mse, train_losses, val_losses =  entrainement_lstm(model, optimizer, criterion, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scheduler, EPOCHS)
                #retourne uniquement le mse à la fin de la dernière epoch ou au moment de la condition d'arrêt


                plt.plot(train_losses, label='Training Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'entrainement avec paramètres : hidden_size : {hidden_size}, num_layers : {num_layers}, , lr : {lr}')
                plt.legend(f'entrainement avec paramètres : hidden_size : {hidden_size}, num_layers : {num_layers}, , lr : {lr}')
            
                save_dir = "/Users/jeremytournellec/Desktop/figuresLSTM/"
                legend_name = f'hidden_size{hidden_size}_layers{num_layers}_lr{lr}'
                save_path = os.path.join(save_dir, f'loss_curve_{legend_name}.png')
                plt.savefig(save_path)
                plt.show()
                plt.close()
                    
                if mse < best_mse: #meilleur mse de tous les modèles avant leur condition d'arrêt
                    best_mse = mse
                    best_model = model
                    best_params = {
                    "hidden_size": hidden_size,
                    "lr": lr,
                    "num_layers": num_layers,
                    }

