import torch
import joblib
import random
from torch_geometric.data import DataLoader
from models.gcn3_neu import *

# Methoden zum trainieren der GCN-Modelle
# Methoden zum speichern und laden der Modelle

def train_model(dataset_list_review, batch_size, gcn_model, device, param, key, dataset_name, path):
    # Variablen
    learning_rate = param["lr"]
    num_epochs = param["epochs"]
    review_results = {}
    # Datensatz 
    dataset = dataset_list_review[key]
    tr_loader, v_loader, t_loader  = create_splitted_dataloader(dataset, batch_size)   
    # Model helfer Klasse    
    training = Train_GCN(gcn_model, device, learning_rate)

    train_results = []

    # Trainieren des Models
    for epoch in range(1, num_epochs):
        # train
        _ = training.train(tr_loader)        
        train_loss, train_acc = training.test(tr_loader)
        # val
        val_loss, val_acc = training.test(v_loader)
        # save
        train_results.append([epoch, train_acc, val_acc, train_loss, val_loss])
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}, Loss train: {train_loss}, Loss val: {val_loss}')
    # test
    test_loss, test_acc = training.test(t_loader)
    # save
    review_results[key] = [train_results, test_loss, test_acc]
    print(f'Test Acc: {test_acc:.4f}')

    #Speichern des Models
    print(f"save:  {key}")
    path_run = path / f"models/training/{dataset_name}_{str(key)}"  
    # Speichern der Parameter + letzten Acc/Loss
    param["val_loss"] =  train_results[-1][4]
    param["val_acc"] = train_results[-1][2]
    param["train_loss"] = train_results[-1][3]
    param["train_acc"] = train_results[-1][1]
    save_hyperparam_results(param, path_run / "param.yml")
    torch.save(gcn_model.state_dict(), path_run / "model" )
    joblib.dump(review_results, path_run / "results.joblib")

def load_gcn_key(dataset_name, key, input_dim, device, path_folder):
    path_model = path_folder / f"models/training/{dataset_name}_{str(key)}"
    model = GCN3(hidden_channels=100, input_dim = input_dim, num_layer = 3).to(device)
    model.load_state_dict(torch.load(str(path_model / "model")))
    model.eval()
    return model

def load_gcn(dataset_name, input_dim, device, path_folder):
    path_model = path_folder / f"models/training/{dataset_name}"
    model = GCN3(hidden_channels=100, input_dim = input_dim, num_layer = 3).to(device)
    model.load_state_dict(torch.load(str(path_model / "model")))
    model.eval()
    return model


def save_hyperparam_results(text, path):
    """
    Function to save hyperparameters in a `.yml` file.
    Parameters:
    :param text: The hyperparameters dictionary.
    :param path: Path to save the hyperparmeters.
    """
    print("save Hyperparameter")
    with path.open('w') as f:
        keys = list(text.keys())
        for key in keys:
            f.writelines(f"{key}: {text[key]}\n")

def create_splitted_dataloader(dataset, batch_size): 
    # Split Datasets in Train, Test und Validation Dataset
    

    split1 = int(0.7*len(dataset))
    split2 = int(0.9*len(dataset))
    try:
        random.seed(33)   # Damit gleich gesplittet wir
        random.shuffle(dataset)   # Mischen
    except:
        torch.manual_seed(33) 
        dataset = dataset.shuffle(33)
        

    ds_train = dataset[:split1]
    ds_val = dataset[split1:split2]
    ds_test = dataset[split2:]

    # Dataloader erstellen
    train_loader = DataLoader(ds_train, batch_size=batch_size)
    val_loader = DataLoader(ds_val, batch_size=batch_size)
    test_loader = DataLoader(ds_test, batch_size=batch_size)
    return train_loader, val_loader, test_loader

class Train_GCN():
    def __init__(self, model, device, lr):
        self.model = model  
        self.device = device
        self.lr = lr
        self.optimizer, self.criterion = self.init_operators()
        

    def init_operators(self):      
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        return optimizer, criterion

    def train(self, train_loader):
        self.model.train()
        
        loss_sum = 0
        for data in train_loader:     #train_loader:  # Iterate in batches over the training dataset.
            data = data.to(self.device)
            batch = data.batch
            #model.setbatch(batch)

            # 1) Forward-Propagation
            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            out = self.model(data.x, data.edge_index, batch)  # Perform a single forward pass.
            # label = data.y.to(device)     
            

            # 2) Loss berechnen           
            label = data.y
            label =label.long()
            
            
            loss = self.criterion(out, label)  # Compute the loss.
            loss_sum += loss
            loss.backward()  # Derive gradients.

            # 3) Backward-Propagation
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            self.optimizer.step()  # Update parameters based on gradients.

            # 4) Gradienten löschen
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)          
            self.optimizer.zero_grad()  # Clear gradients.
        return loss_sum/len(train_loader)

    def test(self, loader):
        self.model.eval()

        correct = 0
        loss_sum = 0
        counter = 0
        #  i = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            counter +=1
            data = data.to(self.device)                 
            label = data.y.squeeze(-1).to(self.device)  # zum trainieren des Baseline-Datensatzes         
            batch = data.batch
            #model.setbatch(batch)
            out = self.model(data.x, data.edge_index, batch)  

            # 2) Loss berechnen           
            label = data.y
            label =label.long()

            loss = self.criterion(out, label)  # Compute the loss.
            loss_sum += loss.item()
            #print(out)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            #print(pred)
            correct += int((pred == label).sum())  # Check against ground-truth labels.
            #  i += 1
            #  if i ==2:
            #   break
        epoch_acc = correct / len(loader.dataset)  # Rate der richtigen Vorhersagen.
        epoch_loss = loss_sum / counter # Durchschnittlicher Loss
        return epoch_loss, epoch_acc


