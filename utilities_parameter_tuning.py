import matplotlib
import matplotlib.pyplot as plt
from train_gcn import Train_GCN
import torch
import os
matplotlib.style.use('ggplot')

# Methoden zum Parameter - Tuning der GCNs
# Quelle: https://debuggercafe.com/manual-hyperparameter-tuning-in-deep-learning-using-pytorch/

def save_plots(
    train_acc, valid_acc, train_loss, valid_loss, 
    acc_plot_path, loss_plot_path
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_plot_path)
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)

def save_hyperparam(text, path):
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

def create_run(path):
    """
    Function to create `run_<num>` folders in the `outputs` folder for each run.
    """
    files = [p for p in path.iterdir()]
    num_run_dirs = len(files)
    run_dir = path / f"run_{num_run_dirs+1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    #os.makedirs(run_dir)
    return run_dir

def run_tuning(param, path, dataset_name, gcn_model, device, train_loader, val_loader, test_loader):
    path_tuning = path / f"models/tuning/{dataset_name}"
    run_dir = create_run(path_tuning)
    print(run_dir)
    lr = param["lr"]
    num_epochs = param["num_epochs"]
    # Write the hyperparameters to a YAML file.
    save_hyperparam(param, run_dir/ "hyperparam.yml")
    train_results = []
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    training = Train_GCN(gcn_model, device, lr)

    for epoch in range(1, num_epochs):
        _ = training.train(train_loader)
        with torch.no_grad():
            train_loss_epoch, train_acc_epoch = training.test(train_loader)      
            val_loss_epoch, val_acc_epoch = training.test(val_loader)
        #train_results.append([epoch, train_acc, val_acc, loss_mean.item()])
        train_loss.append(train_loss_epoch)
        valid_loss.append(val_loss_epoch)
        train_acc.append(train_acc_epoch)
        valid_acc.append(val_acc_epoch)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_epoch:.4f}, Validation Acc: {val_acc_epoch:.4f}, Loss Training: {train_loss_epoch:.4f}, Loss Validation: {val_loss_epoch:.4f}')
    test_loss, test_acc = training.test(test_loader)
    print(f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, valid_acc, train_loss, valid_loss, 
        run_dir  / "accuracy.png",
        run_dir / "loss.png")
    

