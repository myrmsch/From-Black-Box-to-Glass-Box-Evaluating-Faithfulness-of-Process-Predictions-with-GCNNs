
from tqdm import tqdm
import torch
import random
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method import PGExplainer
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import os

# Methoden zum Parameter-Tuning des PGExplainers

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

def save_explainer(explainer_pge, path_pgexplainer):
    #pgexplainer_training_name = "pgexplainer_" + dataset_name + str(lr)
    torch.save(explainer_pge.state_dict(), path_pgexplainer)
    print(f"Saved: {path_pgexplainer}")



def run_tuning(param, model, ds_train, ds_test, dataset_name, device, path, hidden_channel, sparsity_add = 0):
    '''
    Tuning des PGExplainers durchführen und Ergebnisse als yaml-File speichern
    '''
    # Parameter
    learning_rate = param["lr"]
    sparsity = param["sparsity_ziel"]
    coff_size = param["coff_size"]
    coff_ent  = param["coff_ent"]
    t0 = param["t0"]
    t1 = param["t1"]
    epochs = param["epochs"]

    # Speicherort
    path_tuning = path / f"xai_methods/pgexplainer/tuning/{dataset_name}"
    # timestamp zum speichern nutzen
    ts = int(datetime.timestamp(datetime.now()))
    name = f"{str(ts)}_{str(learning_rate)} _{dataset_name}" 
    print(f"---------------Learning-Rate: {learning_rate}--------------------------")
    
    # Explainer instantiieren
    explainer = PGExplainer(model, hidden_channel*2, device = device, epochs = epochs, lr = learning_rate, coff_ent = coff_ent, coff_size = coff_size, t0 = t0, t1 = t1 )

    # Training durchführen
    explainer.train_explanation_network(ds_train)

    # Ergebnisse der Erklärungen mit trainierten Explainer
    col = get_results(param, ds_test, explainer, device, sparsity, path_tuning / f"{name}.yml", sparsity_add)
    #list_explainer.append(explainer)
    return col


def get_results(param, ds_test, explainer_pge, device, sparsity, path, sparsity_add):   

    
    xCol_pge = XCollector()    

    print("PGExplainer Anwenden")
    for data in tqdm(ds_test): 
        
        data = data.to(device) 
        top_k  = int(data.x.shape[0] * (1-sparsity)) + sparsity_add
        _, masks, related_preds = explainer_pge(data.x, data.edge_index, top_k = top_k)
        xCol_pge.collect_data(masks, related_preds)
    explainer_pge.__clear_masks__()

    # Ergebnisse in param Speichern
    param["fidelity+"] = xCol_pge.fidelity
    param["fidelity-"] = xCol_pge.fidelity_inv
    param["sparsity"]  = xCol_pge.sparsity

    print(f"Fidelity+: {xCol_pge.fidelity}, Fidelity-: {xCol_pge.fidelity_inv}, Sparsity: {xCol_pge.sparsity}")

    save_hyperparam_results(param, path)

    #collector_list_pgexplainer.append(xCol_pge)
    return xCol_pge

def split_dataset_2_8(dataset):
    try:
        random.seed(33)   # Damit gleich gesplittet wir
        random.shuffle(dataset)   # Mischen
    except:
        torch.manual_seed(33) 
        dataset = dataset.shuffle(33)

    # Split Datasets in Train, Test und XAI_Validation Dataset
    split = int(0.8*len(dataset))
    #dataset = dataset.shuffle()
    #xai_set = dataset[:1]
    ds_train = dataset[:split]
    ds_test = dataset[split:]
    return ds_train, ds_test

def visual_tuning_results(dataset_name, path, gnnexplainer = False):
    '''
    Tuning-Ergebnisse in Form eines Bar-Diagramms darstellen
    '''
    # Welcher Explainer verwendet wurde
    if not gnnexplainer:
        directory = path / f"xai_methods/pgexplainer/tuning/{dataset_name}"
    else:
        directory = path / f"xai_methods/gnnexplainer/tuning/{dataset_name}"

    # Laden der Tuning Ergebnisse
    results = load_yaml_files(directory)

    # Name des Files, Fidelity+ und Fidelity- Werte in Listen darstellen
    names = [x["name"] for x in results]
    fidelity = [x["fidelity+"] for x in results]
    fidelity_inv = [x["fidelity-"] for x in results]

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,7))

    # Bar-Diagramm für Fidelity+ Werte
    ax1.barh(names,  fidelity,  align='center')
    ax1.set_xlabel('Fidelity+')

    # Bar-Diagramm für Fidelity- Werte
    ax2.barh(names,  fidelity_inv,  align='center')
    ax2.set_xlabel('Fidelity-')

def load_yaml_files(directory):
    '''
    yaml-File laden und als python-Dict darstellen
    '''
    # liste aller yml Files im Ordner bekommen
    items2 = directory.glob('*'+".yml")
    # durch diese Iterieren und als Dict öffnen
    results = []
    for item in items2:  
        name = os.path.basename(item)
        with open(item, "r") as f:
            data_dict = yaml.safe_load(f)
        # file-name im dict speichern
        data_dict["name"] = name
        results.append(data_dict)
    return results

