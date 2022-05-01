

from tqdm import tqdm
import torch
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method import GNNExplainer
from datetime import datetime

# Methoden zum Tuning des GNNExplainers

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




def run_tuning(param, model, ds_test, device, path, dataset_name):
    """
    Tuning durchführen für GNNExplainer
    Parameters:
    :param param: Parameter.
    :param model: zugrundeliegendes GCN-Model
    :param ds_test: Datensatz zum trainieren
    :param device: genutzte Laufzeit
    :param path: Path to save the hyperparmeters.
    :param dataset_name: Name des genutzten Datensatzes
    """
    lr = param["lr"]
    epochs = param["epochs"]
    sparsity = param["sparsity_ziel"]

    # Speicherort
    path_tuning = path / f"xai_methods/gnnexplainer/tuning/{dataset_name}"
    # timestamp zum speichern nutzen
    ts = int(datetime.timestamp(datetime.now()))
    name = f"{str(ts)}_{str(lr)} _{dataset_name}" 

    print(f'-------------Learning Rate: {lr}--------------')
    # Init Explainer
    explainer = GNNExplainer(model, epochs=epochs, lr=lr, explain_graph=True)
    # Init Collector
    xCol = XCollector()      

    print("GNNExplainer anwenden")
    for data in tqdm(ds_test): 
        
        data = data.to(device) 
        y = int(data.y.item())
        soft_masks, hard_masks, related_preds = \
            explainer(data.x, data.edge_index,  sparsity=sparsity, num_classes=2)
        xCol.collect_data(hard_masks, related_preds, y )

    # Ergebnisse in param Speichern
    param["fidelity+"] = xCol.fidelity
    param["fidelity-"] = xCol.fidelity_inv
    param["sparsity"]  = xCol.sparsity
    print(f"Fidelity+: {xCol.fidelity}, Fidelity-: {xCol.fidelity_inv}, Sparsity: {xCol.sparsity}")

    # Ergebnisse Speichern
    save_hyperparam_results(param, path_tuning / f"{name}.yml")
    
    #list_explainer.append(explainer) 
    return xCol

  