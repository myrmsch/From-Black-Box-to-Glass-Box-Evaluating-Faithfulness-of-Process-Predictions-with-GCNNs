import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch
from torch_geometric.data import DataLoader, Data
from torch import Tensor
from dig.xgraph.method.pgexplainer import PlotUtils

from torch_geometric.utils import to_networkx
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
import numpy as np
from torch_geometric.utils import to_networkx
from collections import defaultdict
from torch_geometric.utils.loop import add_self_loops, remove_self_loops


# utilities.py enthält verschiedene Methoden der Visualisierung von Erklär-Graphen. Des Weiteren ist hier der Random-Explainer implementiert

def plot_explanation(data, explanation, method_name, graph = None, title = None, top_k = None, ax = None):
    '''
    Diese Methode viusalisert eine Erklärung. Sie erhält die Edge-Masken als Input und erstellt ein NetworkX Objekt
    :param data: Instanz-Graph
    :param explanation: Edge-Maske
    :param method_name: Name der Methode
    :param title: Titel über der erstellten Grafik
    :param top_k: Muss im Fall des PGExplainers mit angegeben werden
    :param ax: Falls mehrere Graphen als Subplots visualisiert werden, muss die Position des Subplots mitangegeben werden
    '''
    # Von Cuda auf cpu
    data = data.to('cpu')
    try:
        explanation = explanation.to("cpu")
    except:
        pass
    if graph == None:
        graph = to_networkx(data)        

    # 1. Vorverarbeiten der Erklärung
    if method_name == "gnn_explainer":
        nodelist, edge_list  = preprocess_gnn_explainer(data.edge_index, explanation, data.x.size(0))
    elif method_name == "pgexplainer":
        nodelist, edge_list  = preprocess_gnn_explainer(data.edge_index, explanation, data.x.size(0), top_k = top_k, selfloops= False, hard_mask = False)
    elif method_name == "gradcam":
        nodelist, edge_list  = preprocess_gnn_explainer(data.edge_index, explanation, data.x.size(0))
    # elif method_name == "subgraphx":
    #     nodelist = explanation[0]["coalition"]
    #     edge_list = None
    # elif method_name == "ggnn_explainer":
    #     nodelist = explanation
    #     edge_list = None
    
    # 2. Visualisieren der Erklärung
    if method_name != "ggnn_explainer":
        if ax == None:
            plot_subgraph(graph, nodelist, edge_list)
        else:
            plot_subgraph_subplot(graph, nodelist, ax, edgelist=edge_list, title_sentence=title)
    else:
        if ax == None:
            plot_subgraph(graph, nodelist, edge_mask=False, title_sentence=title)
        else:
            plot_subgraph_subplot(graph, nodelist, ax, edgelist=edge_list, title_sentence=title, edge_mask = False)

def preprocess_gnn_explainer(edge_index, edge_mask, num_nodes, top_k = None, hard_mask = True, selfloops = True):
    if selfloops == True:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_mask = torch.FloatTensor(edge_mask)
    return get_topk_edges_subgraph(edge_index, edge_mask, top_k, hard_mask = hard_mask)
    

    

def get_topk_edges_subgraph(edge_index, edge_mask, top_k, hard_mask):
    '''
    Von einer Weichen-Maske die top_k Kanten erhalten 
    '''
    # if un_directed:
    #     top_k = 2 * top_k
    if hard_mask == False:
        edge_mask = edge_mask.reshape(-1)

        # Anzahl an Edges, welche aussortiert werden
        thres_index = max(edge_mask.shape[0] - top_k, 0)

        # Grenzwert erhalten
        threshold = float(edge_mask.reshape(-1).sort().values[thres_index])

        # Edge Mask mit Boolschen Wertden >= Grenzwert
        edge_mask = (edge_mask >= threshold)

    # Indizes der gewählten Kanten in edge_mask erhalten
    selected_edge_idx = np.where(edge_mask == 1)[0].tolist()
    nodelist = []
    edgelist = []

    # gewählte Knoten und Kanten Indices erhalten
    for edge_idx in selected_edge_idx:
        edges = edge_index[:, edge_idx].tolist()
        nodelist += [int(edges[0]), int(edges[1])]
        edgelist.append((edges[0], edges[1]))
    nodelist = list(set(nodelist))
    return nodelist, edgelist

def plot_subgraph_subplot(graph,
                nodelist,
                ax, 
                colors= "#81DAF5", #'#FFA500', #Union[None, str, List[str]] = '#FFA500',
                labels=None,
                edge_color='gray',
                edgelist=None,
                subgraph_edge_color='black',
                title_sentence=None,
                edge_mask = True,
                figname=None,
                plot_graph = True):
    '''
    Funktion zum erstellen einer graphischen Dartstellung für eine Erklärung, generiert durch den GGNN-Erklärer      

    '''
    # Angelehnt an: https://github.com/divelab/DIG/blob/8f99eb55a0892410e44822b6641124f10f49862d/dig/xgraph/method/subgraphx.py#L171

    #nodelist = nodelist.tolist()
    if edgelist is None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                    if n_frm in nodelist and n_to in nodelist]
    pos = nx.kamada_kawai_layout(graph)
    # Position der Knoten im Graphen in der Knoten-Liste
    pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
    # Label
    n_labels = {x:str(x) for x in list(graph.nodes())} 

    plt.sca(ax)
    #fig = plt.figure(figsize=(15, 8))
    nx.draw_networkx_nodes(graph, pos,
                            nodelist=list(graph.nodes()),
                            #node_color=colors,
                            node_size=150,
                            ax = ax)

    nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_color, ax = ax, arrows=False)
    if edge_mask == True:
        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                                edgelist=edgelist, width=4,
                                edge_color=subgraph_edge_color, 
                                ax = ax,
                                arrows=False)
    else:
        nx.draw_networkx_nodes(graph, pos = pos_nodelist,
                            nodelist=nodelist,
                            node_color=colors,
                            node_size=150,
                            )
    nx.draw_networkx_labels(graph ,pos, labels=n_labels)
    
    ax.set_title(title_sentence, fontsize=10)
    ax.set_axis_off()

def plot_subgraph(graph,
                nodelist,
                colors= '#FFA500', #Union[None, str, List[str]] = '#FFA500',
                labels=None,
                edge_color='gray',
                edgelist=None,
                subgraph_edge_color='black',
                title_sentence=None,
                figname=None,
                plot_graph = True,
                edge_mask = True):
    '''
    Funktion zum erstellen einer graphischen Dartstellung für eine Erklärung      

    '''
    # Angelehnt an: https://github.com/divelab/DIG/blob/8f99eb55a0892410e44822b6641124f10f49862d/dig/xgraph/method/subgraphx.py#L171
    #nodelist = nodelist.tolist()
    if edgelist is None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                    if n_frm in nodelist and n_to in nodelist]
    pos = nx.kamada_kawai_layout(graph)
    pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
    n_labels = {x:str(x) for x in list(graph.nodes())} 

    fig = plt.figure(figsize=(15, 8))
    nx.draw_networkx_nodes(graph, pos,
                            nodelist=list(graph.nodes()),
                            #node_color=colors,
                            node_size=300)

    nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

    if edge_mask == True:

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                            edgelist=edgelist, width=6,
                            edge_color=subgraph_edge_color,
                            arrows=False)
    else:
        nx.draw_networkx_nodes(graph, pos = pos_nodelist,
                            nodelist=nodelist,
                            node_color=colors,
                            node_size=150,
                            )
    nx.draw_networkx_labels(graph ,pos, labels=n_labels)
    plt.title(title_sentence)


def plot_explanation_graphs(instance, feature, device, explainer_pge, explainer_gnnexplainer, explainer_grad_cam, title, sparsity = 0.7, not_ohe = None):
    '''
    Darstellen der Erklär-Graphen aller XAI-Techniken als Matplotlib-Subplot
    '''
    # Instanz-Variablen
    instance = instance.to(device)
    x = instance.x
    y = int(instance.y.item())
    edge_index = instance.edge_index
    top_k_vis  = int(x.shape[0] * (1-sparsity)) 

    # Erklärungen erstellen
    _, soft_mask_pgexplainer, pge_explainer_preds = explainer_pge(x, edge_index, top_k = top_k_vis)
    _, hard_mask_gnnexplainer, gnnexplainer_preds = explainer_gnnexplainer(x, edge_index, num_classes = 2, sparsity = sparsity)
    _, hard_mask_gradcam, gradcam_preds = explainer_grad_cam(x, edge_index, sparsity=sparsity, num_classes=2)

    # Erklär-Maske für Label wählen
    soft_mask_pgexplainer  = soft_mask_pgexplainer[0]   # PGExplainer gibt Maske und Preds für Ziellabel aus
    hard_mask_gnnexplainer  = hard_mask_gnnexplainer[y]  # GNNExplainer und Grad-Cam geben für jedes Label eine Maske und Preds aus
    hard_mask_gradcam = hard_mask_gradcam[y]

    # Graphen erstellen
    plot_subgraphs(instance, hard_mask_gnnexplainer, hard_mask_gradcam, soft_mask_pgexplainer, top_k_vis, feature, title, not_ohe)
    return pge_explainer_preds, gnnexplainer_preds, gradcam_preds

def get_labels_ohe(nodes, node_features):   
      
        n_labels = {}
        for n_id, node in enumerate(nodes):
            # Float zu Int transformieren
            node = list(map(int, node))
            # Erhalten der Benennungen durch Multiplizieren der node_feature-Liste mit der Encoded node-Liste. 
            # Durch OHE ist Wert eines Features [1,0]. Ergebnis ist eine Liste aller Feature. Output: {node_id: [feature]}
            n_labels[n_id] = sum([[s] * n for s, n in zip(node_features, node)], [])
        return n_labels

def get_labels_ohe_event_based(nodes, node_features, empty_columns):   
        # Die ersten Spalten wurden nicht durch ohe Verschlüsselt. über empty_columns wird festgelegt, ab welcher Spalte ohe angewendet wurden
        # Für die nicht über ohe encodeden Werte, werden Platzhalter eingefüllt 
        n_labels = {}
        node_features = node_features[empty_columns:]
        for n_id, node in enumerate(nodes):
            # Float zu Int transformieren
            node = list(map(int, node))
            # Feature, welche nicht ohe sind aus Knoten ausschließen
            node = node[empty_columns:]
            # Erhalten der Benennungen durch Multiplizieren der node_feature-Liste mit der Encoded node-Liste. 
            # Durch OHE ist Wert eines Features [1,0]. Ergebnis ist eine Liste aller Feature. Output: {node_id: [feature]}
            # Außerdem werden als 0 die nicht ohe encodeden Spalten angehängt
            n_labels[n_id] = [0] * empty_columns + sum([[s] * n for s, n in zip(node_features, node)], [])
        return  n_labels

def plot_subgraphs(instance, hard_mask_gnnexplainer, hard_mask_gradcam, soft_mask_pgexplainer, top_k_vis, feature, title, not_ohe):
    # Darstellen als Subplot
    fig, ax = plt.subplots(2, 2,figsize=(16, 12))
    fig.suptitle(title)

    # Legende-Elemente erstellen
    # Knoten-ID Feature zuordnen
    if not_ohe == None:  # Aktivität baiserte Instancen, welche lediglich ohe wurden
        lab = get_labels_ohe(instance.x, feature)
    else:  # Event-basierte Ansätze, welches sowohl ohe als auch ordinal-Encoded Feature besitzen
        lab = get_labels_ohe_event_based(instance.x, feature, not_ohe)
    # Legenden Eintrag erstellen als String (Knoten id: Feature des Knotens)
    legend = [str(x) + ": " + str(lab[x]) for x in lab]
    # Custom Elemente der legende erstellen (Blaue Linie für jeden Eintrag der Legende)
    cmap = plt.cm.coolwarm
    element = Line2D([0], [0], color=cmap(0.), lw=4)
    custom_lines = [element for i in range(len(legend))]


    # Graphen erstellen
    plot_explanation(instance, hard_mask_gnnexplainer, "gnn_explainer", ax = ax[0,0], title = "GnnExplainer")
    plot_explanation(instance, hard_mask_gradcam, "gradcam", ax = ax[0,1], title = "GradCam")
    plot_explanation(instance, soft_mask_pgexplainer, "pgexplainer", top_k = top_k_vis, ax = ax[1,0], title = "PGExplainer")

    # leere Axe für Legende
    ax[1,1].set_axis_off()
    # Legende erstellen
    ax[1,1].legend(custom_lines, legend, loc='lower right' ) 

    




class RandomExplainer():
    '''
    Klasse, welche Random-Explainer enthält. Erklärungen, in Form von Hard-Masks, werden durch zufällige Auswahl von Graphen Bestand-Teilen zurückgegeben.
    Es können entweder zufällig Kanten oder Knoten des Graphen ausgewählt werden.
    '''
    def __init__(self) -> None:
        pass

    def explain(self, x, edge_index, sparsity, model, random_edge):
        #top_k  = int(x.shape[0] * (1-sparsity))   #Knoten Sparsity
        #top_e = int(edge_index[0].shape(0) * (1-sparsity)) # Kanten Sparsity

        # edge_mask auf Basis der Sparsity erstellen  
        while True:
            if random_edge == True:   
                # erstellen einer hard-edge-mask 
                mask =  np.random.choice([0, 1], size=(edge_index[0].size(0)), p=[sparsity, 1-sparsity])
                mask = torch.from_numpy(mask)
            else:
                # erstellen einer hard-node-mask
                mask =  np.random.choice([0, 1], size=(x.size(0)), p=[sparsity, 1-sparsity])            
                mask = torch.from_numpy(mask)
            #print(mask)
            if torch.sum(mask).item() != 0:
                break




        _, pred_mask, preds = self.create_preds(x, edge_index, model, mask)
        return None, pred_mask , preds


    def create_preds(self, x, edge_index, model, mask, random_edge = False):
        # GNN-Model Vorhersagen erhalten
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=-1)
        pred_labels = probs.argmax(dim=-1)
        
        # GNN Embeding erhalten
        # embed = model.get_emb(x, edge_index)

        # Graph-Klassifikation
        # ursprungswert - entspricht der Vorhersage des Modells
        probs = probs.squeeze()
        label = pred_labels

        # masked value
        # Erhalte edge_mask durch Anwenden des PGExplainer
        # _, edge_mask = explainer.explain(x, edge_index, embed=embed, tmp=1.0, training=False)
        data = Data(x=x, edge_index=edge_index)

        # Liste der "wichtigen" Knoten bestimmen auf Basis der Edge_mask bestimmen (bei hard_edge_mask)  
        if random_edge == True:    
            # Für eine edge_maske ermitteln  
            selected_nodes = self.calculate_selected_nodes(data, mask)
        else:
            # indizes ableiten für node_mask
            selected_nodes = (mask == 1).nonzero(as_tuple=False).squeeze(dim = 1)
            
        # Aussortierte Knoten
        #maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]

        maskout_nodes_list = [node for node in range(x.shape[0]) if node not in selected_nodes]
        #print(selected_nodes)
        masked_node_list = [node for node in range(x.shape[0]) if node in selected_nodes]
        #print(masked_node_list)

        value_func = self.GnnNetsGC2valueFunc(model, target_class=label)

        # Score für Teilgraph aus "wichtigen" Knoten
        masked_pred = self.gnn_score(masked_node_list, data, value_func, subgraph_building_method='zero_filling')

        # Score für Teilgraph ohne "wichtige" Knoten
        maskout_pred = self.gnn_score(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
        #maskout_pred = 0
        
        # Score für die Spärlichkeit der Erklärung
        sparsity_score = 1.0 - len(masked_node_list) / data.x.shape[0]

        # return variables
        pred_mask = [mask]
        related_preds = [{
            'masked': masked_pred,
            'maskout': maskout_pred,
            'origin': probs[label],
            'sparsity': sparsity_score}]
        return None, pred_mask, related_preds

    def calculate_selected_nodes(self, data, hard_mask):
        # if top_k != None:
        #     threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
        #     hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]
        selected_nodes = []
        edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
        selected_nodes = list(set(selected_nodes))
        return selected_nodes

    # Enclosure: Beibehaltung des Zustands mit inneren Funktionen. Value_func wird nach Aufrufen der äußeren Methode nur zurückgegeben aber nicht ausgeführt. 
    # Die Value_Func Methode speichert einen Snap-Shot der Variablen der äußeren Methode und kann daher, wenn sie aufgerufen wird, auf diese zurückgreifen
    def GnnNetsGC2valueFunc(self, gnnNets, target_class):
        '''Berenchnet Vorhersage des GNN Model für eine Zielklasse'''
        def value_func(batch):
            with torch.no_grad():
                logits = gnnNets(data=batch)
                probs = F.softmax(logits, dim=-1)
                score = probs[:, target_class]
            return score
        return value_func


    def gnn_score(self, coalition: list, data: Data, value_func: str,
              subgraph_building_method='zero_filling') -> torch.Tensor:
        """ Berechnen des Wertes eines Subgraphen bestehend aus ausgewählten Knoten"""
        num_nodes = data.num_nodes
        #subgraph_build_func = self.get_graph_build_func(subgraph_building_method)

        # Maske initialisieren (aus nullen)
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)

        # Knoten, welche im Subgraph enthalten sind, auf 1 setzen
        mask[coalition] = 1.0

        # Subgraph erstellen (x und edge_index bestimmen) als Batch-Object
        ret_x, ret_edge_index = self.graph_build_zero_filling(data.x, data.edge_index, mask)    
        mask_data = Data(x=ret_x, edge_index=ret_edge_index)
        mask_data = Batch.from_data_list([mask_data])

        # Score berechnen indem Vorhersagewahrscheinlichkeit des GNN-Models bei Input des Subgraphen bestimmt wird. Wahrscheinlichkeit der Target-Class zurückgeben
        score = value_func(mask_data)
        # get the score of predicted class for graph or specific node idx
        return score.item()

    def graph_build_zero_filling(self, X, edge_index, node_mask: np.array):
        """ Erstellung von Teilgraphen durch die Maskierung der nicht ausgewählten Knoten mit Nullen """
        ret_X = X * node_mask.unsqueeze(1)
        return ret_X, edge_index



        



    
    

