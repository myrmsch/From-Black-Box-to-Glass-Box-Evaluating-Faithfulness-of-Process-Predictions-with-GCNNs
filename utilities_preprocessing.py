
from torch_geometric.utils import to_networkx
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
import numpy as np
from torch_geometric.utils import to_networkx
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import DataLoader, Data
from torch import Tensor

# Hilfs-Methoden und Klasse für Preprocessing
# Darstellung von Graphen

class GraphDatasetHelper():
    def __init__(self, data = None, dic_encoder = None, scaler = None, node_feature = None) -> None:
        self.dic_encoder = dic_encoder          # Falls Daten mit einem Label Encoder codiert wurden. Angepasset Encoder wird benötigt, um Daten zu Ursprungswerten zu transformieren
        self.scaler = scaler                    # Falls Daten skaliert wurden. Angepasster Scaler wird benötigt, um Daten zu Ursprungswerten zu transformieren 
        self.data = data                        # Prozess-Instanz
        self.node_feature = node_feature        # Werden benötigt, um OneHotEncoded Feature zu decoden

    

    def visualise_dataset(self, data):        
        ''' Visualisierung einer Graphen-Instanz mit Knoten-Featurn'''
        self.data = data
        dic_encoder = self.dic_encoder 
        scaler = self.scaler 

        # Netzwerkx Objekt erstellen
        G = to_networkx(data)

        # Array aller Knoten erstellen
        if data.num_features > 1:
            nodes = data.x.squeeze(1).tolist()
        else:
            nodes = data.x.tolist()
        
        # Normalisierung Rückgängig machen und Node-Werte von float zu int transformieren
        if scaler != None:
            nodes = scaler.inverse_transform(nodes)
            nodes = np.around(np.array(nodes))
     

        # Encoding Rückgängig machen
        codes = []
        # Label-Encoding
        if dic_encoder != None:     
            #ursprünglichen Wert mit durch das Label-Encoding vergebenen Code als Dictionary erhalten
            for encoder in dic_encoder:
                code = self.get_integer_mapping(dic_encoder[encoder])
                codes.append(code)

            n_labels = self.get_label(nodes, codes)
        # One-Hot-Encoding
        else:
            # One-Hot-Encoder, Namen durch Multiplikation mit Feature-Vector erhalten
            n_labels = self.get_labels_ohe(nodes, self.node_feature)


        # Darstellen des Netzwerks
        pos = nx.kamada_kawai_layout(G)  
        fig = plt.figure(figsize=(30, 16))
        plt.axis('off')

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, arrows = True )
        nx.draw_networkx_labels(G,pos, labels=n_labels)
        plt.show()

    def fast_visualisation(self, data, title = None, size = None):
        '''schnelle Visualisierung eines Graphen. Knoten ohne Knoten-Feature'''
        # Netzwerkx Objekt erstellen
        G = to_networkx(data)

        # Array aller Knoten erstellen
        nodes = data.x.squeeze(1).tolist()                                       

        # Darstellen des Netzwerks
        pos = nx.kamada_kawai_layout(G) 
        if size ==  None: 
            fig = plt.figure(figsize=(15, 8))
        else:
            fig = plt.figure(figsize=size)

        plt.axis('off')

        if title != None: 
            plt.title(title)


        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, arrows = False )
        #nx.draw_networkx_labels(G,pos, labels=n_labels)
        plt.show()
        return plt

    def get_integer_mapping(self, le):
        '''
        Return a dict mapping labels to their integer values
        from an SKlearn LabelEncoder
        le = a fitted SKlearn LabelEncoder
        '''
        res = {}
        for cl in le.classes_:
            res.update({cl:le.transform([cl])[0]})

        return res

    # Erhalten der durch das LabelEncoding verschlüsselten ursprünglichen Werte. Output: {node_id: [feature]}
    def get_label(self, nodes, codes):
        node_label = {}
        for n_id, node in enumerate(nodes):
            translations = []
            for id, feature in enumerate(node):  
                #if id != 3:    
                for key, value in codes[id].items():
                    if feature == value:
                        translations.append(key)
            node_label[n_id] =translations
                #else:
                #node_label.append("error: " + str(node.item()))

        return node_label
    
    def get_labels_ohe(self, nodes, node_features):   
        '''
        Label der Knoten erhalten. OHE Encoding wird in Ursprungs-Bezeichnungen zurück transformiert
        '''
      
        n_labels = {}
        for n_id, node in enumerate(nodes):
            # Float zu Int transformieren
            node = list(map(int, node))
            # Erhalten der Benennungen durch Multiplizieren der node_feature-Liste mit der Encoded node-Liste. 
            # Durch OHE ist Wert eines Features [1,0]. Ergebnis ist eine Liste aller Feature. Output: {node_id: [feature]}
            n_labels[n_id] = sum([[s] * n for s, n in zip(node_features, node)], [])
        return n_labels

   
    def visualise_training(self, train_results, title = "Accuracy in Abhängigkeit zur Trainings-Epoche", x_axis_title = "Accuracy" ):
        '''
        Training der GCN-Methoden visualisieren
        '''
        # number of employees of A
        acc = [x[1] for x in train_results]
        epochs = [x[0] for x in train_results]


        # plot a line chart
        plt.plot(epochs, acc, 'o-g')
        # plt.plot(epochs, fidelity_inv_values, 'o-b')

        # set axis titles
        plt.xlabel("Epochen")
        plt.ylabel(x_axis_title)
        # set chart title
        plt.title(title)
        #plt.legend(['Fidelity+', 'Fidelity-'])
        plt.show()

    def visualise_event_based(self, not_ohe_columns):

        ''' 
        Visualisierung einer Graphen-Instanz mit Knoten-Featurn der nach dem Event-Based Ansatz vorverarbeitet wurde. 
        Hier sind nicht alle Feature OHE. Es werden die Werte der OHE im Knoten angegeben. Für die Nicht OHE-Feature werden Lückenfüller eingesetzt.
        :param not_ohe_columns: Anzahl der nicht OHE Spalten
        '''
        # self.data = data
        # dic_encoder = self.dic_encoder 
        # scaler = self.scaler 

        # Netzwerkx Objekt erstellen
        G = to_networkx(self.data)

        # Array aller Knoten erstellen
        if self.data.num_features > 1:
            nodes = self.data.x.squeeze(1).tolist()
        else:
            nodes = self.data.x.tolist()
   

        # One-Hot-Encoder, Namen durch Multiplikation mit Feature-Vector erhalten
        n_labels = self.get_labels_ohe_event_based(nodes, self.node_feature, not_ohe_columns)


        # Darstellen des Netzwerks
        pos = nx.kamada_kawai_layout(G)  
        fig = plt.figure(figsize=(30, 16))
        plt.axis('off')

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, arrows = True )
        nx.draw_networkx_labels(G,pos, labels=n_labels)
        plt.show()
    
    def get_labels_ohe_event_based(self, nodes, node_features, empty_columns):   
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
    

  