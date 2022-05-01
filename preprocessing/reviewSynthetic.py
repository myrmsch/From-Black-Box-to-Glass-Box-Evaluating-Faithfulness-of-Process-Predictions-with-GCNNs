import pm4py
import numpy as np
import pandas as pd
import torch as torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

class ReviewSynLog():
  def __init__(self, normalisation, label_encoding):
    self.normalisation = normalisation
    self.label_encoding = label_encoding
    

  def load_dataset(self, path, get_log = False):
    '''
    Datensatz laden
    Args:
      path: Speicherpfad
      get_log (bool, optional): Ob Log Instanz ausgegeben werden soll
    rtype (dataframe, log)
      log optional
    '''
    # Read Dataset
    path = str(path)
    log_a = pm4py.read_xes(path)

    # Transform to Dataframe
    df_a = pm4py.convert_to_dataframe(log_a)

    if get_log == True:
      return df_a, log_a
    return df_a

  def preprocess_data(self, df_log):
    # rename columns
    new_names = {"org:resource":"resource", "time:timestamp": "timestamp", "concept:name": "event_name", "lifecycle:transition": "transition", "case:concept:name": "review_id", "case:description": "description"}
    df_log = df_log.rename(columns = new_names)

    # nutzlose Spalten entfernen
    df_log = df_log.drop(columns="description")

    # NaNs in Result-Spalte entfernen
    df_log["result"] = df_log["result"].fillna(0)

    # Convertieren von Timestamps zu Integer Timestamps
    df_log["timestamp"] = df_log["timestamp"].apply(lambda x: int(round(x.timestamp())))

    # Label (y-Varbiable) für Process Instanzen erstellen
    df_log = df_log.groupby("review_id").apply(lambda x: self.add_labels_and_nodes(x))
    df_log.reset_index(drop=True, inplace=True)

    # review_id zu int-Typ transformieren
    df_log["review_id"] = df_log["review_id"].astype(int)
    # result Spalte enthält int und string Typen. Alle zu string transformieren als Vorbereidung für das Encoding
    df_log["result"] =df_log["result"].astype(str)
    return df_log

  def encode_data(self, df, feature, label_encoder = True, normalization = True):
    '''
    Encoded Zellenwerte in Integerwerte.
    args:
      df: DataFrame
      feature: Spaltennamen
      label_encoder (bool): Wenn Werte mit Label encoder verschlüsselt werden sollen True. Ansonsten wird OneHotEncoder und Ordinal Encoder angewendet
    '''

    # Split in X und y
    X = df.drop("result_process", axis = 1)
    y = df["result_process"]

    # Spalten, welche Encoded werden sollen
    col_trans = feature #["resource", "event_name", "transition", "result"]

    if label_encoder == True:
      X, encoder = self.label_encoding(X, col_trans)
    else:
      X, encoder = self.one_hot_encoding(X, col_trans)

    # Normalisieren der Feature Werte
    if label_encoder == True and normalization == True:
      df_feature = X[col_trans]
      scaler = MinMaxScaler()
      df_feature = pd.DataFrame(scaler.fit_transform(df_feature), 
                                        columns=df_feature.columns, index=df_feature.index)
      X[col_trans] = df_feature
      
    else:
      scaler = None

    # X = X.drop([col_trans], axis = 1)
    # X = pd.concat([X, df_feature], axis = 1)
    # 
    

    return X, y, encoder, scaler

  def label_encoding(self,X, col_trans):
    # Initialisierung der Label Encoder
    d = defaultdict(LabelEncoder)

    # Encoding the variable
    X[col_trans] = X[col_trans].apply(lambda x: d[x.name].fit_transform(x))
    return X, d

  def one_hot_encoding(self, X, col_trans):
    # Encoding
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)
    # o_encoder = OrdinalEncoder()

    # Nominale Variablen
    nominal_col = col_trans # ["event_name", "transition", "result"]
    # Ordinale Variablen
    # ordinal_col = ["Timestamp", "ProcessID"] 

    # Encoding Ordinale Spalten
    # X[ordinal_col] = pd.DataFrame(o_encoder.fit_transform(X[ordinal_col]))

    # One Hot Encoding anwenden
    t = pd.DataFrame(oh_encoder.fit_transform(X[nominal_col[:]]))

    # Index reparieren
    t.index = X.index

    # Rename Columns
    t.columns = oh_encoder.get_feature_names(nominal_col[:])

    # Nominale Spalten entfernen und mit den Encoded Spalten ersetzen
    X.drop(nominal_col[:], axis = 1, inplace= True)

    # Zusammenführen der Dataframes
    X_encoded = pd.concat([X, t], axis=1)
    return X_encoded, oh_encoder



  def add_labels_and_nodes(self, group): 
    '''
    Label erstellen, welche ein positives oder negatives Ergebniss des Review Prozesses darstellen. Ergebniss aus X-Daten entfernen. 
    Gruppen welche die benötigten Events nicht enthalten, werden aus dem Datensatz aussortiert
    Positives Event: accept
    Negatives Event: reject
    ''' 
    # Start- und Endknoten einfügen
    group = self.add_nodes(group)
    if "accept" in group.event_name.values:
      group["result_process"] = 1
      #group = group[group.event_name != "accept"]
      #index_names = group[ (group['event_name'] >= "accept")].index
      #group.drop(index_names, inplace=True)
      group = group[group.event_name != "accept"]
      return group
    elif "reject" in group.event_name.values:
      group["result_process"] = 0
      #index_names = group[ (group['event_name'] >= "reject")].index
      #group.drop(index_names, inplace=True)
      group = group[group.event_name != "reject"]
      return group

  def add_nodes(self, g):
    g.reset_index(drop=True, inplace = True)
    ts_0 = g.iloc[0]["timestamp"]
    ts_1 = g.iloc[-1]["timestamp"]
    review_id = g.iloc[0]["review_id"]
    
    g.loc[-1] = ["__INVALID__",ts_0 - 1,"start",'start',review_id, "0"]  # Start-Knoten hinzufügen
    g.loc[len(g.index)] = ["__INVALID__",ts_1 +1,"end",'start',review_id, "0"] # End-Knoten hinzufügen
    g.index = g.index + 1  # Index anpassen
    g.sort_index(inplace=True) 
    return g

  def visualize_graph(self, data, scaler, feature, d = None):
    """
    Methode zum Visualisieren eines Prozess Instanz mittels Networkx.
    Args:
      data (Data): Instanz
      d (dict): Dictionary mit LabelEncoder der Feature
      scaler: Verwendeter Scaler
    Return:
      nx: networkx
    """
    # Netzwerkx Objekt erstellen
    G = to_networkx(data)

    # Array aller Knoten erstellen
    if data.num_features > 1:
      nodes = data.x.squeeze(1).tolist()
    else:
      nodes = data.x.tolist()
    
    # Normalisierung Rückgängig machen und Node-Werte von float zu int transformieren
    if self.normalisation == True:
      nodes = scaler.inverse_transform(nodes)
      nodes = np.around(np.array(nodes))
    # else:
    #   nodes = [int(x) for x in nodes]                                        

    # Encoder 
    # encoder_name = d['event_name']
    # encoder_transition = d["transition"]
    # encoder_result = d["result"]
    # encoder_timestamp = d["timestamp"]

    #ursprünglichen Wert mit Code als Dictionary erhalten
    codes = []
    if d != None:
      for encoder in d:
        code = self.get_integer_mapping(d[encoder])
        codes.append(code)

      # code_names = self.get_integer_mapping(encoder_name)
      # code_transitions = self.get_integer_mapping(encoder_transition)
      # code_result = self.get_integer_mapping(encoder_result)
      # #code_timestamp = self.get_integer_mapping(encoder_timestamp)
      # #print(code_names)

      # Knoten mit zugehörigen dekotierten Features 
      # codes = [code_names, code_transitions, code_result]
      n_labels = self.get_label(nodes, codes)
      #print(n_labels)
    else:
      # Wenn kein Encoder übergeben wurde, werden die Nummern zur graphischen Darstellung verwendet
      n_labels = {}
      for n_id, node in enumerate(nodes):
        n_labels[n_id] = node


    # Darstellen des Netzwerks
    pos = nx.kamada_kawai_layout(G)  
    fig = plt.figure(figsize=(15, 8))
    plt.axis('off')

    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, arrows = True )
    nx.draw_networkx_labels(G,pos, labels=n_labels)
    plt.show()

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

  


class CreateDatasetReview(InMemoryDataset):
  def __init__(self, root, df, feature, device, transform=None, pre_transform=None):
      self.df = df
      self.feature_nodes = feature
      self.device = device
      super(CreateDatasetReview, self).__init__(root, transform, pre_transform)
      self.data, self.slices = torch.load(self.processed_paths[0])
      

  @property
  def raw_file_names(self):
      return []
  @property
  def processed_file_names(self):
      return ['./review_dataset_startnodes.dataset'] #'./xgnn_ppm.dataset'
      

  def download(self):
      pass
    
  # def get_node_ids(self):
  #     return self.node_ids

  def process(self):
    
    data_list = []

    # Definieren ob GPU oder CPU verwendet wurde
    device = self.device

    # process by session_id
    grouped = self.df.groupby('review_id')

    # Define group to be explained in the TestSet
    # X_test = df_log_a[df_log_a["review_id"]==8]
    
    for reviewId, group in tqdm(grouped):           
        # Neue Ids für items in einer Session (von 0 startend)
                  
        features = self.feature_nodes

        # Node_id identifiziert Knoten eindeutig. Knoten mit gleichen Features erhalten die selbe -node_id-
        ids =  pd.factorize(group[features].apply(tuple, axis=1))[0] 
        group["node_id"] = ids
        #self.node_ids.append(ids)
                  
        # Array aller Knoten mit Features erstellen: Feature sind durch die Variable -features- definiert. Duplikate werden entfernt. -node_id- Spalte wird ebenfalls entfernt, da diese nicht als Feature einbezogen wird
        node_features = group.loc[group.review_id==reviewId,["node_id"]+ features].sort_values('node_id').drop_duplicates(subset=["node_id"]).drop(columns = ["node_id"]).values
        # node_features = group.loc[group.session_id==session_id,['sess_item_id','item_id']].sort_values('sess_item_id').drop_duplicates(subset = ["sess_item_id"]).values 
        
        #Tensor mit Knoten erstellen
        node_features = torch.from_numpy(node_features).float().to(device) #.unsqueeze(1)

        # Ziel und Start Knoten definieren: TODO: Timestamps einbeziehen und parallele Prozesse darzustellen
        target_nodes = group.node_id.values[1:]
        source_nodes = group.node_id.values[:-1]
        x = node_features
        
        # -edge_index-: Tensor bestehend aus zwei gleich langen Listen, welche Verbindungen zwischen Knoten definieren. Tensor([start_knoten_id][ziel_knoten_id])
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)            
        
        # Tensor mit Label des Graphen (Shape: (1,))
        y = torch.cuda.FloatTensor([group.result_process.values[0]])
        #y = torch.cuda.FloatTensor(label)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])
    #return data_list