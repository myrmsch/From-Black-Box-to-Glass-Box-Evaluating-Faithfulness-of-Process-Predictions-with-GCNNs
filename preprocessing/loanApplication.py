
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import pm4py

class LoanApplicationDataset(InMemoryDataset):
    def __init__(self, root, df, feature, transform=None, pre_transform=None):
        self.df = df
        self.feature = feature
        super(LoanApplicationDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./xgnn_ppm.dataset']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []

        # process by session_id
        grouped = self.df.groupby('ProcessID')

        # Define group to be explained in the TestSet
        # X_test = df_log_a[df_log_a["ProcessID"]==8]
        
        for ProcessID, group in tqdm(grouped):           
            # Neue Ids für items in einer Session (von 0 startend)
                      
            #features = ["Events","LoanGoal"]
            ids =  pd.factorize(group[self.feature].apply(tuple, axis=1))[0] 
            group["NodeId"] = ids
            #self.node_ids.append(ids)                     
            
            node_features = group.loc[group.ProcessID==ProcessID,["NodeId"]+ self.feature].sort_values('NodeId').drop_duplicates(subset=["NodeId"]).drop(columns = ["NodeId"]).values
            # node_features = group.loc[group.session_id==session_id,['sess_item_id','item_id']].sort_values('sess_item_id').drop_duplicates(subset = ["sess_item_id"]).values 
            
            #create nodes
            node_features = torch.from_numpy(node_features).float().to(device) #.unsqueeze(1)
            target_nodes = group.NodeId.values[1:]
            source_nodes = group.NodeId.values[:-1]
            x = node_features            
            
            # create edges
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)            
            
            y = torch.cuda.FloatTensor([group.Label.values[0]])
            #y = torch.cuda.FloatTensor(label)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class LoanApplication():
    def __init__(self):
      # End Events:
      self.process_end = ["A_Pending", "A_Denied", "A_Cancelled"]

    def load_dataset(self, path):
      # Read Dataset
      # log_o = pm4py.read_xes(path + 'BPI_Challenge_2017_Offer.xes')
      log_a = pm4py.read_xes(path + "BPI_Challenge_2017.xes")

      # Transform to Dataframe
      # pd_o = pm4py.convert_to_dataframe(log_o)
      pd_a = pm4py.convert_to_dataframe(log_a)

      return pd_a

    def preprocess_dataset(self, df):
      # Rename Columns
      df.rename(columns={"concept:name":"Events", "case:concept:name":"ProcessID", "time:timestamp": "Timestamp", "case:LoanGoal": "LoanGoal" }, inplace=True)

      # Select columns for Modell training
      df = df[["Events","ProcessID","Timestamp","LoanGoal"]]

      # End Events: Sagen aus, ob Credit gewährt wurde oder nicht. Werden aus Process Entfernt und als Label verwedent
      # Siehe Analyse der End Events weiter unten: Löschen der Gruppe mit zwei A_Denied Events 
      # Anzahl der Reihen mit einem Event in einer Gruppe
      g3 = df.groupby(['ProcessID', 'Events']).agg(number_of_rows=('Events', 'count')).reset_index()
      # ProcessID der Gruppe erhalten
      a = g3[(g3["number_of_rows"] >= 2) & (g3["Events"] == self.process_end[1])]["ProcessID"].values[0]
      # Aus Dataframe löschen
      df = df[df["ProcessID"] != a]
      # Spalte mit Label der Bewerbungsprozesse (Gruppen) erstellen
      df = df.groupby("ProcessID").apply(lambda x: self.add_label(x))
      df.reset_index(drop = True, inplace = True )



      
      # create Pytorch Dataset 
      # dataset = LoanApplicationDataset(root="../")

      return df

    def encode_data(self, df):
      # Split in X und y 
      X = df.drop("Label", axis =1)
      y = df["Label"]

      # Encoding
      oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)
      o_encoder = OrdinalEncoder()

      # Nominale Variablen
      nominal_col = ["Events", "LoanGoal"]
      # Ordinale Variablen
      ordinal_col = ["Timestamp", "ProcessID"] 

      # Encoding Ordinale Spalten
      X[ordinal_col] = pd.DataFrame(o_encoder.fit_transform(X[ordinal_col]))

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
      return X_encoded,y

      

    def visualize_dataset(self):
      pass

    # Methode um jeder Gruppe das passende Label zu zuordnen
    def add_label(self, group):  
      if self.process_end[0] in group.Events.values:
        group["Label"] = 0
        group = group[group.Events != self.process_end[0]]
        return group
      elif self.process_end[1] in group.Events.values:
        group["Label"] = 1
        group = group[group.Events != self.process_end[1]]
        return group
      elif self.process_end[2] in group.Events.values:
        group["Label"] = 2
        group = group[group.Events != self.process_end[2]]
        return group

