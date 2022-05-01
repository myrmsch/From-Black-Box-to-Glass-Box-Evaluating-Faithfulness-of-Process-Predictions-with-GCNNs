
from torch_sparse import SparseTensor
from torch.nn import Linear, ModuleList
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
from dig.xgraph.models import GCNConv

import torch_geometric.nn as gnn 
from torch_geometric.nn import global_mean_pool
from torch_geometric.data.batch import Batch
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
import torch

## Eigene GCNConv Klasse, da diese für GNN-LRP angepasst werden musste
class GCNConv_n(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)


        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out



class GCN3(torch.nn.Module):
    def __init__(self, hidden_channels, input_dim, num_layer):
        super().__init__()    #GCN, self
        dim_node = input_dim
        dim_hidden = hidden_channels
        num_layer = num_layer
        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.lin = Linear(dim_hidden, 2)

        self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, 2)]
        ))

        self.dropout = nn.Dropout()

        #self.emb = 0

    # def setbatch(self, batch):
    #     self.batch = batch

    def forward(self, *args, **kwargs): #x, edge_index, batch

        # Input Werte auslesen
        x, edge_index, batch = self.get_data(*args, **kwargs)
  
        # 1. Obtain node embeddings 
        
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
      
        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        x = post_conv
          
        
        #self.emb = x
        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training = self.training)
        # x = self.lin(x)

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)       
        
        return out

    def get_data(self, *args, **kwargs):
      '''
      von https://github.com/divelab/DIG/blob/dig/dig/xgraph/models/models.py
      Ermöglicht dem Model verschiedene Input Parameter zu verarbeiten
      '''
      data: Batch = kwargs.get("data") or None

      if not data:
          if not args:             # Werte direkt x, edge-index und batch zugewiesen
              assert 'x' in kwargs
              assert 'edge_index' in kwargs
              x, edge_index = kwargs['x'], kwargs['edge_index'],
              batch = kwargs.get('batch')
              if batch is None:
                  batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
          elif len(args) == 2:      # Nur x und edge_index als Input übergeben. Batch wird manuel berechnet
              x, edge_index = args[0], args[1]
              batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
          elif len(args) == 3:      # x, edge_index und bath als Input übergeben
              x, edge_index, batch = args[0], args[1], args[2]
          else:
              raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
      else:                         # Data-Objekt als Input übergeben
          x, edge_index, batch = data.x, data.edge_index, data.batch

      return x, edge_index, batch

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
      '''
      Auslesen der Node Embeddings
      '''
      x, edge_index, batch = self.get_data(*args, **kwargs)
      
      # 1. Node Embeddings erhalten
      post_conv = self.relu1(self.conv1(x, edge_index))
      for conv, relu in zip(self.convs, self.relus):
          post_conv = relu(conv(post_conv, edge_index))
          
      return post_conv

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)