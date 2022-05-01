
import torch
import joblib
import matplotlib.pyplot as plt 
import random
from torch_geometric.data.batch import Batch

# Methoden, welche zum Generieren von Erklärungen der die in DIG implementierten Explainer verwendet werden. 
# Methoden zur Visualisierung der Evaluations-Metriken in Form von Diagrammen

def train_pgexplainer(explainer_pge, pgexplainer_training, train_set, path, pgexplainer_training_name):  
  '''
  PGExplainer trainieren
  :param explainer_pge: PGExplainer
  :pgexplainer_training: Trainieren (True), ansonsten laden
  :train_set: Datensatz zum trainieren
  :path: Pfad
  :pgexplainer_training_name: Speichername
  '''
  path_pgexplainer =  path / "xai_methods/pgexplainer/trainiert" # Speicherort für Parameter des PGExplainer
  #Trainieren des Explainers
  if pgexplainer_training == True:
    print("-------------- Training PGExplainer------------------")
    explainer_pge.train_explanation_network(train_set)
    # Speichern der Parameter des trainierten Explainer
    torch.save(explainer_pge.state_dict(), str(path_pgexplainer / pgexplainer_training_name))
    print("-------------- Training PGExplainer Beendet------------------")
  else: 
    # Laden der Paremeter des trainierten Explainers
    state_dict = torch.load(str(path_pgexplainer / pgexplainer_training_name))
    explainer_pge.load_state_dict(state_dict)
  return explainer_pge

def pgexplainer_run(data, collector_list_pgexplainer, explainer_pge, steps, sparsity_add = 0, pge_utilities=None):
  '''
  Generieren von Erklärungen mittels GNNExplainer und speichern dieser im Collector. Es werden Erklärungen für mehrere Sparsity-Werte generiert
  :param collector_list_pgexplainer: Speichern der Collectoren
  :param explainer_random: Explainer
  :param steps: Liste der Sparsity-Werte
  :param sparsity_add: Sparsity kann nicht direkt als Parameter übergeben werden, sondern muss in top-k Wert umgerechnet werden. Über sparsity_add kann dieser beeinflusst werden
  :param pge_utilities: Fehler im DIG package wurde selbst behoben. Package mittlerweile aktualisiert
  '''

  x = data.x 
  #y = int(data.y.item())    
  edge_index = data.edge_index

  #Speichern der Werte in XCollector
  with torch.no_grad():
    for id, xCollector_pgexplainer in enumerate(collector_list_pgexplainer):
      
      # Sparsity des x_Collectors bestimmen
      sparsity = round(steps[id],1)     
      # Berechnen des Top-K Wertes auf Basis der vordefinierten Sparsity. Es wird immer aufgerundet
      top_k  = int(data.x.shape[0] * (1-sparsity)) + sparsity_add
 
      #create_explanations gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden.
      #_, masks, related_preds = pge_utilities.create_explanations(x, edge_index, top_k, model, explainer_pge)
      _, soft_masks, related_preds = explainer_pge(x, edge_index, top_k = top_k)
      # Entfernt durch PyTorch gespeicherte Zusatzinformationen
      soft_masks = [soft_mask.detach() for soft_mask in soft_masks]
      xCollector_pgexplainer.collect_data(soft_masks, related_preds)
      collector_list_pgexplainer[id] = xCollector_pgexplainer
  return collector_list_pgexplainer

def gnnexplainer_run(data_t, collector_list_gnnexplainer, explainer_gnnexplainer, steps):  
  '''
  Generieren von Erklärungen mittels GNNExplainer und speichern dieser im Collector. Es werden Erklärungen für mehrere Sparsity-Werte generiert
  :param collector_list_gnnexplainer: Speichern der Collectoren
  :param explainer_random: Explainer
  :param steps: Liste der Sparsity-Werte
  '''
  x = data_t.x
  y = int(data_t.y.item())  
  edge_index = data_t.edge_index

  for id, xCollector_gnnexplainer in enumerate(collector_list_gnnexplainer):
    sparsity = round(steps[id],1)  
    #Speichern der Werte in XCollector
    # gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden.    
    _, masks, related_preds = explainer_gnnexplainer(x, edge_index, num_classes = 2, sparsity = sparsity)
    masks = [mask.detach() for mask in masks]
    xCollector_gnnexplainer.collect_data(masks, related_preds, y)
    collector_list_gnnexplainer[id] = xCollector_gnnexplainer
  return collector_list_gnnexplainer

def gradcam_run(data_t, collector_list_gradcam, explainer_gradcam, steps):
  '''
  Generieren von Erklärungen mittels Grad-Cam und speichern dieser im Collector. Es werden Erklärungen für mehrere Sparsity-Werte generiert
  :param collector_list_gradcam: Speichern der Collectoren
  :param explainer_random: Explainer
  :param steps: Liste der Sparsity-Werte
  '''

  # Graph-Instanz 
  x = data_t.x
  y = int(data_t.y.item())  
  edge_index = data_t.edge_index

  results= []
  # Iterieren durch die xCollectoren für verschiedene Sparsity Werte
  for id, xCollector_grad_cam in enumerate(collector_list_gradcam):
    # Sparsity des x_Collectors bestimmen
    sparsity = round(steps[id],1)     

    with torch.no_grad():    
    # gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden.
      _, hard_masks, related_preds = explainer_gradcam(x, edge_index, sparsity=sparsity, num_classes=2)
      hard_masks = [hard_mask.detach() for hard_mask in hard_masks]
    results.append(related_preds[y])

    # Mask und Vorhersagen im xCollector speichern
    xCollector_grad_cam.collect_data(hard_masks, related_preds, y)    
    collector_list_gradcam[id] = xCollector_grad_cam    
  return collector_list_gradcam, results

def random_node_run(data_t, collector_list_random_node, explainer_random, steps, model):
  '''
  Generieren von Erklärungen mittels zufällig gewählter Knoten und speichern dieser im Collector. Es werden Erklärungen für mehrere Sparsity-Werte generiert
  :param collector_list_random_node: Speichern der Collectoren
  :param explainer_random: Explainer
  :param steps: Liste der Sparsity-Werte
  '''
  #print("Ausführung des GNNExplainers") 

  # Graph-Instanz 
  x = data_t.x
  # y = int(data_t.y.item())  
  edge_index = data_t.edge_index
 
  # Iterieren durch die xCollectoren für verschiedene Sparsity Werte
  for id, xCollector_random_node in enumerate(collector_list_random_node):
    # Sparsity des x_Collectors bestimmen
    sparsity = round(steps[id],1)     

    with torch.no_grad():        
    # gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden.
      _, hard_masks, related_preds = explainer_random.explain(x, edge_index, sparsity, model, random_edge = False)
      #masks = [mask.detach() for mask in masks]

    # Mask und Vorhersagen im xCollector speichern
    xCollector_random_node.collect_data(hard_masks, related_preds)
    collector_list_random_node[id] = xCollector_random_node    
  return collector_list_random_node

def random_node_prefix_run(data_t, collector, explainer_random, sparsity, model):   

    # Graph-Instanz 
    x = data_t.x
    # y = int(data_t.y.item())  
    edge_index = data_t.edge_index 

    with torch.no_grad():        
      # gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden.
      _, node_mask, related_preds = explainer_random.explain(x, edge_index, sparsity, model, random_edge = False)     
    collector.collect_data(node_mask, related_preds)
    #collector_list_random_node[id] = xCollector_random_node    
    return collector

def gnnexplainer_prefix_run(data_t, collector, explainer_gnnexplainer, sparsity):
    # Variablen
    x = data_t.x
    y = int(data_t.y.item())  
    edge_index = data_t.edge_index
    # gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden. 
    _, masks, related_preds = explainer_gnnexplainer(x, edge_index, num_classes = 2, sparsity = sparsity)
    masks = [mask.detach() for mask in masks]
    collector.collect_data(masks, related_preds, y)
    return collector

def gradcam_prefix_run(data_t, collector, explainer_gradcam, sparsity):

  # Graph-Instanz 
  x = data_t.x
  y = int(data_t.y.item())  
  edge_index = data_t.edge_index   

  with torch.no_grad():        
  # gibt für Prozess-Instanzen die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden.
    _, masks, related_preds = explainer_gradcam(x, edge_index, sparsity=sparsity, num_classes=2)
  collector.collect_data(masks, related_preds, y)   
  return collector

def pgexplainer_prefix_run(data, collector, explainer_pge, sparsity, sparsity_add = 0):

  x = data.x 
  y = int(data.y.item())    
  edge_index = data.edge_index
 
  with torch.no_grad():
    # top_k Wert erhalten
    top_k  = int(data.x.shape[0] * (1-sparsity)) + sparsity_add
    #gib die edge-Masks sowie die Scores für die Vorhersagen aus. Diese können dem XCollector übergeben werden
    _, masks, related_preds = explainer_pge(x, edge_index, top_k = top_k)
    # print(related_preds)
  #masks = [mask.detach() for mask in masks]
  collector.collect_data(masks, related_preds)

  return collector





def plot_results(collector_list_gradcam, collector_list_gnnexplainer, collector_list_pgexplainer, collector_list_random_node, title, path, x_axis = 1, y_axis = -0.1):
  '''
  Ergebnisse der Evaluation in einem Balken-Diagramm darstellen
  '''
  # Ergebnisse werden in Linien-Diagrammen dargestellt
  #sparsity_plt = steps

  # Fidelity+ Werte in Liste
  sparsity_fidelity_grad_cam = [x.fidelity for x in collector_list_gradcam]
  sparsity_fidelity_gnnexplainer = [x.fidelity for x in collector_list_gnnexplainer]
  sparsity_fidelity_pgexplainer = [x.fidelity for x in collector_list_pgexplainer]
  sparsity_fidelity_random_node = [x.fidelity for x in collector_list_random_node]

  # Fidelity- Werte in Liste
  sparsity_fidelity_inv_grad_cam = [x.fidelity_inv for x in collector_list_gradcam]
  sparsity_fidelity_inv_gnnexplainer = [x.fidelity_inv for x in collector_list_gnnexplainer]
  sparsity_fidelity_inv_pgexplainer = [x.fidelity_inv for x in collector_list_pgexplainer]
  sparsity_fidelity_inv_random_node = [x.fidelity_inv for x in collector_list_random_node]

  # Sparsity Werte in Liste
  sparsity_grad_cam = [x.sparsity for x in collector_list_gradcam]
  sparsity_gnnexplainer = [x.sparsity for x in collector_list_gnnexplainer]
  sparsity_pgexplainer = [x.sparsity for x in collector_list_pgexplainer]
  sparsity_random_node = [x.sparsity for x in collector_list_random_node]


  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
  fig.suptitle(title)
  ax1.set_ylim(bottom=y_axis, top=x_axis)
  ax2.set_ylim(bottom=y_axis, top=x_axis)

  # plot two lines
  ax1.plot(sparsity_grad_cam, sparsity_fidelity_grad_cam, 'o-g')
  ax1.plot(sparsity_gnnexplainer, sparsity_fidelity_gnnexplainer, 'o-r')
  ax1.plot(sparsity_pgexplainer, sparsity_fidelity_pgexplainer, 'o-m')
  ax1.plot(sparsity_random_node, sparsity_fidelity_random_node, 'o-y')

  ax2.plot(sparsity_grad_cam, sparsity_fidelity_inv_grad_cam, 'o-g')
  ax2.plot(sparsity_gnnexplainer, sparsity_fidelity_inv_gnnexplainer, 'o-r')
  ax2.plot(sparsity_pgexplainer, sparsity_fidelity_inv_pgexplainer, 'o-m')
  ax2.plot(sparsity_random_node, sparsity_fidelity_inv_random_node, 'o-y')

  # set axis titles
  ax1.set(xlabel='Sparsity', ylabel='Fidelity+')
  ax2.set(xlabel='Sparsity', ylabel='Fidelity-')

  # legend
  fig.legend(['Grad Cam', "GNNExplainer", "PGExplainer",  "Random-Node"]) 

  # Save results
  results = [
           [sparsity_fidelity_grad_cam, sparsity_fidelity_inv_grad_cam, sparsity_grad_cam],
           [sparsity_fidelity_gnnexplainer,sparsity_fidelity_inv_gnnexplainer,sparsity_gnnexplainer],
           [sparsity_fidelity_pgexplainer, sparsity_fidelity_inv_pgexplainer, sparsity_pgexplainer],
           [sparsity_fidelity_random_node, sparsity_fidelity_inv_random_node, sparsity_random_node]
  ]
  joblib.dump(results, path / "evaluation_results.joblib")  
  plt.savefig(path / "plot")
  plt.show()

def plot_results_prefix(datalist, title, path, \
  sparsity_fidelity_grad_cam, sparsity_fidelity_inv_grad_cam, \
    sparsity_fidelity_gnnexplainer, sparsity_fidelity_inv_gnnexplainer, \
      sparsity_fidelity_pgexplainer, sparsity_fidelity_inv_pgexplainer, \
        sparsity_fidelity_random_node, sparsity_fidelity_inv_random_node  ):
  '''
  Ergebnisse der Evaluation der Präfix-Buckets in einem Balken-Diagramm darstellen. 
  '''
  pref = [key for key in datalist]
  #pref = pref[:-1]

  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
  fig.suptitle(title)
  ax1.set_ylim(bottom=-0.2, top=0.5)
  ax2.set_ylim(bottom=-0.2, top=0.5)

  # plot two lines
  ax1.plot(pref, sparsity_fidelity_grad_cam, 'o-g')
  ax1.plot(pref, sparsity_fidelity_gnnexplainer, 'o-r')
  ax1.plot(pref, sparsity_fidelity_pgexplainer, 'o-m')
  ax1.plot(pref, sparsity_fidelity_random_node, 'o-y')

  ax2.plot(pref, sparsity_fidelity_inv_grad_cam, 'o-g')
  ax2.plot(pref, sparsity_fidelity_inv_gnnexplainer, 'o-r')
  ax2.plot(pref, sparsity_fidelity_inv_pgexplainer, 'o-m')
  ax2.plot(pref, sparsity_fidelity_inv_random_node, 'o-y')

  # set axis titles
  ax1.set(xlabel='Länge der Präfixe', ylabel='Fidelity+')
  ax2.set(xlabel='Länge der Präfixe', ylabel='Fidelity-')


  # legend
  fig.legend(['Grad Cam', "GNNExplainer", "PGExplainer",  "Random-Node"])
  plt.show()
  plt.savefig(path / "plot")



  # legend
  fig.legend(['Grad Cam', "GNNExplainer", "PGExplainer",  "Random-Node"])
  plt.show()
  plt.savefig(path / "plot")

def load_eval_results(path_results, title = None):
  '''
  Gespeicherte Ergebnisse können geladen werden. Wird ein Titel mitangegeben, wird direkt das Diagramm gezeichnet
  :param path_results: Speicherort der Ergebnisse
  :param title (optional): Name des zu zeichnenden Diagramms
  '''
  # laden
  results = joblib.load(path_results /"evaluation_results.joblib") 
  grad_cam_results = results[0]
  gnnexplainer_results = results[1]
  pgexplainer_results = results[2]
  ranodm_node_results = results[3]
  # Wenn Titel mit angegeben, Diagramm zeichnen
  if title != None:
    plot_results_load(title, path_results, grad_cam_results[0], grad_cam_results[1], grad_cam_results[2], \
      gnnexplainer_results[0], gnnexplainer_results[1], gnnexplainer_results[2], \
        pgexplainer_results[0], pgexplainer_results[1], pgexplainer_results[2], \
          ranodm_node_results[0], ranodm_node_results[1], ranodm_node_results[2])
    return grad_cam_results, gnnexplainer_results, pgexplainer_results, ranodm_node_results
  else:
    return grad_cam_results, gnnexplainer_results, pgexplainer_results, ranodm_node_results

def plot_results_load(title, path, \
  sparsity_fidelity_grad_cam, sparsity_fidelity_inv_grad_cam, sparsity_grad_cam, \
    sparsity_fidelity_gnnexplainer, sparsity_fidelity_inv_gnnexplainer, sparsity_gnnexplainer, \
      sparsity_fidelity_pgexplainer, sparsity_fidelity_inv_pgexplainer, sparsity_pgexplainer, \
        sparsity_fidelity_random_node, sparsity_fidelity_inv_random_node, sparsity_random_node, x_axis = 1, y_axis = -0.1  ):
  '''
  Stellt die Ergebnisse der Evaluation dar, nachem sie durch die load_eval_results-Methode geladen wurden. 
  '''
  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
  fig.suptitle(title)
  ax1.set_ylim(bottom=y_axis, top=x_axis)
  ax2.set_ylim(bottom=y_axis, top=x_axis)

  # plot two lines
  ax1.plot(sparsity_grad_cam, sparsity_fidelity_grad_cam, 'o-g')
  ax1.plot(sparsity_gnnexplainer, sparsity_fidelity_gnnexplainer, 'o-r')
  ax1.plot(sparsity_pgexplainer, sparsity_fidelity_pgexplainer, 'o-m')
  ax1.plot(sparsity_random_node, sparsity_fidelity_random_node, 'o-y')

  ax2.plot(sparsity_grad_cam, sparsity_fidelity_inv_grad_cam, 'o-g')
  ax2.plot(sparsity_gnnexplainer, sparsity_fidelity_inv_gnnexplainer, 'o-r')
  ax2.plot(sparsity_pgexplainer, sparsity_fidelity_inv_pgexplainer, 'o-m')
  ax2.plot(sparsity_random_node, sparsity_fidelity_inv_random_node, 'o-y')

  # set axis titles
  ax1.set(xlabel='Sparsity', ylabel='Fidelity+')
  ax2.set(xlabel='Sparsity', ylabel='Fidelity-')

  # legend
  fig.legend(['Grad Cam', "GNNExplainer", "PGExplainer",  "Random-Node"]) 

  results = [
           [sparsity_fidelity_grad_cam, sparsity_fidelity_inv_grad_cam, sparsity_grad_cam],
           [sparsity_fidelity_gnnexplainer, sparsity_fidelity_inv_gnnexplainer, sparsity_gnnexplainer],
           [sparsity_fidelity_pgexplainer, sparsity_fidelity_inv_pgexplainer, sparsity_pgexplainer],
           [sparsity_fidelity_random_node, sparsity_fidelity_inv_random_node, sparsity_random_node]]
  	
  # Speichern und Diagramm abbilden
  plt.savefig(path / "plot")
  joblib.dump(results, path / "evaluation_results.joblib")  
  plt.show()





def split_dataset_2_8(dataset):
  '''
  Datensatz in Test (20%) und Train (80%) Datensätze splitten 
  '''
  try:
      random.seed(33)   # Damit gleich gesplittet wir
      random.shuffle(dataset)   # Mischen
  except:
      torch.manual_seed(33) 
      dataset = dataset.shuffle(33)

  # Split Datasets in Train, Test und XAI_Validation Dataset
  split = int(0.8*len(dataset))
  
  ds_train = dataset[:split]
  ds_test = dataset[split:]
  return ds_train, ds_test

def get_instanz_prediction_gcn(data_list, model, device):
  # Vorhersage für Daten-Instanzen machen
  # Batch erstellen
  batch_instance = Batch.from_data_list(data_list = data_list)

  with torch.no_grad():
    batch_instance.to(device)
    #model.setbatch(test_batch.batch)
    out = model(data = batch_instance)      
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    
  return pred, out, batch_instance.y

def prediction_check(datalist, model, number_instances, device):
  '''
  Eine Liste erstellen, welche nur Instanzen enthält, welche durch das Model richtig klassifiziert werden
  :param model: gcn model
  :param number_instances: Länge der liste
  :param device: cpu/ gpu
  '''
  data_list = []
  for data in datalist:
    pred, out, label = get_instanz_prediction_gcn([data], model, device)
    #label = label.to("cpu").numpy
    if pred == label:
      data_list.append(data)
    if len(data_list) >= number_instances:
      return data_list
  return data_list
