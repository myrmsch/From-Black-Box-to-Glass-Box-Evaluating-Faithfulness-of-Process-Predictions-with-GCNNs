# From Black Box to Glass Box: Evaluating Faithfulness of Process Predictions with GCNNs

### About the project

We investigate the faithfulness of three post-hoc GCNN-based explanation techniques for process outcome predictions and discuss their general suitability for PBPM. 

***
### Naming scheme

The starting point is the loan log and the review log. After preprocessing, there are four data sets. Two contain the activity-based instance graphs 
and two contain the buckets with the prefixes of the event-based instance graphs. The event-based datasets also contain the entire graph.
The four data sets are:

*Activity-based encoding*
- Review-Log:  	review_sn_f3_ohe_2
- Loan-Log:		loan_sn_f2_ohe_wEvents_4_nreb
  
*Event-based encoding*
- Review-Log:	review_all_events_f5_ohe_2_prefix
- Loan-Log:		loan_all_events_fall_ohe_wEvents_5_2

***
### Folder structure

**Main folder**

  Contains the scripts for the evaluation of the XAI methods as well as auxiliary scripts (utilities*) 
	- **utilities.py:** contains various methods for visualizing explanation graphs. Furthermore, the random explainer is implemented here.
	- **utilities_xai.py:** methods which are used to generate explanations of the explainers implemented in DIG. 
					Methods for visualizing the evaluation metrics in the form of diagrams
	- **utilities_parameter_tuning:** methods for parameter tuning of the GCNs
	- **train_gcn.py:** Methods for training the GCN models
					Methods to save and load the models
	- **utilities_pgexplainer_tuning:** methods for parameter tuning of the PGExplainer
	- **utilities_gnnexplainer:** methods for tuning the GNNExplainer
	- **utilities_preprocessing:** auxiliary methods and class for preprocessing
			
**Folders**

- **datasets:** Here the datasets are stored in their original form (XES format), and after preprocessing.
- **preprocessing:** Here are scripts for preprocessing the event logs
- **models:** The model used and the scripts for parameter tuning and training of the models are stored here. The results of the tuning runs are stored in subfolders
- **xai_methods:** The parameter tuning of the XAI methods used is carried out here. The results of the tuning runs are stored in subfolders
	- Loan_Tuning_Results & Review_Tuning_Results. Bar charts of the fidelity values achieved.
  - Naming set names .yml file in subfolder tuning/{dataset_name}/{bar_diagram_naming}
- xai_results: contains the results of the evaluation of the XAI methods
	- plot: contains diagram with the results of the evaluation
	- evaluation_results.joblib: array with results

 
**Further information:**

_.joblib_ files are arrays of results or feature names stored (often loaded in code)
_.YMl_ files contain results and/or parameters from tuning/training runs. Serve as information on actions taken  
_run_x_ folders are older runs. The final runs are located directly in the respective Result/Training/Tuning folder.

### Credits

Datasets:

- BPI Challenge 2017 - Loan eventlog (https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884/1)
- Synthetic eventlog - Review event log (https://data.4tu.nl/articles/dataset/Synthetic_event_logs_-_review_example_large_xes_gz/12716609/1)

The experimental evaluation is implemented in Python 3.8.5. We use PyTorch Geometrics and PyTorch back-ends for an efficient GPU-based implementation. 
For the model implementation, we use the open-source library DIG, which provides a module dedicated to explainability in GNNs. 
For the visualization of the explanation results and processes, we employ NetworkX, a Python package for graph data structures in particular, and PM4Py, a Python package for process mining.
