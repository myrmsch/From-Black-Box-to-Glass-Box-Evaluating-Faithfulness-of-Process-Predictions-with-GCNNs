Masterarbeit: Graphenbasierte neuronale Netzte (GNN) für erklärbare prädikative Prozessanalysen

Benennungsschema:

Ausgangspunkt sind der Loan-Log und der Review-Log. 
Nach dem Preprocessing exitieren vier Datensätze. Zwei enthalten die Aktivät-basierten Instanz-Graphen 
und zwei die Buckets mit den Präfixen der Event-basierten Instanz-Graphen.
Die Event-basierten Datensätzen enthalten ebenfalls den gesamt Graph.
Die vier Datensätze sind:

Aktivität-basiert
- Review-Log:  	review_sn_f3_ohe_2
- Loan-Log:		loan_sn_f2_ohe_wEvents_4_nreb
Event-basiert
- Review-Log:	review_all_events_f5_ohe_2_prefix
- Loan-Log:		loan_all_events_fall_ohe_wEvents_5_2

Ordnerstruktur:
Ordner:
- datasets: 	Hier werden die Datensätze in ihrer ursprünglichen Form (XES-Format), und nach der Vorverarbeitung gespeichert
- preprocessing: 	Hier befinden sich Skripte zur Vorverarbeitung der Event-Logs
- models:		Hier ist das verwendete Model und die Skripte zum Parameter-Tuning sowie Training der Modelle gespeichert. Die Ergebnisse der Tuning-Läufen ist in Unterordnern gespeichert
- xai_methods: 	Hier wird das Parameter-Tuning der verwendeten XAI-Methoden durchgeführt. Die Ergebnisse der Tuning-Läufe ist in Unterordnern gespeichert
	- Loan_Tuning_Results & Review_Tuning_Results. Bar-Diagramme der erreichten Fidelity-Werte. Benennungen stellen Namen .yml-Datei im Unterordner tuning/{dataset_name}/{bar_diagramm_benennung}
- xai_results: 	Hier sind die Ergebnisse der Evaluation der XAI-Methoden gespeichert
	- plot: enthält Diagramm mit den Ergebnissen der Evaluation
	- evaluation_results.joblib: Array mit Ergebnissen 
- Hauptordner	Hier befinden sich die Skripte für die Evaluation der XAI-Methoden sowie Hilfs-Skripte (utilities*) 
	- utilities.py: 		enthält verschiedene Methoden der Visualisierung von Erklär-Graphen. Des Weiteren ist hier der Random-Explainer implementiert
	- utilities_xai.py: 	Methoden, welche zum Generieren von Erklärungen der die in DIG implementierten Explainer verwendet werden. 
					Methoden zur Visualisierung der Evaluations-Metriken in Form von Diagrammen
	- utilities_parameter_tuning: Methoden zum Parameter - Tuning der GCNs
	- train_gcn.py 		Methoden zum trainieren der GCN-Modelle
					Methoden zum speichern und laden der Modelle
	- utilities_pgexplainer_tuning: ethoden zum Parameter-Tuning des PGExplainers
	- utilities_gnnexplainer: Methoden zum Tuning des GNNExplainers
	- utilities_preprocessing: Hilfs-Methoden und Klasse für Preprocessing
					Darstellung von Graphen

.joblib-Files sind Arrays mit Ergebnissen oder Feature-Namen gespeichert. Werden häufig im Code geladen
.YMl-Files enthalten Ergebnisse und/oder Parameter von Tuning/Trainings Läufen. Dienen als Information zu vorgenommenen Aktionen  
