# Repairability
In Datasets we present the artificially deteriorated datasets used in our work as well as the datasets obtained when 
applying the following repairing pipeline: 
1. Missing values are replaced with the attributes' mean
2. Outliers are identified with an isolation forest and then removed

In Examples we give an example of the script used to apply the repairing pipeline on 
deteriorated datasets (ScriptRepairMissingValuesAndOutliers.py) and an example of the script used 
to compute repairability (ScriptComputeRepairability.py).

In output we present the degrees of repairability obtained from our experiences and a visualisation of these 
experiments results is available in the notebook "Visualisation.ipynb".

The function allowing the computation of the degree og repairability is available in "repairability.py"

"requirements.txt" list the libraries version requirements to execute the scripts in this repository.