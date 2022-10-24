# Data-Mining-Blood-Glucose
Use k-means to extract information about glucose dataset

Detailed description of the project, the training models used, and conclusions can be found in the DataMiningPaper.pdf file.

Execution instructions: run main.py with "CGMData.csv" and "InsulinData.csv" in the same directory.
Results will be stored in "results.csv"
It will contain:
a) Percentage time in hyperglycemia (CGM > 180 mg/dL),
b) percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
c) percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
d) percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
e) percentage time in hypoglycemia level 1 (CGM < 70 mg/dL), and
f) percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)
