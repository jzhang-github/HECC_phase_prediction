# HECC phase prediction
Machine learning models for predicting the single-phase synthesizability of HEC carbides (HECCs)
Link for figure
## Predict from chemical formula
```
python 
```
### Example:
```
python pred_formula.py Ti0.2V0.2Cr0.2Nb0.2Ta0.2 # fractions in this formula refer to the molecular percentages of constituent precursors.
```
#### Output:
```
Prediction from ANN: 0.995
Prediction from SVM: 1.000
Average: 0.998
```
## Model I: Artificial neural netwok (ANN)
### Training
```
python 
```
### Predicting
```
python
```

### For the architecture modification of ANN model
Line

## Model II: Support vector machine (SVM)
### Training
```
python 
```
### Predicting
```
python
```

### For the architecture modification of SVM model
Line

# Other
### Train a model with your own dataset
Code line

# Results
| Material | ANN | SVM | Material | ANN | SVM |
|------ | ------ | ------ |------ | ------ | ------ |
| test | test | test | test | test | test | test |
