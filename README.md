# HECC phase prediction
Machine learning models for predicting the single-phase synthesizability of HEC carbides (HECCs)
![ML framewok](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/Pictures/ML_frame.svg)
## 1. Predict from chemical formula

```
python 
```
### Example:
#### Run:
```
python pred_formula.py Ti0.2V0.2Cr0.2Nb0.2Ta0.2 # fractions in this formula refer to the molecular percentages of constituent precursors.
```
#### Output:
```
Prediction from ANN: 0.995
Prediction from SVM: 1.000
Average: 0.998
```
<br/>

## 2. Model I: [Artificial neural network](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/ANN/ANN_manual.md)


## 3. Model II: [Support vector machine](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/SVM/SVM_manual.md)
<br/>  

## 4. Results
| Material | ANN | SVM | Material | ANN | SVM |
|------ | ------ | ------ |------ | ------ | ------ |
| test | test | test | test | test | test | test |
