# HECC phase prediction
Machine learning models for predicting the single-phase synthesizability of HEC carbides (HECCs)
![ML framewok](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/Pictures/ML_frame.svg)
## 1. Predict from chemical formula

 ### 1.1 Print help tab

```
python predict_from_formula.py -h
```
**Output:**
```
HECC phase prediction.

optional arguments:
  -h, --help            show this help message and exit
  --ann_model_path ANN_MODEL_PATH
                        Path to the ANN model.
  --svm_model_path SVM_MODEL_PATH
                        Path to the SVM model.
  --max_min_path MAX_MIN_PATH
                        Path to the file that contains the max and min values
                        of previous features.
  --formula FORMULA [FORMULA ...]
                        A list of chemical formula that contains the cations
                        only.
```

Any uninstalled modules can be installed by `pip` or `anaconda`. For example: `pip install tensorflow` or `conda install numpy`.
```
python 
```
### 1.2 Predict single-phase synthesizability from chemical formulas
**Example:**
<br/>
**Run:**
```
python predict_from_formula.py --formula TiVCrNbTa VCrNbMoTa TiVCrZrMo # Only cations should be included here.
```
**Output:**
```
Phase code: Single phase: 0.0; multi phase: 1.0

Prediction(s) from ANN: 0.049 0.047 1.000
Prediction(s) from SVM: 0.000 0.000 1.000
```
<br/>

## 2. Model I: [Artificial neural network](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/ANN/ANN_manual.md)


## 3. Model II: [Support vector machine](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/SVM/SVM_manual.md)
<br/>  

## 4. Results
| Material | ANN | SVM | Material | ANN | SVM |
|------ | ------ | ------ |------ | ------ | ------ |
| test | test | test | test | test | test | test |
