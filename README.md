# HECC phase prediction
Machine learning models for predicting the single-phase synthesizability of HEC carbides (HECCs)
![ML framewok](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/Pictures/ML_frame.svg)
## 1. Quick prediction from chemical formulas

 ### 1.1 Print help tab

```
python predict_from_formula.py -h
```

Any uninstalled modules can be installed by `pip` or `anaconda`. Click [here](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/docs/InstallDependenciesForHeccPrediction.md) for more details. 

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

<br/>
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

**Note:** <br/>

* These formulas give the same result: `TiVCrNbTa`, `Ti1V1C1rNb1Ta1`, `Ti0.2V0.2Cr0.2Nb0.2Ta0.2`, `Ti0.03V0.03Cr0.03Nb0.03Ta0.03`.
    
* Direct predictions in the `output` are the `multi-phase probability`, **NOT** the `single-phase probability` because single- and multi-phase samples were labeled as `0` and `1`, respectively. 
<br/>

## 2. Model I: [Artificial neural network](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/ANN/ANN_manual.md)


## 3. Model II: [Support vector machine](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/SVM/SVM_manual.md)
<br/>  

## 4. Results
| Material | ANN | SVM | Material | ANN | SVM |
|------ | ------ | ------ |------ | ------ | ------ |
| test | test | test | test | test | test | test |
