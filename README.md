![ML framewok](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/imgs/ML%20framework.svg)

<br/>
<br/>
# HECC phase prediction
Machine learning models for predicting the single-phase synthesizability of high-entropy-ceramic carbides (HECCs)

# Note:
**If you found strange errors for running scripts under this repo on Windows, please try again on Linux instead.**

## 1. Quick prediction from chemical formulas

 ### 1.1 Print help tab

```
python predict_from_formula.py -h
```

Any uninstalled modules can be installed by `pip` or [`anaconda`](https://www.anaconda.com/). Click [here](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/docs/InstallDependenciesForHeccPrediction.md) for more details. 

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
<!--* These models for quick prediction were trained solely on properties of constituent carbides. As shown below:
![10 features](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/Pictures/10Features.svg)
; * For models on all independent features in the original paper, please refer to [ANN](https://github.com/jzhang-github/HECC_phase_prediction#2-source-code-of-artificial-neural-network) and [SVM](https://github.com/jzhang-github/HECC_phase_prediction#3-source-code-of-support-vector-machine)-->
<br/>
<br/>

## 2. Source code of [Artificial neural network](https://github.com/jzhang-github/HECC_phase_prediction/tree/main/ANN)


## 3. Source code of [Support vector machine](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/SVM)
<br/>  

## 4. Input features used in refined models

| Feature | Description | 
|------ | ------ |  
| ΔS<sub>mix</sub> | Mixing entropy |
| V<sub>average</sub> | Average volume of constituent TMCs per formula unit |
| σ<sub>V</sub> | Volume deviation of constituent TMCs |
| m<sub>average</sub> | Average mass of constituent TMCs per formula unit |
| σ<sub>m</sub> | Mass deviation of constituent TMCs |
| ρ<sub>average</sub> | Average density of constituent TMCs |
| σ<sub>ρ</sub> | Density deviation of constituent TMCs |
| σ<sub>χ</sub> | Deviation of electronegativity of constituent TMCs |
| VEC<sub>average</sub> | Valence electron concentration (VEC) of HECC candidates |
| σ<sub>VEC</sub> | VEC deviation of constituent TMCs |

**Data of all input features:** click [here](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/docs/Input%20data.csv)
## 5. Dependent modules used in this project

* **Python 3.8.5**
* numpy==1.19.5
* pymatgen==2020.11.11
* tensorflow==2.4.1
* scikit-learn==0.24.1
