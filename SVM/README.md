# Support vector machine for predicting single-phase stability of high-entropy ceramics

### Default architecture:
kernel: linear; C parameter: 1.0.
### 1. Change settings

Modify the [source code](https://github.com/jzhang-github/HECC_phase_prediction/blob/7aadf7b73039bcf33f2c1b7e1bae1a301a03c2d2/SVM/svm_liner_scaler.py#L10-L24) for your own purpose

### 2. Training
2.1 Modify [this line](https://github.com/jzhang-github/HECC_phase_prediction/blob/7aadf7b73039bcf33f2c1b7e1bae1a301a03c2d2/SVM/svm_liner_scaler.py#L13) as `True`.

2.2 Run
```
python svm_liner_scaler.py
```
### 3. Predicting
3.1 Modify [this line](https://github.com/jzhang-github/HECC_phase_prediction/blob/7aadf7b73039bcf33f2c1b7e1bae1a301a03c2d2/SVM/svm_liner_scaler.py#L23) as `True`.

3.2 Run
```
python svm_liner_scaler.py
```

### 4. Note
4.1 Structure of files

Run 
```
tree
```

Output
```
SVM
├── model_save
│   ├── model_1.svm
│   ├── model_2.svm
│   └── ...
├── svm_liner_scaler.py
├── varibles.txt
├── x_data.txt
├── x_pred.txt
└── y_data.txt
```

4.2 Input features used in SVM model
<br/>

| Feature | Description | 
|------ | ------ | 
| ΔH<sub>mix</sub> | Mixing enthalpy per formula unit | 
| ΔV<sub>mix</sub> | Volume change per formula unit due to mixing |
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
