# Artificial neural network for predicting HECC phase

### Default architecture:
2 hidden layers, 6 nodes per layer.
### 1. Change settings

Modify the [source code](https://github.com/jzhang-github/HECC_phase_prediction/blob/1bf0398edeaf0fd3115df1ed0fe1360490957bc8/ANN/CV-NN-ML_parallel.py#L15-L41) for your own purpose

### 2. Training
2.1 Modify [this line](https://github.com/jzhang-github/HECC_phase_prediction/blob/1bf0398edeaf0fd3115df1ed0fe1360490957bc8/ANN/CV-NN-ML_parallel.py#L15) as `True`.

2.2 Run
```
python CV-NN-ML_parallel.py
```
### 3. Predicting
3.1 Modify [this line](https://github.com/jzhang-github/HECC_phase_prediction/blob/1bf0398edeaf0fd3115df1ed0fe1360490957bc8/ANN/CV-NN-ML_parallel.py#L36) as `True`.

3.2 Run
```
python CV-NN-ML_parallel.py
```

### 4. Note
* **Structure of files**

Run 
```
tree
```

Output
```
ANN
├── checkpoint  # Out puts are saved here
│   ├── cp.ckpt # Saved models
│   │   ├── model_1_dense_layer.model
│   │   ├── model_2_dense_layer.model
│   │   └── ...
│   ├── log     # log files for training
│   │   ├── 2_layer-6_nodes.global.acc.loss
│   │   ├── confusion_matrix.txt
│   │   ├── model_1-2_layer-6_nodes.acc.loss
│   │   ├── model_1_weights.txt
│   │   ├── model_2-2_layer-6_nodes.acc.loss
│   │   ├── model_2_weights.txt
│   │   ├── ...
│   │   ├── ...
│   │   └── varibles.txt
│   └── pred     # Results of prediction
│       ├── prediction_argmax.txt
│       └── prediction_softmax.txt
├── CV-NN-ML_parallel.py
├── x_data.txt   # Input features for training
├── x_pred.txt   # Input features for prediction
└── y_data.txt   # Input labels for training
```

* **Description of input features**
![12 features](https://github.com/jzhang-github/HECC_phase_prediction/blob/main/Pictures/12Features.svg)
