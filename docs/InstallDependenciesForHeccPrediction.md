## Install dependencies

If you have `ModuleNotFoundError` when you run `python predict_from_formula.py -h`, you can install it using `pip` or `conda`.

**Example:**

**Error**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'numpy'
```

**run**
```
pip install numpy
```

or

```
conda install numpy
```

**Dependent modules used in this project**

* numpy==1.19.5
* pymatgen==2020.11.11
* tensorflow==2.4.1
* scikit-learn==0.24.1
