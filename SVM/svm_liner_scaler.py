import sklearn
from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
import joblib
import os

#######################################
SEED = 333
Number_of_fold = 53
#######################################
Train_model = True
Save_model = True
Model_save_path = './model_save/'
feature_file = 'x_data.txt'
label_file   = 'y_data.txt'
kernel = 'linear' #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
C_parameter = 1.0
class_weight = 'balanced' #default: None
degree = 3 #Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
#######################################
Predict = True    #!!!important!!!: the predict set should be included and scaled in the same scaler. if true, then load saved model, and predict new samples.
pred_feature_file = 'x_pred.txt'
#######################################

def import_data(feature_file, label_file=None):
    x_data = np.loadtxt(feature_file)
    y_data = [None]
    if label_file: #type(label_file) is not 'NoneType':
        y_data = np.loadtxt(label_file)
    return x_data, y_data

def scale_data(x_data, scaler=False):
    if scaler:  # scaler alredy exist.
        number_of_features = np.shape(x_data)[1]
        with open('variables.txt', 'r') as f:
            x_min, x_max = [], []
            comment = f.readline()
            for i in range(number_of_features):
                line = f.readline()
                x_min.append(float(line))
            blank = f.readline()
            comment = f.readline()
            for i in range(number_of_features):
                line = f.readline()
                x_max.append(float(line))
    else:
        x_min = np.min(x_data, axis=0)
        x_max  = np.max(x_data, axis=0)
        with open('variables.txt', 'w') as f:
            f.write('x_min'+'\n')
            np.savetxt(f, x_min)
            f.write('\n'+'x_max'+'\n')
            np.savetxt(f, x_max)

    for i in range(np.shape(x_data)[1]):
        for j in range(np.shape(x_data)[0]):
            x_data[j][i] = (x_data[j][i] - x_min[i]) / (x_max[i] - x_min[i])
    return x_data

def shuffle_data(x_data, y_data=None):
    np.random.seed(SEED)
    np.random.shuffle(x_data)
    if y_data is not None:
        np.random.seed(SEED)
        np.random.shuffle(y_data)
    return x_data, y_data

def training():
    x_train, y_train = import_data(feature_file, label_file)
    x_train = scale_data(x_train, scaler=False)
    x, y = shuffle_data(x_train, y_train)
    global_acc = []
    global_pred_array = []
    N_support = []
    global_step = 0
    kf = KFold(n_splits=Number_of_fold)
    for x_train_index, x_test_index in kf.split(x):
        x_train, x_test, y_train, y_test = [], [], [], []
        for i in x_train_index:
            x_train.append(x[i])
            y_train.append(y[i])
        for i in x_test_index:
            x_test.append(x[i])
            y_test.append(y[i])

        clf = svm.SVC(kernel=kernel, C=C_parameter, class_weight=class_weight, degree=degree)
        clf.fit(x_train, y_train)

        #save the model
        if Save_model:
            if not os.path.exists(Model_save_path):
                os.makedirs(Model_save_path)
            model_name = str(Model_save_path)+'model_'+str(global_step)+'.svm'
            global_step += 1
            joblib.dump(clf, model_name)

        #predict
        y_pred = clf.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        global_acc.append(acc)
        y_pred_all = clf.predict(x)
        global_pred_array.append(y_pred_all)
        N_support.append(np.sum(clf.n_support_))
    print('====Training results====')
    print("Number of support vectors:", np.mean(N_support))
    print('Validation accuracy:', np.mean(global_acc))

    #show confusion matrix
    global_pred_array = np.array(global_pred_array)
    global_pred_list = global_pred_array.mean(axis=0)
    global_pred_list = np.around(global_pred_list)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y, global_pred_list, normalize='all').ravel()
    #1 is defined positive, 0 is defined as negative.
    print('TPR, FPR, FNR, TNR:', tp, fp, fn, tn)

def predicting():
    x_pred = import_data(pred_feature_file)[0]
    x_pred = scale_data(x_pred, scaler=True)
    print('Number of new samples:', len(x_pred))

    y_pred_all = []
    global_step = 0
    for i in range(Number_of_fold):
        model_name = str(Model_save_path)+'model_'+str(global_step)+'.svm'
        clf_pred = joblib.load(model_name)
        y_pred = clf_pred.predict(x_pred)
        y_pred_all.append(y_pred)
        global_step += 1
    y_pred_all = np.array(y_pred_all)
    y_pred_all = y_pred_all.mean(axis=0)
    print('====Predict results====')
    for i in y_pred_all:
        print(i)
    #print(y_pred_all)

if __name__ == '__main__':
    if Train_model:
        training()
    if Predict:
        predicting()
