
#import modules
import time
now_time = time.time()
import tensorflow              as tf
import numpy                   as np
import os
import multiprocessing
import sklearn
from   sklearn.model_selection import KFold

#important global variables
#==========================
if __name__ == '__main__':
    Train_model          = False                         #run the training or not.
    feature_file         = 'x_data.txt'
    label_file           = 'y_data.txt'
    Activation_function  = 'softmax'                     #alternatives: relu, softmax, sigmoid, tanh
    Output_activation    = 'softmax'
    Optimizer            = 'adam'                        #alternatives: sgd, adagrad, adadelta, adam
    Cost_function        = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) #alternatives: 'mse'
    Metrics              = 'sparse_categorical_accuracy' #metrics are dependent on activation function. alternatives: binary_accuracy, categorical_accuracy, etc.
    Number_of_layers     = 2                             #number of hidden layers
    Nodes_per_layer      = 6                            #number of nodes of hidden layer
    Number_of_out_node   = 2                             #number of categories
    Batch_size           = 16                            #2^n, n is a integer.
    Epochs               = 1000
    Number_of_fold       = 10
    #Early_stop           = 500                           #this tag controls early stoping. If the performance does not improve for 500 steps, the model will stop.
    Model_save           = False                         #as you can learn from this tag.
    Regularization       = False
    Verbose              = 0                             #alternatives: 0, no output on the screen; 1, output with progress bar; 2, output without progress bar.
    Model_save_path      = './checkpoint/cp.ckpt'
    Log_save_path        = './checkpoint/log'
#==========================
    Load_and_predict     = True                         #if True, load saved model and predict samples.
    Prediction_save_path = './checkpoint/pred'
    Predict_feature_file = 'x_pred.txt'
#==========================
    SEED                 = 666
    Number_of_cores      = min(Number_of_fold, 55)
#==========================

#import data, features and labels
def import_data():
    x_data = np.loadtxt(feature_file)
    y_data = np.loadtxt(label_file)
    x_min  = np.min(x_data, axis=0)
    x_max  = np.max(x_data, axis=0)
    for i in range(np.shape(x_data)[1]):
        for j in range(np.shape(x_data)[0]):
            x_data[j][i] = (x_data[j][i] - x_min[i]) / (x_max[i] - x_min[i])

    #shuffle the data
    np.random.seed(SEED)
    np.random.shuffle(x_data)
    np.random.seed(SEED)
    np.random.shuffle(y_data)
    count_for_0 = np.sum(y_data==0)
    count_for_1 = np.sum(y_data==1)
    count_total = count_for_0 + count_for_1
    weight_for_0 = (1 / count_for_0) * count_total / 2.0
    weight_for_1 = (1 / count_for_1) * count_total / 2.0
    with open(str(Log_save_path)+'/variables.txt', 'w') as f:
        f.write('x_min'+'\n')
        np.savetxt(f, x_min)
        f.write('\n'+'x_max'+'\n')
        np.savetxt(f, x_max)
    return x_data, y_data, weight_for_0, weight_for_1

def Training_module(x_data, y_data, x_train_index, x_test_index, model_index, global_acc, global_val_acc, global_loss, global_val_loss, weight_for_0, weight_for_1):
    #split the data in to training and testing part.
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in x_train_index:
        x_train.append(x_data[i])
        y_train.append(y_data[i])
    for i in x_test_index:
        x_test.append(x_data[i])
        y_test.append(y_data[i])

    #convert the data into tf format
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test  = tf.convert_to_tensor(x_test,  dtype=tf.float32)
    y_test  = tf.convert_to_tensor(y_test,  dtype=tf.float32)
    
    #build an empty model
    model = tf.keras.models.Sequential()
    #add the input layer
    model.add(tf.keras.Input(shape=(x_train.shape[1],), name='Input_layer'))
    #add hidden layer(s)
    if Regularization == True:
        print("L2 regularizers is used.")
        for i in range(Number_of_layers):
            model.add(tf.keras.layers.Dense(Nodes_per_layer, activation=Activation_function, kernel_regularizer=tf.keras.regularizers.l2(), name="Hidden_layer_"+str(i)))
    elif Regularization == False:
        for i in range(Number_of_layers):
            Layer_name = str("Hidden_layer "+str(i))
            model.add(tf.keras.layers.Dense(Nodes_per_layer, activation=Activation_function, name="Hidden_layer_"+str(i)))
    else:
        print("Illegal value for Regularization.")
        os._exit(0)
    #add the output layer
    model.add(tf.keras.layers.Dense(Number_of_out_node, activation=Output_activation, name="Output_layer"))
    
    #compile
    model.compile(optimizer=Optimizer,
                  loss=Cost_function,
                  metrics=[Metrics])
    
    #fit the model
    class_weight = {0: weight_for_0, 1: weight_for_1}
    history = model.fit(x_train, y_train,
                        batch_size=Batch_size, epochs=Epochs, verbose=Verbose,
                        validation_data=(x_test, y_test), validation_freq=1,
                        class_weight=class_weight)
    
    #print summary and trainable variables
    model.summary()
    file = open(str(Log_save_path)+'/model_'+str(model_index)+'_weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    #save the model
    model_name = str(Model_save_path)+'/'+'model_'+str(model_index)+'_dense_layer.model'
    model.save(model_name)
    
    #save history
    acc      = history.history['sparse_categorical_accuracy']
    val_acc  = history.history['val_sparse_categorical_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    global_acc.append(acc[-1])
    global_val_acc.append(val_acc[-1])
    global_loss.append(loss[-1])
    global_val_loss.append(val_loss[-1])

    #write result into disk
    acc_loss_filename = 'model'+'_'+str(model_index)+'-'+str(Number_of_layers)+'_'+'layer'+'-'+str(Nodes_per_layer)+'_'+'nodes'+'.acc.loss'
    acc_loss_path     = os.path.join(Log_save_path, acc_loss_filename)
    with open(acc_loss_path, 'w') as f:
        f.write('%-5s' % 'step' + '%13s' % 'acc' + '%13s' % 'val_acc' +'%13s' % 'loss' +'%13s' % 'val_loss' + '\n')
        for i in range(Epochs):
            f.write('%-5s' % str(int(i)) + '%13.8f' % acc[i] + '%13.8f' % val_acc[i] +'%13.8f' % loss[i] +'%13.8f' % val_loss[i] + '\n')

def CV_ML_RUN():
    kf = KFold(n_splits=Number_of_fold)
    model_index = 0
    
    #create log file
    if not os.path.exists(Log_save_path):
        os.makedirs(Log_save_path)
    
    print('Parent process %s.' % os.getpid())
    p               = multiprocessing.Pool(Number_of_cores)
    global_acc      = multiprocessing.Manager().list()
    global_val_acc  = multiprocessing.Manager().list()
    global_loss     = multiprocessing.Manager().list()
    global_val_loss = multiprocessing.Manager().list()

    x_data, y_data, weight_for_0, weight_for_1 = import_data()

    for x_train_index, x_test_index in kf.split(x_data):
        model_index += 1
        p.apply_async(Training_module, args=(x_data, y_data, x_train_index, x_test_index, model_index, global_acc, global_val_acc, global_loss, global_val_loss, weight_for_0, weight_for_1))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    
    #write the global result of every CV
    global_acc_loss_filename = str(Number_of_layers)+'_'+'layer'+'-'+str(Nodes_per_layer)+'_'+'nodes'+'.global.acc.loss'
    global_acc_loss_path     = os.path.join(Log_save_path, global_acc_loss_filename)
    with open(global_acc_loss_path, 'w') as f:
        f.write('%-8s' % ' ' + '%13s' % 'g_acc' + '%13s' % 'g_val_acc' +'%13s' % 'g_loss' +'%13s' % 'g_val_loss' + '\n')
        for i in range(Number_of_fold):
            f.write('%-8s' % str('model ' +str(i + 1)) + '%13.8f' % global_acc[i] + '%13.8f' % global_val_acc[i] +'%13.8f' % global_loss[i] +'%13.8f' % global_val_loss[i] + '\n')
        global_acc_mean      = np.mean(global_acc)
        global_val_acc_mean  = np.mean(global_val_acc)
        global_loss_mean     = np.mean(global_loss)
        global_val_loss_mean = np.mean(global_val_loss)
        f.write('%-8s' % 'mean' + '%13.8f' % global_acc_mean + '%13.8f' % global_val_acc_mean +'%13.8f' % global_loss_mean +'%13.8f' % global_val_loss_mean + '\n')
    
    total_time = time.time() - now_time
    print("total_time", total_time, "s")

def load_and_pred(file_name_of_x_data, write_pred_log=True, confusion_matrix=True): #predict training sample to get the confusion matrix, or predict new samples.
    #load new data
    x_pred = np.loadtxt(file_name_of_x_data)
    number_of_features = np.shape(x_pred)[1]
    with open(str(Log_save_path)+'/variables.txt', 'r') as f:
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
    for i in range(np.shape(x_pred)[1]):
        for j in range(np.shape(x_pred)[0]):
            x_pred[j][i] = (x_pred[j][i] - x_min[i]) / (x_max[i] - x_min[i])

    #load the models and predict
    predictions_all = []
    for i in range(Number_of_fold):
        model_name = str(Model_save_path)+'/'+'model_'+str(i + 1)+'_dense_layer.model'
        new_model = tf.keras.models.load_model(model_name)
        predictions = new_model.predict([x_pred])
        predictions_all.append(predictions)
    predictions_all = np.array(predictions_all)
    prediction_mean = np.mean(predictions_all, axis=0)
    prediction_argmax = np.argmax(prediction_mean, axis=1)
    if write_pred_log:
        if not os.path.exists(Prediction_save_path):
            os.makedirs(Prediction_save_path)
        np.savetxt(str(Prediction_save_path)+'/prediction_softmax.txt', prediction_mean)
        np.savetxt(str(Prediction_save_path)+'/prediction_argmax.txt', prediction_argmax)
    if confusion_matrix:
        y_data = np.loadtxt(label_file)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_data, prediction_argmax, normalize='all').ravel()
        with open(str(Log_save_path)+'/confusion_matrix.txt', 'w') as f:
            f.write('TN\tFP\tFN\tTP\n')
            f.write(str(tn)+' '+str(fp)+' '+str(fn)+' '+str(tp))

#train and test, main part
if __name__ == '__main__': 
    if Train_model:
        CV_ML_RUN()
        load_and_pred(feature_file, write_pred_log=False, confusion_matrix=True)
    if Load_and_predict:
        load_and_pred(Predict_feature_file, write_pred_log=True, confusion_matrix=False)
