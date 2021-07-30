# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:14:59 2021

@author: ZHANG Jun
"""

from sys import argv
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.ion import Ion
import json
import os
import tensorflow as tf
import copy
import joblib
import argparse
import sys

class HeccPred(object):
    def __init__(self,
                 ann_model_path,
                 svm_model_path,
                 max_min_path):
        self.ann_model_path = ann_model_path
        self.svm_model_path = svm_model_path
        self.max_min_path   = max_min_path
        self.ann_models     = self.load_ann(self.ann_model_path)
        self.svm_models     = self.load_svm(self.svm_model_path)

        with open('properties_of_precursors.json', 'r') as json_file:
            self.pre_prop = json.load(json_file)
        self.prop_list = ['mixing_entropy',
                          'mean_volume',
                          'diff_volume',
                          'mean_mass',
                          'diff_mass',
                          'mean_density',
                          'diff_density',
                          'mean_VEC',
                          'diff_VEC',
                          'diff_electronegativity']

    def get_feat(self, comp):
        ele_name_list   = [x.name for x in comp.elements]
        atomic_con_list = [comp.get_atomic_fraction(x) for x in ele_name_list]
        pre_list        = [x + 'C' for x in ele_name_list]
        num_of_pre      = len(ele_name_list)
        feat_dict = {}

        # mixing entropy
        feat_dict['mixing_entropy'] = 0.
        for i in range(num_of_pre):
            feat_dict['mixing_entropy'] += -8.3145 * atomic_con_list[i] * np.log(atomic_con_list[i])

        for prop in ['volume', 'mass', 'density', 'VEC', 'electronegativity']:
            feat_dict['mean_' + prop], feat_dict['diff' + prop] =  0.0, 0.0
            for i in range(num_of_pre):
                feat_dict['mean_' + prop] += self.pre_prop[pre_list[i]][prop + '_per_formula'] * atomic_con_list[i]
            prop_diff_std_sqr = 0.0
            for i in range(num_of_pre):
                prop_diff_std_sqr += atomic_con_list[i] * (1 - self.pre_prop[pre_list[i]][prop + '_per_formula'] / feat_dict['mean_' + prop]) ** 2
            feat_dict['diff_' + prop] = prop_diff_std_sqr ** 0.5
        feat = [feat_dict[x] for x in self.prop_list]
        return feat

    def scale_feat(self, feat):
        feat = copy.deepcopy(feat)
        number_of_features = np.shape(feat)[1]
        with open(self.max_min_path, 'r') as f:
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
        for i in range(np.shape(feat)[1]):
            for j in range(np.shape(feat)[0]):
                feat[j][i] = (feat[j][i] - x_min[i]) / (x_max[i] - x_min[i])
        return feat

    def load_ann(self, ann_path):
        number_of_models = len(os.listdir(ann_path))
        ann_models = []
        for i in range(number_of_models):
            model_name = os.path.join(ann_path, 'model_'+str(i + 1)+'_dense_layer.model')
            ann_models.append(tf.keras.models.load_model(model_name))
        return ann_models

    def load_svm(self, svm_path):
        number_of_models = len(os.listdir(svm_path))
        svm_models = []
        for i in range(number_of_models):
            model_name = os.path.join(svm_path, 'model_'+str(i)+'.svm')
            svm_models.append(joblib.load(model_name))
        return svm_models

    def pred_from_ann(self, feat): # type of feat: np.array or a list of list
        ann_pred_list = []
        for ann in self.ann_models:
            ann_pred_list.append(ann(feat))
        ann_pred_list = np.array(ann_pred_list)
        ann_pred_list = np.mean(ann_pred_list, axis=0)
        return ann_pred_list

    def pred_from_svm(self, feat): # type of feat: np.array
        svm_pred_list = []
        for svm in self.svm_models:
            svm_pred_list.append(svm.predict(feat))
        svm_pred_list = np.array(svm_pred_list)
        svm_pred_list = np.mean(svm_pred_list, axis=0)
        return svm_pred_list

    def __call__(self, formula_list):
        comp_list = [Ion.from_formula(x) for x in formula_list]
        feat_list = [self.get_feat(comp) for comp in comp_list]
        feat_list = self.scale_feat(feat_list)
        feat_list = tf.constant(feat_list, dtype='float32')
        ann_pred  = self.pred_from_ann(feat_list)
        svm_pred  = self.pred_from_svm(feat_list)

        print('Phase code: Single phase: 0.0; multi phase: 1.0', end='\n\n')
        print('Prediction(s) from ANN:', end=' ')
        [print('%.3f' % x, end=' ') for x in ann_pred[:,1]]
        print()

        print('Prediction(s) from SVM:', end=' ')
        [print('%.3f' % x, end=' ') for x in svm_pred]
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HECC phase prediction.')
    parser.add_argument('--ann_model_path',
                        help='Path to the ANN model.',
                        default='OtherFiles/models/ANN',
                        type=str)
    parser.add_argument('--svm_model_path',
                        help='Path to the SVM model.',
                        default='OtherFiles/models/SVM',
                        type=str)
    parser.add_argument('--max_min_path',
                        help='Path to the file that contains the max and min values of previous features.',
                        default='OtherFiles/models/variables.txt',
                        type=str)
    parser.add_argument('--formula',
                        help='A list of chemical formulas that contain the cations only.',
                        nargs='+')
    args = parser.parse_args(sys.argv[1:])
    predictor = HeccPred(args.ann_model_path, args.svm_model_path, args.max_min_path)
    predictor(args.formula)

