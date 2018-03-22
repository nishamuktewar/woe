# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import os
import numpy as np
import woe.feature_process as fp
import woe.GridSearch as gs

import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/IPython/extensions/")
%load_ext autoreload
%autoreload 2

if __name__ == '__main__':
    config_path = os.getcwd()+'//examples/config2.csv'
    data_path = os.getcwd()+'//examples/UCI_Credit_Card2.csv'
    feature_detail_path = os.getcwd()+'//examples/features_detail.csv'
    rst_pkl_path = os.getcwd()+'//examples/woe_rule.pkl'
    rebin_feature_path = os.getcwd()+'//examples/features_rebin.csv'
    
    # train woe rule
    feature_detail,rst = fp.process_train_woe(infile_path=data_path
                                           ,outfile_path=feature_detail_path
                                           ,rst_path=rst_pkl_path
                                           ,config_path=config_path
                                           ,rebin_feature_path=rebin_feature_path)
    # proc woe transformation
    woe_train_path = os.getcwd()+'//examples/dataset_train_woed.csv'
    fp.process_woe_trans(data_path,rst_pkl_path,woe_train_path,config_path,rebin_feature_path)
    # here i take the same dataset as test dataset
    woe_test_path = os.getcwd()+'//examples/dataset_test_woed.csv'
    fp.process_woe_trans(data_path,rst_pkl_path,woe_test_path,config_path,rebin_feature_path)