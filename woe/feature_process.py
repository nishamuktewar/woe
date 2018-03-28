# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import numpy as np
import pandas as pd
import woe.config as config
import woe.eval as woeeval
import copy
import pickle
import time

class node:
    '''Tree Node Class
    '''
    def __init__(self,var_name=None,iv=0,split_point=None,right=None,left=None):
        self.var_name = var_name  # The column index value of the attributes that are used to split data sets
        self.iv = iv  # The info value of the node
        self.split_point = split_point  # Store split points list
        self.right = right  # Right sub tree
        self.left = left  # Left sub tree


class InfoValue(object):
    '''
    InfoValue Class
    '''
    def __init__(self):
        self.var_name = []
        self.split_list = []
        self.iv = 0
        self.woe_list = []
        self.iv_list = []
        self.is_discrete = 0
        self.sub_total_sample_num = []
        self.positive_sample_num = []
        self.negative_sample_num = []
        self.sub_total_num_percentage = []
        self.positive_rate_in_sub_total = []
        self.negative_rate_in_sub_total = []
        self.weight_freq = []
        self.weight_positive_freq = []
        self.weight_negative_freq = []
        self.cum_weight_freq = []
        self.cum_weight_positive_freq = []
        self.cum_weight_negative_freq = []
        self.total_weight_freq = 0
        self.total_weight_positive_freq = 0
        self.total_weight_negative_freq = 0
        self.perc_cum_weight_freq = []
        self.perc_cum_weight_positive_freq = []
        self.perc_cum_weight_negative_freq = []
        self.ks_list = []
        self.maxks = 0
        self.pos_trend = 0
        self.neg_trend = 0
        self.linearity = 0

    def init(self,civ):
        self.var_name = civ.var_name
        self.split_list = civ.split_list
        self.iv = civ.iv
        self.woe_list = civ.woe_list
        self.iv_list = civ.iv_list
        self.is_discrete = civ.is_discrete
        self.sub_total_sample_num = civ.sub_total_sample_num
        self.positive_sample_num = civ.positive_sample_num
        self.negative_sample_num = civ.negative_sample_num
        self.sub_total_num_percentage = civ.sub_total_num_percentage
        self.positive_rate_in_sub_total = civ.positive_rate_in_sub_total
        self.negative_rate_in_sub_total = civ.negative_rate_in_sub_total
        self.weight_freq = civ.weight_freq
        self.weight_positive_freq = civ.weight_positive_freq
        self.weight_negative_freq = civ.weight_negative_freq
        self.cum_weight_freq = civ.cum_weight_freq
        self.cum_weight_positive_freq = civ.cum_weight_positive_freq
        self.cum_weight_negative_freq = civ.cum_weight_negative_freq
        self.total_weight_freq = civ.total_weight_freq
        self.total_weight_positive_freq = civ.total_weight_positive_freq
        self.total_weight_negative_freq = civ.total_weight_negative_freq
        self.perc_cum_weight_freq = civ.perc_cum_weight_freq
        self.perc_cum_weight_positive_freq = civ.perc_cum_weight_positive_freq
        self.perc_cum_weight_negative_freq = civ.perc_cum_weight_negative_freq
        self.ks_list = civ.ks_list
        self.maxks = civ.maxks
        self.pos_trend = civ.pos_trend
        self.neg_trend = civ.neg_trend
        self.linearity = civ.linearity

class DisInfoValue(object):
    '''
    A Class for the storage of discrete variables transformation information
    '''
    def __init__(self):
        self.var_name = None
        self.origin_value = []
        self.woe_before = []


def change_feature_dtype(df,variable_type):
    '''
    change feature data type by the variable_type DataFrame
    :param df: dataset DataFrame
    :param variable_type: the DataFrame about variables dtypes
    :return: None
    '''
    s = 'Changing Feature Dtypes'
    print(s.center(60,'-'))
    for vname in df.columns:
        try:
            df[vname] = df[vname].astype(variable_type.loc[vname,'v_type'])
            print(vname,' '*(40-len(vname)),'{0: >10}'.format(variable_type.loc[vname,'v_type']))
        except Exception:
            print('[error]',vname)
            print('[original dtype] ',df.dtypes[vname],' [astype] ',variable_type.loc[vname,'v_type'])
            print('[unique value]',np.unique(df[vname]))

    s = 'Variable Dtypes Have Been Specified'
    print(s.center(60,'-'))

    return

def check_point(df,var,split,min_sample):
    """
    Check whether the segmentation points cause some packet samples to be too small;
    If there is a packet sample size of less than 5% of the total sample size,
    then merge with the adjacent packet until more than 5%;
    Applies only to continuous values
    :param df: Dataset DataFrame
    :param var: Variables list
    :param split: Split points list
    :param min_sample: Minimum packet sample size
    :return: The split points list checked out
    """
    new_split = []
    if split is not None and split.__len__()>0:
        # print('run into if line:98')
        new_split.append(split[0])
        # print(new_split)
        # Try the left section of the first split point partition;
        # If not meet the conditions then the split point will be removed
        pdf = df[df[var] <= split[0]]
        if (pdf.shape[0] < min_sample) or (len(np.unique(pdf['target']))<=1):
            # print('run into if line:105')
            new_split.pop()
            # print(new_split)
        for i in range(0,split.__len__()-1):
            pdf = df[(df[var] > split[i]) & (df[var] <= split[i+1])]
            if (pdf.shape[0] < min_sample) or (np.unique(pdf['target']).__len__()<=1):
                # print('run into if line:112')
                continue
            else:
                # print('run into if line:115')
                new_split.append(split[i+1])
                # print(new_split)

        #If the remaining sample is too small then remove the last one
        # print(new_split)
        # print(new_split.__len__())
        if new_split.__len__()>1 and len(df[df[var] >= new_split[new_split.__len__()-1]])<min_sample:
            # print('run into if line:120')
            new_split.pop()
            # print(new_split)
        #If the remaining samples have only a positive or negative target then remove the last one
        if new_split.__len__()>1 and np.unique(df[df[var] >= new_split[new_split.__len__()-1]]['target']).__len__()<=1:
            # print(split)
            # print(split[split.__len__()-1])
            # print(df[df[var] >= new_split[new_split.__len__()-1]].shape)
            # print(np.unique(df[df[new_split] > new_split[new_split.__len__()-1]]['target']))
            # print('run into if line:125')
            new_split.pop()
            # print(new_split)
        #If the split list has only one value, and no smaller than this value
        if new_split == []:
            new_split = split
    else:
        pass
    return new_split

def calulate_iv(df,var,global_bt,global_gt,global_wbt,global_wgt):
    '''
    calculate the iv and woe value without split
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    # a = df.groupby(['target']).count()
    groupdetail = {}
    bt_sub = sum(df['target'])
    bri = (bt_sub + 0.0001)* 1.0 / global_bt
    gt_sub = df.shape[0] - bt_sub
    gri = (gt_sub + 0.0001)* 1.0 / global_gt

    groupdetail['woei'] = np.log(bri / gri)
    groupdetail['ivi'] = (bri - gri) * np.log(bri / gri)
    groupdetail['sub_total_num_percentage'] = df.shape[0]*1.0/(global_bt+global_gt)
    groupdetail['positive_sample_num'] = bt_sub
    groupdetail['negative_sample_num'] = gt_sub
    groupdetail['positive_rate_in_sub_total'] = bt_sub*1.0/df.shape[0]
    groupdetail['negative_rate_in_sub_total'] = gt_sub*1.0/df.shape[0]

    wbt_sub = sum(df['weight'])
    wbri = (wbt_sub + 0.0001)* 1.0 / global_wbt
    wgt_sub = df.shape[0] - wbt_sub
    wgri = (wgt_sub + 0.0001)* 1.0 / global_wgt
    groupdetail['weight_freq'] = df.shape[0]*1.0/(global_wbt+global_wgt)
    groupdetail['weight_positive_freq'] = wbt_sub
    groupdetail['weight_negative_freq'] = wgt_sub

    return groupdetail


def calculate_iv_split(df,var,split_point,global_bt,global_gt):
    """
    calculate the iv value with the specified split point
    note:
        the dataset should have variables:'target' which to be encapsulated if have time
    :return:
    """
    #split dataset
    dataset_r = df[df.loc[:,var] > split_point][[var,'target','weight']]
    dataset_l = df[df.loc[:,var] <= split_point][[var,'target','weight']]

    r1_cnt = sum(dataset_r['target'])
    r0_cnt = dataset_r.shape[0] - r1_cnt

    l1_cnt = sum(dataset_l['target'])
    l0_cnt = dataset_l.shape[0] - l1_cnt

    if r0_cnt == 0 or r1_cnt == 0 or l0_cnt == 0 or l1_cnt ==0:
        return 0,0,0,dataset_l,dataset_r,0,0

    lbr = (l1_cnt+ 0.0001)*1.0/global_bt
    lgr = (l0_cnt+ 0.0001)*1.0/global_gt
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = (r1_cnt+ 0.0001)*1.0/global_bt
    rgr = (r0_cnt+ 0.0001)*1.0/global_gt
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr

    return woel,woer,iv,dataset_l,dataset_r,ivl,ivr


def binning_data_split(df,var,global_bt,global_gt,global_wbt,global_wgt,min_sample,alpha=0.01):
    """
    Specify the data split level and return the split value list
    :return:
    """
    iv_var = InfoValue()
    # Calculates the IV of the current node before splitted
    gd = calulate_iv(df, var,global_bt,global_gt,global_wbt,global_wgt)

    woei, ivi = gd['woei'],gd['ivi']

    if np.unique(df[var]).__len__() <=8:
        # print('running into if')
        split = list(np.unique(df[var]))
        split.sort()
        # print('split:',split)
        #Segmentation point checking and processing
        split = check_point(df, var, split, min_sample)
        split.sort()
        # print('after check:',split)
        iv_var.split_list = split
        return node(split_point=split,iv=ivi)

    percent_value = list(np.unique(np.percentile(df[var], range(100))))
    percent_value.sort()

    if percent_value.__len__() <=2:
        iv_var.split_list = list(np.unique(percent_value)).sort()
        return node(split_point=percent_value,iv=ivi)

    # A sentry that attempts to split the current node
    # Init bestSplit_iv with zero
    bestSplit_iv = 0
    bestSplit_woel = []
    bestSplit_woer = []
    bestSplit_ivl = 0
    bestSplit_ivr = 0
    bestSplit_point = []

    #remove max value and min value in case dataset_r  or dataset_l will be null
    for point in percent_value[0:percent_value.__len__()-1]:
        # If there is only a sample or a negative sample, skip
        if set(df[df[var] > point]['target']).__len__() == 1 or set(df[df[var] <= point]['target']).__len__() == 1 \
                or df[df[var] > point].shape[0] < min_sample or df[df[var] <= point].shape[0] < min_sample :
            continue

        woel, woer, iv, dataset_l, dataset_r, ivl, ivr = calculate_iv_split(df,var,point,global_bt,global_gt)

        if iv > bestSplit_iv:
            bestSplit_woel = woel
            bestSplit_woer = woer
            bestSplit_iv = iv
            bestSplit_point = point
            bestSplit_dataset_r = dataset_r
            bestSplit_dataset_l = dataset_l
            bestSplit_ivl = ivl
            bestSplit_ivr = ivr

    # If the IV after division is greater than the IV value before the current segmentation, the segmentation is valid and recursive
    # specified step learning rate 0.01
    if bestSplit_iv > ivi*(1+alpha) and bestSplit_dataset_r.shape[0] > min_sample and bestSplit_dataset_l.shape[0] > min_sample:
        presplit_right = node()
        presplit_left = node()

        # Determine whether the right node satisfies the segmentation prerequisite
        if bestSplit_dataset_r.shape[0] < min_sample or set(bestSplit_dataset_r['target']).__len__() == 1:
            presplit_right.iv = bestSplit_ivr
            right = presplit_right
        else:
            right = binning_data_split(bestSplit_dataset_r,var,global_bt,global_gt,global_wbt,global_wgt,min_sample,alpha=0.01)

        # Determine whether the left node satisfies the segmentation prerequisite
        if bestSplit_dataset_l.shape[0] < min_sample or np.unique(bestSplit_dataset_l['target']).__len__() == 1:
            presplit_left.iv = bestSplit_ivl
            left = presplit_left
        else:
            left = binning_data_split(bestSplit_dataset_l,var,global_bt,global_gt,global_wbt,global_wgt,min_sample,alpha=0.01)

        return node(var_name=var,split_point=bestSplit_point,iv=ivi,left=left,right=right)
    else:
        # Returns the current node as the final leaf node
        return node(var_name=var,iv=ivi)


def search(tree,split_list):
    '''
    search the tree node
    :param tree: a instance of Tree Node Class
    :return: split points list
    '''
    if isinstance(tree.split_point, list):
        split_list.extend(tree.split_point)
    else:
        split_list.append(tree.split_point)

    if tree.left is not None:
        search(tree.left,split_list)

    if tree.right is not None:
        search(tree.right,split_list)

    return split_list


def format_iv_split(df,var,split_list,global_bt,global_gt,global_wbt,global_wgt,is_continuous,global_numeric_missing,global_categorical_missing,missing_woe=0):
    '''
    Given the dataset DataFrame and split points list then return a InfoValue instance;
    Just for continuous variable
    :param df:
    :param var:
    :param split_list:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    df_missing = pd.DataFrame()
    civ = InfoValue()
    civ.var_name = var
    civ.split_list = split_list
    if is_continuous == 1:
        df_notmissing = df.loc[df[var] != global_numeric_missing]
        #print 'df_notmissing', df_notmissing.shape[0]
        if df.loc[df[var] == global_numeric_missing].shape[0] > 0:
            civ.split_list = [global_numeric_missing] + split_list
            df_missing = df.loc[df[var] == global_numeric_missing]
    else:
        civ.split_list = split_list
        df_notmissing = df.loc[df[var] != missing_woe]
        #print 'df_notmissing', df_notmissing.shape[0]
        if (missing_woe != 0) & (df.loc[df[var] == missing_woe].shape[0] > 0):
            df_missing = df.loc[df[var] == missing_woe]

    #print 'split_list: ', civ.split_list
    civ.sub_total_sample_num = []
    civ.positive_sample_num = []
    civ.negative_sample_num = []
    civ.sub_total_num_percentage = []
    civ.positive_rate_in_sub_total = []
    civ.weight_freq = []
    civ.weight_positive_freq = []
    civ.weight_negative_freq = []
    civ.cum_weight_freq = []
    civ.cum_weight_positive_freq = []
    civ.cum_weight_negative_freq = []
    civ.perc_cum_weight_freq = []
    civ.perc_cum_weight_positive_freq = []
    civ.perc_cum_weight_negative_freq = []
    civ.ks_list = []
    civ.maxks = 0
    civ.pos_trend = 0
    civ.neg_trend = 0
    civ.linearity = 0

    
    # todo when missing
    range_start = 0
    if df_missing.shape[0] > 0:
        gd = calulate_iv(df_missing,var,global_bt,global_gt,global_wbt,global_wgt)
        woei, ivi = gd['woei'],gd['ivi']
        civ.woe_list.append(woei)
        civ.iv_list.append(ivi)
        civ.sub_total_sample_num.append(df_missing.shape[0])
        civ.positive_sample_num.append(gd['positive_sample_num'])
        civ.negative_sample_num.append(gd['negative_sample_num'])
        civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
        civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
        civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])
        civ.weight_freq.append(gd['weight_positive_freq'] + gd['weight_negative_freq'])
        civ.weight_positive_freq.append(gd['weight_positive_freq'])
        civ.weight_negative_freq.append(gd['weight_negative_freq'])
        civ.cum_weight_freq.append(gd['weight_positive_freq'] + gd['weight_negative_freq'])
        civ.cum_weight_positive_freq.append(gd['weight_positive_freq'])
        civ.cum_weight_negative_freq.append(gd['weight_negative_freq'])
        range_start = 1
        
    dfcp = df_notmissing[:]

    for i in range(range_start, civ.split_list.__len__()):
        #if civ.split_list.__len__() > 1:
            dfi = dfcp[dfcp[var] <= civ.split_list[i]]
            dfcp = dfcp[dfcp[var] > civ.split_list[i]]
            gd = calulate_iv(dfi, var,global_bt,global_gt,global_wbt,global_wgt)
            woei, ivi = gd['woei'],gd['ivi']
            civ.woe_list.append(woei)
            civ.iv_list.append(ivi)
            civ.sub_total_sample_num.append(dfi.shape[0])
            civ.positive_sample_num.append(gd['positive_sample_num'])
            civ.negative_sample_num.append(gd['negative_sample_num'])
            civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
            civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
            civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])
            civ.weight_freq.append(gd['weight_positive_freq'] + gd['weight_negative_freq'])
            civ.weight_positive_freq.append(gd['weight_positive_freq'])
            civ.weight_negative_freq.append(gd['weight_negative_freq'])
            if i==0:
                civ.cum_weight_freq.append(gd['weight_positive_freq'] + gd['weight_negative_freq'])
                civ.cum_weight_positive_freq.append(gd['weight_positive_freq'])
                civ.cum_weight_negative_freq.append(gd['weight_negative_freq'])
            else:
                civ.cum_weight_freq.append(civ.cum_weight_freq[i-1] + gd['weight_positive_freq'] + gd['weight_negative_freq'])
                civ.cum_weight_positive_freq.append(civ.cum_weight_positive_freq[i-1] + gd['weight_positive_freq'])
                civ.cum_weight_negative_freq.append(civ.cum_weight_negative_freq[i-1] + gd['weight_negative_freq'])
                if gd['woei'] > civ.woe_list[i-1]:
                    civ.pos_trend = civ.pos_trend + 1
                else:
                    civ.neg_trend = civ.neg_trend + 1

    if dfcp.shape[0]>0:
        gd = calulate_iv(dfcp, var,global_bt,global_gt,global_wbt,global_wgt)
        woei, ivi = gd['woei'],gd['ivi']
        civ.woe_list.append(woei)
        civ.iv_list.append(ivi)
        civ.sub_total_sample_num.append(dfcp.shape[0])
        civ.positive_sample_num.append(gd['positive_sample_num'])
        civ.negative_sample_num.append(gd['negative_sample_num'])
        civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
        civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
        civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])
        civ.weight_freq.append(gd['weight_positive_freq'] + gd['weight_negative_freq'])
        civ.weight_positive_freq.append(gd['weight_positive_freq'])
        civ.weight_negative_freq.append(gd['weight_negative_freq'])
        #if civ.split_list.__len__() > 1:
        civ.cum_weight_freq.append(civ.cum_weight_freq[civ.split_list.__len__()-1] + gd['weight_positive_freq'] + gd['weight_negative_freq'])
        civ.cum_weight_positive_freq.append(civ.cum_weight_positive_freq[civ.split_list.__len__()-1] + gd['weight_positive_freq'])
        civ.cum_weight_negative_freq.append(civ.cum_weight_negative_freq[civ.split_list.__len__()-1] + gd['weight_negative_freq'])
        if gd['woei'] > civ.woe_list[civ.split_list.__len__()-1]:
            civ.pos_trend = civ.pos_trend + 1
        else:
            civ.neg_trend = civ.neg_trend + 1
        '''
        else:
            civ.cum_weight_freq.append(gd['weight_positive_freq'] + gd['weight_negative_freq'])
            civ.cum_weight_positive_freq.append(gd['weight_positive_freq'])
            civ.cum_weight_negative_freq.append(gd['weight_negative_freq'])
        '''    
                        
    civ.iv = sum(civ.iv_list)
    civ.total_weight_freq = sum(civ.weight_freq)
    civ.total_weight_positive_freq = sum(civ.weight_positive_freq)
    civ.total_weight_negative_freq = sum(civ.weight_negative_freq)
    
    civ.perc_cum_weight_freq = [(x+0.0001)*100.0/ civ.total_weight_freq for x in civ.cum_weight_freq]
    civ.perc_cum_weight_positive_freq = [(x+0.0001)*100.0/ civ.total_weight_positive_freq for x in civ.cum_weight_positive_freq]
    civ.perc_cum_weight_negative_freq = [(x+0.0001)*100.0/ civ.total_weight_negative_freq for x in civ.cum_weight_negative_freq]
    
    civ.ks_list = list(np.abs(np.array(civ.perc_cum_weight_positive_freq) - np.array(civ.perc_cum_weight_negative_freq)))
    civ.maxks = np.max(civ.ks_list)
    civ.linearity = np.abs(civ.pos_trend - civ.neg_trend)*100.0/(civ.split_list.__len__()+0.0001)
    '''
    print ('-'*80)
    print 'split\t', 'weight_freq\t','weight_positive_freq\t','weight_negative_freq\t', 'perc_cum_weight_freq\t', 'ks', 'maxks', 'pos_trend', 'neg_trend', 'woe', 'linearity'
    print ('-'*80)
    for i in range(0, split_list.__len__()):
        print civ.split_list[i], civ.weight_freq[i], civ.weight_positive_freq[i], civ.weight_negative_freq[i], civ.perc_cum_weight_freq[i], civ.ks_list[i], civ.maxks, civ.pos_trend, civ.neg_trend, civ.woe_list[i], civ.linearity 
    '''
    return civ
    
def woe_trans(dvar,civ):
    # replace the var value with the given woe value
    var = copy.deepcopy(dvar)
    if not civ.is_discrete:
        if civ.woe_list.__len__()>1:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([i for i in civ.split_list])
            split_list.append(float("inf"))

            for i in range(civ.woe_list.__len__()):
                var[(dvar > split_list[i]) & (dvar <= split_list[i+1])] = civ.woe_list[i]
        else:
            var[:] = civ.woe_list[0]
    else:
        split_map = {}
        for i in range(civ.split_list.__len__()):
            for j in range(civ.split_list[i].__len__()):
                split_map[civ.split_list[i][j]] = civ.woe_list[i]
        var = var.map(split_map)

    return var

def proc_woe_discrete(df,var,global_bt,global_gt,global_wbt,global_wgt,min_sample,global_numeric_missing,global_categorical_missing,alpha=0.01):
    '''
    process woe transformation of discrete variables
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process discrete variable:'+str(var)
    print(s.center(60, '-'))

    df_notmissing = df.loc[df[var] != global_categorical_missing]
    
    df = df[[var,'target','weight']]
    div = DisInfoValue()
    div.var_name = var
    rdict = {}
    cpvar = df[var]
    missing_woe = 0
    # print(df.head(5))
    # print('np.unique(df[var])：',np.unique(df[var]))
    for var_value in np.unique(df[var]):
        # Here come with a '==',in case type error you must do Nan filling process firstly
        df_temp = df[df[var] == var_value]
        gd = calulate_iv(df_temp,var,global_bt,global_gt,global_wbt,global_wgt)
        woei, ivi = gd['woei'],gd['ivi']
        div.origin_value.append(var_value)
        div.woe_before.append(woei)
        if var_value == global_categorical_missing:
            missing_woe = woei
        rdict[var_value] = woei
        # print(var_value,woei,ivi)
    #print(cpvar.head(5))
    cpvar = cpvar.map(rdict)
    #print(cpvar.head(5))
    df.loc[:,(var)] = cpvar
    #print(df.head(5))
    df_notmissing = df.loc[df[var] != missing_woe]
    #print('np.unique(df_notmissing[var])：',np.unique(df_notmissing[var]))
    
    iv_tree = binning_data_split(df_notmissing,var,global_bt,global_gt,global_wbt,global_wgt,min_sample,alpha)

    # Traversal tree, get the segmentation point
    split_list = []
    search(iv_tree, split_list)
    #print 'after search split', split_list
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    #print 'after unique', split_list
    split_list.sort()

    # Segmentation point checking and processing
    split_list = check_point(df_notmissing, var, split_list, min_sample)
    split_list.sort()
    #print 'split_list checkpoint', split_list


    split_list_temp = []
    origin_value = []
    woe_before = []
    a = []
    range_start = 0
    if missing_woe != 0:
        split_list = [missing_woe] + split_list
        split_list_temp.append(missing_woe)
        split_list_temp.append(float("-inf"))
        split_list_temp.extend([i for i in split_list if i is not missing_woe])
        split_list_temp.append(float("inf"))
        origin_value.append(global_categorical_missing)
        woe_before.append(missing_woe)
        a.append([global_categorical_missing])
        origin_value.extend([i for i in div.origin_value if i is not global_categorical_missing])
        woe_before.extend([i for i in div.woe_before if i is not missing_woe])
        range_start = 1
    else:
        origin_value.extend([i for i in div.origin_value])
        woe_before.extend([i for i in div.woe_before])
        split_list_temp.append(float("-inf"))
        split_list_temp.extend([i for i in split_list])
        split_list_temp.append(float("inf"))
            
    # print 'split_list after adding missing woe', split_list
    # print 'split_list_temp', split_list_temp
    # print 'origin_value', origin_value
    # print 'woe_before', woe_before
    civ = format_iv_split(df,var,split_list,global_bt,global_gt,global_wbt,global_wgt,0,global_numeric_missing,global_categorical_missing,missing_woe)
    civ.is_discrete = 1

    # The below logic replaces the woe values with the original value and orders the split_list by the woe                 
    # dealing with the missing bucket
    
    for i in range(range_start,split_list_temp.__len__()-1):
        temp = []
        # not_in_list =[]
        for j in range(range_start, origin_value.__len__()):
            if (woe_before[j]>split_list_temp[i]) & (woe_before[j]<=split_list_temp[i+1]):
                temp.append(origin_value[j])
            '''
            # when the search returns only one value/node, fix this part, that is, split length = 3
            elif (split_list_temp.__len__() == 3):
                not_in_list.append(origin_value[j])
            '''
        if temp != [] :
            a.append(temp)

    civ.split_list = a
    # print 'civ.split_list', civ.split_list
    return civ

def proc_woe_discrete_rebin(df,var,rebin_list,global_bt,global_gt,global_wbt,global_wgt,min_sample,global_numeric_missing,global_categorical_missing,alpha=0.01):
    '''
    process woe transformation of discrete variables based on the re-binning logic
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'rebin process discrete variable:'+str(var)
    print(s.center(60, '-'))
    import ast
    df = df[[var,'target','weight']]
    for rebin_val in rebin_list:
        #df.loc[df[var].isin(eval(rebin_val)), (var)] = str(rebin_val).strip('[]')  # ast.literal_eval(str(rebin_val))
        df.loc[df[var].isin(eval('['+str(rebin_val)+']')), (var)] = str(rebin_val)  # ast.literal_eval(str(rebin_val))
        df.loc[:,(var)] = df[var].astype(object)

    div = DisInfoValue()
    div.var_name = var
    rdict = {}
    cpvar = df[var]
    missing_woe = 0
    # print('np.unique(df[var])：',np.unique(df[var]))
    for var_value in np.unique(df[var]):
        # Here come with a '==',in case type error you must do Nan filling process firstly
        df_temp = df[df[var] == var_value]
        gd = calulate_iv(df_temp,var,global_bt,global_gt,global_wbt,global_wgt)
        woei, ivi = gd['woei'],gd['ivi']
        div.origin_value.append(var_value)
        div.woe_before.append(woei)
        if var_value == global_categorical_missing:
            missing_woe = woei
        rdict[var_value] = woei
        # print(var_value,woei,ivi)
    cpvar = cpvar.map(rdict)

    df.loc[:,var] = cpvar
    # print(df.head(5))
    #df.loc[:,var] = cpvar.loc[:,1]

    split_list = list(np.unique(df[[var]]))
    split_list.sort()
    
    # Commenting the section below because this would override the rebinning logic specified if the # of sample data points are < min_sample
    # Segmentation point checking and processing
    # split_list = check_point(df, var, split_list, min_sample)
    # split_list.sort()
        
    civ = format_iv_split(df,var,split_list,global_bt,global_gt,global_wbt,global_wgt,0,global_numeric_missing,global_categorical_missing)
    civ.is_discrete = 1
    
    split_list_temp = []
    split_list_temp.append(float("-inf"))
    split_list_temp.extend([i for i in split_list])
    split_list_temp.append(float("inf"))

    a = []
    for i in range(split_list_temp.__len__() - 1):
        temp = []
        for j in range(div.origin_value.__len__()):
            if (div.woe_before[j]>split_list_temp[i]) & (div.woe_before[j]<=split_list_temp[i+1]):
                temp.append(div.origin_value[j])

        if temp != [] :
            a.append(temp)

    civ.split_list = a    
    return civ
    

def proc_woe_continuous(df,var,global_bt,global_gt,global_wbt,global_wgt,min_sample,global_numeric_missing,global_categorical_missing,alpha=0.01):
    '''
    process woe transformation of discrete variables
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process continuous variable:'+str(var)
    print(s.center(60, '-'))
    df_notmissing = df.loc[df[var] <> global_numeric_missing]
    df_notmissing = df_notmissing[[var,'target','weight']]
    iv_tree = binning_data_split(df_notmissing, var,global_bt,global_gt,global_wbt,global_wgt,min_sample,alpha)

    # Traversal tree, get the segmentation point
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # Segmentation point checking and processing
    split_list = check_point(df_notmissing, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df,var,split_list,global_bt,global_gt,global_wbt,global_wgt,1,global_numeric_missing,global_categorical_missing)

    return civ

def proc_woe_continuous_rebin(df,var,split_list,global_bt,global_gt,global_wbt,global_wgt,min_sample,global_numeric_missing,global_categorical_missing,alpha=0.01):
    '''
    process woe transformation of continuous variables based on the re-binning logic
    :param df:
    :param var:
    :param split_list:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'rebin process continuous variable:'+str(var)
    print(s.center(60, '-'))
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # Segmentation point checking and processing
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df,var,split_list,global_bt,global_gt,global_wbt,global_wgt,1,global_numeric_missing,global_categorical_missing)

    return civ
  
def fillna(dataset,bin_var_list,discrete_var_list,continuous_filler=-1,discrete_filler='missing'):
    """
    fill the null value in the dataframe inpalce
    :param dataset: input dataset ,pandas.DataFrame type
    :param bin_var_list:  continuous variables name list
    :param discrete_var_list: discretevvvv variables name list
    :param continuous_filler: the value to fill the null value in continuous variables
    :param discrete_filler: the value to fill the null value in discrete variables
    :return: null value,replace null value inplace
    """
    for var in [tmp for tmp in bin_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = continuous_filler

    for var in [tmp for tmp in discrete_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = discrete_filler


def process_train_woe(infile_path=None,outfile_path=None,rst_path=None,config_path=None,rebin_feature_path=None):
    print('run into process_train_woe: \n',time.asctime(time.localtime(time.time())))
    data_path = infile_path
    cfg = config.config()
    cfg.load_file(config_path,data_path,rebin_feature_path)
    bin_var_list = [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]
    orig_dataset_train = cfg.dataset_train
    
    # change feature dtypes
    change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    rst = []

    print('cfg.global_bt',cfg.global_bt)
    print('cfg.global_gt', cfg.global_gt)
    print('cfg.global_wbt',cfg.global_wbt)
    print('cfg.global_wgt', cfg.global_wgt)

    # process woe transformation of continuous variables
    print('process woe transformation of continuous variables: \n',time.asctime(time.localtime(time.time())))
    for var in bin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_numeric_missing
        rst.append(proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.global_wbt,cfg.global_wgt,cfg.min_sample,cfg.global_numeric_missing,cfg.global_categorical_missing,alpha=0.05))

    # process woe transformation of continuous variables based on the re-binning logic provided
    print('process woe transformation of continuous variables based on rebin logic: \n',time.asctime(time.localtime(time.time())))
    rebin_var_list = [tmp for tmp in cfg.rebin_var_list if tmp in list(cfg.dataset_train.columns)]
    for var in rebin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_numeric_missing
        var_df = cfg.dataset_rebin.loc[cfg.dataset_rebin['var_name'] == var]
        split_list = list(np.unique(var_df[['split']].astype(float)))
        rst.append(proc_woe_continuous_rebin(cfg.dataset_train,var,split_list,cfg.global_bt,cfg.global_gt,cfg.global_wbt,cfg.global_wgt,cfg.min_sample,cfg.global_numeric_missing,cfg.global_categorical_missing,alpha=0.05))
    
    # process woe transformation of discrete variables
    print('process woe transformation of discrete variables: \n',time.asctime(time.localtime(time.time())))
    discrete_var_list = [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_categorical_missing
        rst.append(proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.global_wbt,cfg.global_wgt,cfg.min_sample,cfg.global_numeric_missing,cfg.global_categorical_missing,alpha=0.05))

    # process woe transformation of discrete variables based on re-binning logic
    print('process woe transformation of discrete variables based on rebin logic: \n',time.asctime(time.localtime(time.time())))
    rebin_discrete_var_list = [tmp for tmp in cfg.rebin_discrete_var_list if tmp in list(cfg.dataset_train.columns)]
    for var in [tmp for tmp in cfg.rebin_discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_categorical_missing
        var_df = cfg.dataset_rebin.loc[cfg.dataset_rebin['var_name'] == var]
        # var_df['split'] = var_df['split'].astype(object)
        var_df.loc[:,'split'] = var_df['split'].astype(object)
        rebin_list = list(np.unique(var_df[['split']]))
        rst.append(proc_woe_discrete_rebin(cfg.dataset_train,var,rebin_list,cfg.global_bt,cfg.global_gt,cfg.global_wbt,cfg.global_wgt,cfg.min_sample,cfg.global_numeric_missing,cfg.global_categorical_missing,alpha=0.05))

    feature_detail = woeeval.eval_feature_detail(rst, outfile_path)

    import pandas as pd
    pd.options.display.float_format = '{:.3f}'.format
    for var in bin_var_list+rebin_var_list+discrete_var_list+rebin_discrete_var_list:
        missing_obs = cfg.dataset_train.loc[cfg.dataset_train[var].isin([cfg.global_numeric_missing, cfg.global_categorical_missing])].shape[0]
        print 'variable = ',var,'\t# obs = ',orig_dataset_train[var].shape[0],'\t# valid = ',(orig_dataset_train[var].shape[0] - missing_obs),'\t% valid = ',(orig_dataset_train[var].shape[0] - missing_obs)*100.0/(orig_dataset_train[var].shape[0])
        df = feature_detail.loc[feature_detail['var_name'] == var]
        print(df[['split_list','sub_total_sample_num','positive_sample_num'
            ,'weight_positive_freq','weight_negative_freq'
            ,'perc_cum_weight_freq','perc_cum_weight_positive_freq','perc_cum_weight_negative_freq'
            ,'woe_list','iv_list','ks_list']])
        woeeval.plot_woe(df, var)

    s = 'summary of WOE transformation'
    print(s.center(60, '-'))
    smry_df = feature_detail[['var_name', 'iv', 'maxks', 'linearity']].drop_duplicates().sort_values('iv', ascending=False)
    print(smry_df)
    
    print('save woe transformation rule into pickle: \n',time.asctime(time.localtime(time.time())))
    output = open(rst_path, 'wb')
    pickle.dump(rst,output)
    output.close()

    return feature_detail,rst


def process_woe_trans(in_data_path=None,rst_path=None,out_path=None,config_path=None,rebin_feature_path=None):
    cfg = config.config()
    cfg.load_file(config_path, in_data_path,rebin_feature_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_numeric_missing

    rebin_var_list = [tmp for tmp in cfg.rebin_var_list if tmp in list(cfg.dataset_train.columns)]
    for var in rebin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_numeric_missing

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_categorical_missing

    rebin_discrete_var_list = [tmp for tmp in cfg.rebin_discrete_var_list if tmp in list(cfg.dataset_train.columns)]
    for var in [tmp for tmp in cfg.rebin_discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = cfg.global_categorical_missing
        # first use the split values from features_rebin to re-bin the existing dataset, that way the WOE transformation is applied correctly
        rebin_var_df = cfg.dataset_rebin.loc[cfg.dataset_rebin['var_name'] == var]
        rebin_var_df['split'] = rebin_var_df['split'].astype(object)
        rebin_list = list(np.unique(rebin_var_df[['split']]))
        for rebin_val in rebin_list:
            cfg.dataset_train.loc[cfg.dataset_train[var].isin(eval('['+str(rebin_val)+']')), (var)] = str(rebin_val)

    change_feature_dtype(cfg.dataset_train, cfg.variable_type)

    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    # Training dataset Woe Transformation
    for r in rst:
        cfg.dataset_train[r.var_name] = woe_trans(cfg.dataset_train[r.var_name], r)
    # Output to a csv
    cfg.dataset_train.to_csv(out_path)
