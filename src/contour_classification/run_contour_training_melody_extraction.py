import contour_utils as cc
import experiment_utils as eu
import mv_gaussian as mv
import clf_utils as cu
import generate_melody as gm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import metrics
import sklearn
import pandas as pd
import numpy as np
import random
import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import boxcox

from contour_utils  import getFeatureInfo



# 2

plt.ion()


mel_type=2

reload(eu)

scores = []
scores_nm = []

# EDIT: For MedleyDB
#with open('melody_trackids.json', 'r') as fhandle:
#    track_list = json.load(fhandle)

# For Orchset
with open('melody_trackids_orch.json', 'r') as fhandle:
    track_list = json.load(fhandle)


track_list = track_list['tracks']

# mdb_files, splitter = eu.create_splits(test_size=0.15)

dset_contour_dict, dset_annot_dict = \
        eu.compute_all_overlaps(track_list, meltype=mel_type)

mdb_files, splitter = eu.create_splits(test_size=0.25)

for i in range(4):
    for train, test in splitter:
        random.shuffle(train)
        n_train = len(train) - (len(test)/2)
        train_tracks = mdb_files[train[:n_train]]
        valid_tracks = mdb_files[train[n_train:]]
        test_tracks = mdb_files[test]

        train_contour_dict = {k: dset_contour_dict[k] for k in train_tracks}
        valid_contour_dict = {k: dset_contour_dict[k] for k in valid_tracks}
        test_contour_dict = {k: dset_contour_dict[k] for k in test_tracks}

        train_annot_dict = {k: dset_annot_dict[k] for k in train_tracks}
        valid_annot_dict = {k: dset_annot_dict[k] for k in valid_tracks}
        test_annot_dict = {k: dset_annot_dict[k] for k in test_tracks}

        reload(eu)
        olap_stats, zero_olap_stats = eu.olap_stats(train_contour_dict)
        OLAP_THRESH = 0.5
        train_contour_dict, valid_contour_dict, test_contour_dict = \
            eu.label_all_contours(train_contour_dict, valid_contour_dict, \
                                  test_contour_dict, olap_thresh=OLAP_THRESH)
        len(train_contour_dict)

        reload(cc)

        anyContourDataFrame = dset_contour_dict[dset_contour_dict.keys()[0]]


        feats, idxStartFeatures, idxEndFeatures = getFeatureInfo(anyContourDataFrame)

        X_train, Y_train = cc.pd_to_sklearn(train_contour_dict,idxStartFeatures,idxEndFeatures)
        X_valid, Y_valid = cc.pd_to_sklearn(valid_contour_dict,idxStartFeatures,idxEndFeatures)
        X_test, Y_test = cc.pd_to_sklearn(test_contour_dict,idxStartFeatures,idxEndFeatures)
        np.max(X_train,0)


        # x,y = cc.pd_to_sklearn(train_contour_dict['AClassicEducation_NightOwl'])
        # train_contour_dict['AClassicEducation_NightOwl']
        # contour_data = train_contour_dict['AClassicEducation_NightOwl']
        # x[68]
        # train_contour_dict['AClassicEducation_NightOwl'].loc[68,:]
        #
        # X_train_boxcox, X_test_boxcox = mv.transform_features(X_train, X_test)
        # rv_pos, rv_neg = mv.fit_gaussians(X_train_boxcox, Y_train)
        #
        # M_train, M_test = mv.compute_all_melodiness(X_train_boxcox, X_test_boxcox, rv_pos, rv_neg)
        #
        # reload(mv)
        # reload(eu)
        # melodiness_scores = mv.melodiness_metrics(M_train, M_test, Y_train, Y_test)
        # best_thresh, max_fscore,vals = eu.get_best_threshold(Y_test, M_test)
        # print "best threshold = %s" % best_thresh
        # print "maximum achieved f score = %s" % max_fscore
        # print melodiness_scores

        reload(cu)
        best_depth, max_cv_accuracy, plot_dat = cu.cross_val_sweep(X_train, Y_train,plot = False)
        print best_depth
        print max_cv_accuracy

        df = pd.DataFrame(np.array(plot_dat).transpose(), columns=['max depth', 'accuracy', 'std'])


        clf = cu.train_clf(X_train, Y_train, best_depth)

        reload(cu)
        P_train, P_valid, P_test = cu.clf_predictions(X_train, X_valid, X_test, clf)
        clf_scores = cu.clf_metrics(P_train, P_test, Y_train, Y_test)
        print clf_scores['test']


        reload(eu)
        best_thresh, max_fscore, plot_data = eu.get_best_threshold(Y_valid, P_valid)
        print "besth threshold = %s" % best_thresh
        print "maximum achieved f score = %s" % max_fscore


        for key in test_contour_dict.keys():
            test_contour_dict[key] = eu.contour_probs(clf, test_contour_dict[key],idxStartFeatures,idxEndFeatures)


        reload(gm)
        mel_output_dict = {}
        for i, key in enumerate(test_contour_dict.keys()):
            print key
            mel_output_dict[key] = gm.melody_from_clf(test_contour_dict[key], prob_thresh=best_thresh)





        reload(gm)

        mel_scores = gm.score_melodies(mel_output_dict, test_annot_dict)


        overall_scores = \
            pd.DataFrame(columns=['VR', 'VFA', 'RPA', 'RCA', 'OA'],
                         index=mel_scores.keys())
        overall_scores['VR'] = \
            [mel_scores[key]['Voicing Recall'] for key in mel_scores.keys()]
        overall_scores['VFA'] = \
            [mel_scores[key]['Voicing False Alarm'] for key in mel_scores.keys()]
        overall_scores['RPA'] = \
            [mel_scores[key]['Raw Pitch Accuracy'] for key in mel_scores.keys()]
        overall_scores['RCA'] = \
            [mel_scores[key]['Raw Chroma Accuracy'] for key in mel_scores.keys()]
        overall_scores['OA'] = \
            [mel_scores[key]['Overall Accuracy'] for key in mel_scores.keys()]

        scores.append(overall_scores)

        print "Overall Scores"
        overall_scores.describe()



    # Tests with multilines

    #
    # from sys import path
    # currpath = os.getcwd()
    # from sys import path
    # path.append('../melody-SFContour')
    # path.append('../')
    # os.chdir("../melody-SFContour")
    # import optparse
    # parser = optparse.OptionParser("")
    # (options, args) = parser.parse_args([])
    # options.Pchangevx = 1
    # options.wNoteTrans = 1
    # options.wContourTrans = 1
    # options.wInstrTrans = 1
    # options.scale = 1
    # options.scaleSurr = 1
    # options.scalePan = 0
    # options.hopsizeInSamples = 256
    # options.hopsizeInSamples = 441
    # import generate_melody_ml as gm2
    # reload(gm2)
    # mel_output_dict_nm = {}
    # for i, key in enumerate(test_contour_dict.keys()):
    #     print key
    #     mel_output_dict_nm[key] = gm2.melody_from_clf(test_contour_dict[key], prob_thresh=best_thresh,options=options)
    # os.chdir(currpath)
    # print os.getcwd()
    # os.chdir("../contour_classification")
    #
    # import generate_melody as gm
    # reload(gm)
    #
    # # key="Beethoven-S3-I-ex2"
    # # df = mel_output_dict[key]
    # #
    # # df_pos = df[df > 0]
    # # df_zero = df[df == 0]
    # # df_neg = df[df < 0]
    # # plt.plot(df_pos.index, df_pos.values, ',g')
    # # plt.plot(df_zero.index, df_zero.values, ',y')
    # # plt.plot(df_neg.index, -1.0*df_neg.values, ',r')
    # # plt.show()
    # #
    # # df.index
    #
    # #df2 = mel_output_dict_nm[key]
    # #times, pitches = df2
    # #pitches[:,0]
    # #df_zero = df[df == 0]
    # #df_neg = df[df < 0]
    # #plt.plot(df_pos, df_pos, ',g')
    # #plt.plot(df_zero, df_zero, ',y')
    # #plt.plot(df_neg, -1.0*df_neg, ',r')
    # #plt.show()
    #
    # mel_scores_nm = gm.score_melodies(mel_output_dict_nm, test_annot_dict)
    #
    # overall_scores = \
    #     pd.DataFrame(columns=['VR', 'VFA', 'RPA', 'RCA', 'OA'],
    #                  index=mel_scores_nm.keys())
    # overall_scores['VR'] = \
    #     [mel_scores_nm[key]['Voicing Recall'] for key in mel_scores_nm.keys()]
    # overall_scores['VFA'] = \
    #     [mel_scores_nm[key]['Voicing False Alarm'] for key in mel_scores_nm.keys()]
    # overall_scores['RPA'] = \
    #     [mel_scores_nm[key]['Raw Pitch Accuracy'] for key in mel_scores_nm.keys()]
    # overall_scores['RCA'] = \
    #     [mel_scores_nm[key]['Raw Chroma Accuracy'] for key in mel_scores_nm.keys()]
    # overall_scores['OA'] = \
    #     [mel_scores_nm[key]['Overall Accuracy'] for key in mel_scores_nm.keys()]
    #
    # print "Overall Scores NM"
    # overall_scores.describe()
    # scores_nm.append(overall_scores)


print "End"


allscores = scores[0]
for i in range(1,len(scores),1):
    allscores = allscores.append(scores[i])
    print i
    print (len(allscores))


allscores.to_csv('allscoresNoTonal.csv')
from pickle import dump
picklefile = 'allscores'
with open(picklefile, 'wb') as handle:
    dump(allscores, handle)
print allscores.describe()

np.argsort(clf.feature_importances_)
np.sum(clf.feature_importances_)
[feats[k] for k in np.argsort(clf.feature_importances_)]


#
# allscores_nm = scores_nm[0]
# for i in range(1,len(scores_nm),1):
#     allscores_nm = allscores_nm.append(scores_nm[i])
#     print i
#     print (len(allscores_nm))
#
# allscores_nm.describe()
#
# from pickle import dump
# picklefile = 'allscores_nm'
# with open(picklefile, 'wb') as handle:
#     dump(allscores_nm, handle)
#
#
#
#
# picklefile = 'allscores'
#
# from pickle import load
# with open(picklefile, 'rb') as handle:
#     b = load(handle)
