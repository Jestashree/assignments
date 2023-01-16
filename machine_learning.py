import pandas as pd
import numpy as np

dataset = pd.read_csv("SDN_traffic.csv")

print(dataset.head())
print(dataset.info())
print(dataset.describe())
print(dataset.duplicated())

x= dataset[['forward_bps_var',
            "tp.sre", "tp.dst", "nw_proto",
            "forward pe", "forward_bc", "forward_pl",
            "forward piat", "forward_pps", "forward_bps", "forward_pl_mean",
            "forward piat mean", "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
            "forward.pps var", "forward_pl_q1", "forward_pl_q3",
            "forward piat.g1", "forward_piat_q3", "forward_pl_max", "forward_pl_min", 
            "forward plat.max", "forward.piat_win", "forward_pps_max", "forward_pps_min",
            "forward bps_max", "forward_bps_min", "forward_duration", "forward_size_packets",
            "forward size bytes", "reverse_pc", "reverse_bc", "reverse.pl", "reverse piat", "reverse PRS", 
            "reverse_bps", "reverse_pl_mean", "reverse_piat.mean", "reverse_pps_nean","reverse bas mean", "reverse_pl_var",
            "reverse plin", "reverse_pl", "reverse_piat", "reverse_piat_var", "reverse_pps_var", "reverse_bps_var",
            "reverse_piat q1", "reverse_pl_q3", "reverse_piat_max","reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
            "reverse_piat_q3", "reverse_pl_max", "reverse bps_max", "reverse_bps_min", "reverse_duration", "reverse_size_packets", "reverse_size_bytes"]]

X.Loc[1877, 'forward_bps_var'] = float(11968865203349)
X.Loc[9131, 'forward_bps_var'] = float(12880593884833)
X.loc[2381, 'forward_bps_var'] = float(39987497172945)
X.Loc[2562, 'forward_bps_var'] = float(663388742992)
X.Loc[1931, 'Torward_bps_var'] = float(37770223877794)
X.Loc[2078, 'forward_bps_var'] = float(9822747730895)
X.loc[2567, 'Torward_bps_var'] = float(37778223877794) 
X.Loc[2586, 'Torward_bps_var'] = float(97227875883751)
X.Loc[2754, 'forward_bps_var'] = float(18789751483737)
X.loc[2765, 'Torward_bps_var'] = float(33969277035759)
X.Loc[2984, 'forward_bps_var'] = float(39284786962856)
X.loc[3844, 'forward_bps_var'] = float(9169996863653)
X.loc[3349, 'Torward_bps_var'] = float(37123283690575)
X.Loc[3507, 'forward bps_var'] = float(61019864598464)
X.loc[3610, 'forward_bps_var'] = float(46849628984872)
X.Loc[3717, 'forward_bps_var'] = float(97158873841506)
X.loc[3845, 'forward_bps_var'] = float(11968865203349)
X.loc[3868, 'forward_bps_var'] = float(85874278395372)

#XX.drop([1877.1931.2070.2381.2562.2567.2586.2754.2765.2904 3044.3349 3507.3630.3717.3845.38681,axis=0)

X["forward bps_var"]= pd.to_numeric(x["forward_bps_var"])
print(X.info())
Y = dataset[["category"]]
Y = Y.to_numpy()
Y = Y.ravel()
Labels, uniques = pd.factorize(Y)
Y = Labels
Y = Y.ravel()

import scipy.stats as stats
X = stats.zscore (X)
X = np.nan_to_num(x)

#https://scikit-Learn.org/stable/modules/multiclass.html Multiclass and multioutput algorithes
#Train decision tree classifier

from sklearn.model_selection import train_test_split
X_train, X.test, Y_train, Y_test = train_test_split(X, Y, random_states=0, test_size=0.3) 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, fi_score, classification_report
from sklearn.model.selection import cross_val_score, KFold
clf = DecisionTreeClassifier(randon_state=0, max_depth=2)
clf.fit(X_train, Y_train)

#Evaluation metrics performance
cv = KFold(n_splits=10, randos_state=0, shuffle=True)
accuracy = clf.score (X_test, Y_test)
KFold10_accuracy = cross_val_score(clf, X_train, Y.train, scoring='accuracy', cv=cv, n_jobs=-1)
print(KFold10_accuracy.mean())
predict = clf.predict(X_test)
cm = confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict,average='weighted', labels=np.unique(predict))
recall = recall_score(Y_test, predict, average='weighted', labels=np.unique(predict)) 
fiscoreMacro = f1_score(Y_test, predict, averages='macro', labels=np.unique(predict)) 
print(classification_report(Y_test, predict, target_names=uniques))

#Find out the most important 18 features

importance = clf.feature_importances_
important_features_dict = {}
for idx, val in enumerate(importance): 
    important_features_dict[idx] = val
important_features_list = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True) 
print(f'10 most important features: {important_features_List[:10]}')

#Plot decision tree and confusion matrix
fn=['forward_bps_var',
    "tp.src" "tp_dst", "nw_proto",
    "forward_pc" "forward_bc" "forward.pl",
    "forward_plat" "forward_pps" "forward_bps", "forward_pl_nean",
    "forward_piat.mean", "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
    "forward_pps_var", "forward_pl_q1", "forward_pl_q3",
    "forward_piat_q1", "forward_piat_q3", "forward_pl_max", "forward_pl_min",
    "forward_pist_max", "forward_piat_min", "forward_pps_max", "forward_pps_min",
    "forward_bps_max", "forward_bps_min", "forward_duration", "forward_size_packets",
    "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat", "reverse_pps", 
    "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean"
    "reverse_bps_nean", "reverse_pl_ver", "reverse_piat var", "reverse_pps_var", "reverse_bps_var",
    "reverse_pl_q1", "reverse_pl_43", "reverse_piat_q1", "reverse_piat_q3", "reverse_pl_max",
    "reverse_pl_min" "reverse piat_max", "reverse_plat_min", "reverse_pps_max", "reverse_pps_min",
    "reverse_bps_max", "reverse_bps_min", "reverse_duration", "reverse_size_packets", "reverse_size_bytes"]
la = ['WWW','DNS','FTP','ICMP','P2P','VOIP']
plt.figure(1,dip=300)
fig = tree.plot_tree(clf,filled=True, feature_names=fn, class_names=la)
plt.title("Decision tree trained on all the features")
plt.show()
import seaborn as sn
import matplotlib.pyplot as plt
labels = uniques
plt.figure(2,figsize=(5, 2))
plt.title("confusion Matrix", fontsize=10)
# normalise
cmnew = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sn.heatmap(cmnew, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels)