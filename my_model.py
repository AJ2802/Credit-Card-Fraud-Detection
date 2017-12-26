import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

show_bdry = False
show_best_c = False
# Either all features in the feature sets are normalized or only the amount value is normalized
def normalize_feature(data, amount_only = False):
    if amount_only:
        data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
    else:
        for feature in data.columns[:-1]:
            data[feature] = StandardScaler().fit_transform(data[feature].values.reshape(-1,1))
    return data

#Data are splitted to train and test sets: in each set
#Let total_normal and total_fraud be the total numbers of normal and fraud activities in the data sets.
#The train set contains (1-test_size)*total_normal normal activities and  (1-test_size)*total_fraud fraud activities.
#The test set contains test_size*total_normal normal activities and  test_size*total_fraud fraud activities.
def split_train_test(fraud_indices, normal_indices, test_size = 0.3):
    number_records_fraud = len(fraud_indices)
    number_records_normal = len(normal_indices)
    test_fraud_end = int(number_records_fraud * test_size)
    test_normal_end = int(number_records_normal  * test_size)

    test_fraud_indices = fraud_indices[0:test_fraud_end]
    train_fraud_indices = fraud_indices[test_fraud_end:]

    test_normal_indices = normal_indices[0:test_normal_end]
    train_normal_indices = normal_indices[test_normal_end:]

    return train_normal_indices, train_fraud_indices, test_normal_indices, test_fraud_indices

#Train set is downsampled so that in the train set the number of normal activities = ratio*the number of normal activities
def getTrainingSample(train_fraud_indices, train_normal_indices, data, train_normal_pos, ratio):
    train_number_records_fraud = int(ratio*len(train_fraud_indices))
    train_number_records_normal = len(train_normal_indices)
    if train_normal_pos + train_number_records_fraud <= train_number_records_normal:
        small_train_normal_indices = train_normal_indices[train_normal_pos: train_normal_pos + train_number_records_fraud]
        train_normal_pos = train_normal_pos + train_number_records_fraud
    else:
        small_train_normal_indices = np.concatenate([train_normal_indices[train_normal_pos: train_number_records_normal],train_normal_indices[0: train_normal_pos + train_number_records_fraud- train_number_records_normal]])
        train_normal_pos = train_normal_pos + train_number_records_fraud - train_number_records_normal

    # Appending the 2 fraud_indices
    under_train_sample_indices = np.concatenate([train_fraud_indices, small_train_normal_indices])
    np.random.shuffle(under_train_sample_indices)
    # Under sample dataset
    under_train_sample_data = data.iloc[under_train_sample_indices,:]

    X_train_undersample = under_train_sample_data.ix[:, under_train_sample_data.columns != 'Class']
    y_train_undersample = under_train_sample_data.ix[:, under_train_sample_data.columns == 'Class']

    return X_train_undersample, y_train_undersample, train_normal_pos

# It is a module of k-nearest nbd. which is called in cross-validation.
def knn_module(X, y, indices, c_param, bdry = None):
    knn = KNeighborsClassifier(n_neighbors = c_param)
    knn.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = knn.predict(X.iloc[indices[1],:].values)
    return y_pred_undersample

# It is a module of support vector machine using Gaussian radix basis function kernel which is called in cross-validation.
def svm_rbf_module(X, y, indices, c_param, bdry = 0.5):
    svm_rbf = SVC(C=c_param, probability = True)
    svm_rbf.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = svm_rbf.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# It is a module of support vector machine using polynomial kernel which is called in cross-validation.
def svm_poly_module(X, y, indices, c_param, bdry = 0.5):
    svm_poly = SVC(C= c_param[0], kernel = 'poly', degree = c_param[1], probability = True)
    svm_poly.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = svm_poly.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# It is a module of logistic regression which is called in cross-validation.
def lr_module(X, y, indices, c_param, bdry = 0.5):
    lr = LogisticRegression(C = c_param, penalty = 'l1')
    lr.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample= lr.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# It is a module of random forest which is called in cross-validation
def rf_module(X, y, indices, c_param, bdry = 0.5):
    rf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, criterion = 'entropy', max_features = 'auto', max_depth = None, min_samples_split  = c_param, random_state=0)
    rf.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = rf.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample

# This is a method to compute recall score and auc.
# y_t stands for actual label while y_p stands for predicted labels.
def compute_recall_and_auc(y_t, y_p):
    # Compute confusion cnf_matrix
    cnf_matrix = confusion_matrix(y_t,y_p)
    np.set_printoptions(precision = 2)
    recall_score = cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])

    # ROC CURVE
    fpr, tpr, thresholds = roc_curve(y_t, y_p)
    roc_auc = auc(fpr, tpr)
    return recall_score, roc_auc

# This is cross_validation method to tune some hyperparameters in a learning models
# such that recall scores in the learning model is optimized.
# The parameter models_dict contains a module of each learning model.
def cross_validation_recall(x_train_data,y_train_data, c_param_range, models_dict, model_name):
    fold = KFold(len(y_train_data),5,shuffle=False)

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    recall_mean = []
    for c_param in c_param_range:
        recall_accs = []
        for iteration, indices in enumerate(fold, start = 1):

            y_pred_undersample = models_dict[model_name](x_train_data, y_train_data, indices, c_param)

            recall_acc, _ = compute_recall_and_auc(y_train_data.iloc[indices[1],:].values, y_pred_undersample)
            recall_accs.append(recall_acc)

        recall_mean.append(np.mean(recall_accs))

    results_table['Mean recall score'] = recall_mean
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    return best_c

# This is cross_validation method to tune the decision boundary threshold in a learning models
# such that a function of auc and recall score is maximized.
# The parameter bdry_dict contains a function of auc and recall score in each learning model.
def decision_boundary(x_train_data,y_train_data, fold,  best_c, bdry_dict, models_dict, model_name):
    bdry_ranges = [0.3, 0.35, 0.4, 0.45, 0.5]
    results_table = pd.DataFrame(index = range(len(bdry_ranges),2), columns = ['C_parameter','Mean recall score * auc'])
    results_table['Bdry_params'] = bdry_ranges

    recall_mean = []
    for bdry in bdry_ranges:
        recall_accs_aucs = []
        for iteration, indices in enumerate(fold, start = 1):
            y_pred_undersample = models_dict[model_name](x_train_data, y_train_data, indices, best_c, bdry)
            recall_acc, roc_auc = compute_recall_and_auc(y_train_data.iloc[indices[1],:].values, y_pred_undersample)
            recall_accs_aucs.append(bdry_dict[model_name](recall_acc, roc_auc))
        recall_mean.append(np.mean(recall_accs_aucs))

    results_table['Mean recall score * auc'] = recall_mean
    best_bdry = results_table.loc[results_table['Mean recall score * auc'].idxmax()]['Bdry_params']

    return best_bdry

# Model method is our model to classify the fraud and normal activities.
def model(X, y, train, bdry_dict = None, best_c = None, best_bdry = None, models = None, mode = None):
    # This is the training part of our model
    if train:
        # Using cross-validaton, some hyperparameters in the following learning models are determined
        # by optimizing the recall score in the model.
        models_dict = {'knn' : knn_module, 'svm_rbf': svm_rbf_module, 'svm_poly': svm_poly_module,
                        'lr': lr_module, 'rf': rf_module}

        # The number of k in the k-nearest nbd. model is determined
        c_param_range_knn = [3,5,7,9]
        best_c_knn = cross_validation_recall(X, y, c_param_range_knn, models_dict, 'knn')

        # The penalty parameter in the support vector machine using Gaussian radix basis function kernel is determined.
        c_param_range_svm_rbf = [0.01, 0.1, 1, 10, 100]
        best_c_svm_rbf = cross_validation_recall(X, y, c_param_range_svm_rbf, models_dict, 'svm_rbf')
        c_param_range_svm_poly = [[0.01, 2], [0.01, 3], [0.01, 4], [0.01, 5], [0.01, 6], [0.01, 7], [0.01, 8], [0.01, 9],
                                  [0.1, 2], [0.1, 3], [0.1, 4], [0.1, 5], [0.1, 6], [0.1, 7], [0.1, 8], [0.1, 9],
                                  [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                                  [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9],
                                  [100, 2], [100, 3], [100, 4], [100, 5], [100, 6], [100, 7], [100, 8], [100, 9]]

        # The penalty parameter and the degree of the polynomial kernel in the support vector machine
        # are determined.
        best_c_svm_poly = cross_validation_recall(X, y, c_param_range_svm_poly, models_dict, 'svm_poly')

        # The penalty parameter in the logistic regression model is determined.
        c_param_range_lr = [0.01,0.1,1,10,100]
        best_c_lr = cross_validation_recall(X, y, c_param_range_lr, models_dict, 'lr')
        # The min_samples_split in the random forest model is determined
        c_param_range_rf = [2, 5, 10, 15, 20]
        best_c_rf = cross_validation_recall(X, y, c_param_range_rf, models_dict, 'rf')
        best_c = [best_c_knn, best_c_svm_rbf, best_c_svm_poly, best_c_lr, best_c_rf, best_c]

        # Using cross-validaton, decision boundary thresholds in the following learning models are determined
        # by optimizing some function of auc and recall score in the model.
        # The function of auc and recall score in each model is stored in the bdry_dict.
        fold = KFold(len(y), 4 ,shuffle=True) # Re-shuffle in the cross-validation are required. Otherise,
                                              # the training in both the previous cross-validation and the current cross
                                              # validation learn the noise in the train set and the quality of decision
                                              # boundary values is damaged.
        # The decision boundary threshold of the support vector machine using Gaussian radix basis function kernel is
        # determined.
        best_bdry_svm_rbf= decision_boundary(X, y, fold, best_c_svm_rbf, bdry_dict, models_dict, 'svm_rbf')

        # The decision boundary threshold of the support vector machine using polynomial kernel is determined.
        best_bdry_svm_poly = decision_boundary(X, y, fold, best_c_svm_poly, bdry_dict, models_dict, 'svm_poly')

        # The decision boundary threshold of the logistic regression model is determined.
        best_bdry_lr = decision_boundary(X, y, fold, best_c_lr, bdry_dict, models_dict, 'lr')

        # The decision boundary threshold of the random forest model is determined.
        best_bdry_rf = decision_boundary(X, y, fold, best_c_lr, bdry_dict, models_dict, 'rf')
        best_bdry = [0.5, best_bdry_svm_rbf, best_bdry_svm_poly, best_bdry_lr, best_bdry_rf]

        # Each model are trained by using the hyperparamters found in the cross-validation
        knn = KNeighborsClassifier(n_neighbors = int(best_c_knn))
        knn.fit(X.values, y.values.ravel())

        svm_rbf = SVC(C=best_c_svm_rbf, probability = True)
        svm_rbf.fit(X.values, y.values.ravel())

        svm_poly = SVC(C=best_c_svm_poly[0], kernel = 'poly', degree = best_c_svm_poly[1], probability = True)
        svm_poly.fit(X.values, y.values.ravel())

        lr = LogisticRegression(C = best_c_lr, penalty ='l1', warm_start = False)
        lr.fit(X.values, y.values.ravel())

        rf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, criterion = 'entropy', max_features = 'auto', max_depth = None, min_samples_split  = int(best_c_rf), random_state=0)
        rf.fit(X.values, y.values.ravel())

        models = [knn, svm_rbf, svm_poly, lr, rf]

        return best_c, best_bdry, models
    else:
        # This part is to classify fraud and normal activities in a test set by our model.
        [knn, svm_rbf, svm_poly, lr, rf] = models
        [_, best_bdry_svm_rbf, best_bdry_svm_poly, best_bdry_lr, best_bdry_rf] = best_bdry

        # Class of activities are predicted by the k nearest nbd. model.
        y_pred_knn = knn.predict(X.values)
        # Class of activities are predicted by the support vector machine using Gaussian radix basis function kenerl
        # according to the decision boundary threshold.
        y_pred_svm_rbf = svm_rbf.predict_proba(X.values)[:,1] >= best_bdry_svm_rbf
        # Class of activities are predicted by the support vector machine using polynomial kenerl
        # according to the decision boundary threshold.
        y_pred_svm_poly = svm_poly.predict_proba(X.values)[:,1] >= best_bdry_svm_poly
        # Class of activities are predicted by the logistic regression according to the decision boundary threshold.
        y_pred_lr= lr.predict_proba(X.values)[:,1] >= best_bdry_lr
        # Class of activities are predicted by the random forest according to the decision boundary threshold.
        y_pred_rf = rf.predict_proba(X.values)[:,1] >= best_bdry_rf

        x_of_three_models = {'knn' : y_pred_knn, 'svm_rbf' : y_pred_svm_rbf, 'svm_poly' : y_pred_svm_poly, 'lr' : y_pred_lr, 'rf': y_pred_rf}
        X_5_data = pd.DataFrame(data = x_of_three_models)

        y_pred= np.sum(X_5_data, axis = 1)>=mode

        y_pred_lr_controls = []
        params = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        # Final class in each test case is determined by a voting system among all these five learning models.
        for param in params:
            y_pred_lr_controls.append(lr.predict_proba(X.values)[:,1] >= param)
        return y_pred, y_pred_lr_controls, params



def run(data, mode, ratio, iteration1, bdry_dict):
    recall_score_list =[]
    auc_list = []
    recall_score_lr_list =[]
    auc_lr_list = []
    best_c = None
    best_bdry = None
    for itr1 in range(iteration1):
        print("percentage: %.2f" %(itr1/iteration1*100))

        # Number of data points in the minority class
        fraud_indices = np.array(data[data.Class == 1].index)
        np.random.shuffle(fraud_indices)

        #Picking the indices of the normal count_classes
        normal_indices = np.array(data[data.Class == 0].index)
        np.random.shuffle(normal_indices)

        # train and test
        train_normal_indices, train_fraud_indices, test_normal_indices, test_fraud_indices = split_train_test(
                                                                                            fraud_indices, normal_indices)
        test_indices = np.concatenate([test_normal_indices,test_fraud_indices])

        test_data = data.iloc[test_indices,:]
        X_test = test_data.ix[:, test_data.columns != 'Class']
        y_test = test_data.ix[:, test_data.columns == 'Class'].values.ravel()

        # Under sample dataset
        X_train_undersample, y_train_undersample, train_normal_pos = getTrainingSample(
                                                                    train_fraud_indices, train_normal_indices, data, 0, ratio)

        # Train our model
        best_c, best_bdry, models = model(X_train_undersample, y_train_undersample, train = True,
                                          bdry_dict = bdry_dict, best_c = best_c, best_bdry = best_bdry)

        if show_best_c:
            print("Some hyperparamter values:")
            print("k-nearest nbd: %.2f, svm (rbf kernel): [%.2f, %.2f], svm (poly kernel): %.2f, logistic reg: %.2f, random forest: %.2f"
                  %(best_c[0], best_c[1], best_c[2][0], best_c[2][1], best_c[3], best_c[4]))

        if show_bdry:
            print("Decision Boundary thresholds:")
            print("k-nearest nbd: %.2f, svm (rbf kernel): %.2f, svm (poly kernel): %.2f, logistic reg: %.2f, random forest: %.2f"
                  %(best_bdry[0], best_bdry[1], best_bdry[2], best_bdry[3], best_bdry[4]))

        # Classify class by our models
        y_pred, y_pred_lr_controls, params = model(X_test, y_test, train = False, bdry_dict = None,
                                                   best_c = best_c, best_bdry = best_bdry, models = models, mode = mode)

        # Collect recall score and auc in our models and the models in a control set for performance comparison.
        recall_score, roc_auc = compute_recall_and_auc(y_test, y_pred)
        recall_score_list.append(recall_score)
        auc_list.append(roc_auc)

        control_recall_all_param = []
        control_roc_all_param = []
        for i in range(len(params)):
            recall_score_lr, roc_auc_lr = compute_recall_and_auc(y_test, y_pred_lr_controls[i]) # for control
            control_recall_all_param.append(recall_score_lr)
            control_roc_all_param.append(roc_auc_lr)

        recall_score_lr_list.append(control_recall_all_param)
        auc_lr_list.append(control_roc_all_param)

    # Compute the mean value of recall score and auc in our models and the models in a control set for performance comparison
    mean_recall_score = np.mean(recall_score_list)
    std_recall_score = np.std(recall_score_list)

    mean_auc= np.mean(auc_list)
    std_auc = np.std(auc_list)

    mean_recall_score_lr = np.mean(recall_score_lr_list, axis = 0)
    std_recall_score_lr = np.std(recall_score_lr_list, axis = 0)
    mean_auc_lr= np.mean(auc_lr_list, axis = 0)
    std_auc_lr = np.std(auc_lr_list, axis = 0)

    result = [mean_recall_score, std_recall_score, mean_auc, std_auc]
    control = [mean_recall_score_lr, std_recall_score_lr, mean_auc_lr, std_auc_lr]
    return result, control, params


#Some parameters setting:
#################################################################################
mode = 2
ratio = 1
iteration1 = 100
show_best_c = True
show_bdry = True
#function to optimize in the decision_boundary method
def lr_bdry_module(recall_acc, roc_auc):
    return 0.9*recall_acc+0.1*roc_auc
def svm_rbf_bdry_module(recall_acc, roc_auc):
    return recall_acc*roc_auc
def svm_poly_bdry_module(recall_acc, roc_auc):
    return recall_acc*roc_auc
def rf_bdry_module(recall_acc, roc_auc):
    return 0.5*recall_acc+0.5*roc_auc

bdry_dict = {'lr': lr_bdry_module,'svm_rbf': svm_rbf_bdry_module,
             'svm_poly': svm_poly_bdry_module, 'rf': rf_bdry_module}
################################################################################

# Implementation of our model according to these parameters setting.:
data = pd.read_csv("creditcard.csv")
data = data.drop(['Time'], axis = 1)
data = normalize_feature(data, amount_only = True)

result, control, params = run(data = data, mode = mode, ratio = ratio, iteration1 = iteration1, bdry_dict = bdry_dict)
print("Hyperparameter values:")
print("ratio: ", ratio, " and mode: ", mode)
print("Result of our model which a voting models among knn, svm_rbf, svm_poly, lr and rf:")
print("mean_recall is ", result[0], " and std is ", result[1])
print("mean_auc is ", result[2], " and std is ", result[3])
print()
print("Control experiments of logistic regression models with different threshold")
print("i.e., fraud is predicted if the probability value exceeds the threshold")
for i, param in enumerate(params):
    print("Threshold", param)
    print("mean_recall is ", control[0][i], " and std is ", control[1][i])
    print("mean_auc is ", control[2][i], " and std is ", control[3][i])
    print()
