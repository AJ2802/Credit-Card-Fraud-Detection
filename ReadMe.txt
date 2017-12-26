Before running the script, please download the and unzip the file creditcard.csv on https://www.kaggle.com/dalpozz/creditcardfraud.

Python version to run my script: 3.6.4
Library required: pandas matplotlib, sklearn and scipy
Implement: my_model.py

The task in this project is to classify the fraud activity and the normal activity as good as possible. The main challenge is the dataset is highly imbalanced. There are 492 frauds out of 284,807 transactions. It is difficult to lower the false negative rate (misclassify fraud activity as normal) while the false positive rate (misclassify normal activity as fraud) is still kept reasonably high. In the analysis of prediction on highly imbalanced dataset, recall score and auc are some good indicators. The indicator recall score is defined as true positive/(true positive + false negative). The indicator auc under ROC curve is the area under the ROC curve [Some people use area of another curve as auc]. Usually, high recall score and high auc under ROC can conclude as an accurate classification. My goal in this project is to build up a model with high recall score and acc.

My model is inspired from the blog "In depth skewed data classif. (93% recall acc now)" written by joparga3In on https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now . In the blog, the train set is downsampled such that number of normal activities is the same as the number of fraud activities. This downsample process actually helps a lot. Then, by using cross-validation technique, the penalty parameter of logistical regression is chosen such that the recall score is optimized. Finely tuning the penalty parameter in this way is another boost to obtain high recall score. The well-trained logistical regression can classify the fraud and normal activity very well. I ran his algorithm for 100 times and get this result in different decision boundary thresholds. Results are follows:

Control experiments of logistic regression models with different threshold
i.e., fraud is predicted if the predicted probability value by the logistic regression model exceeds the threshold
Threshold 0.2
mean of recall score is  1.00  and standard deviation of recall score is  0.001
mean of auc is  0.50 and standard deviation of auc is  0.001

Threshold 0.25
mean of recall score is  1.00  and standard deviation of recall score is  0.003
mean of auc is  0.50  and standard deviation of auc is  0.001

Threshold 0.3
mean of recall score is 1.00  and standard deviation of recall score is 0.003
mean of auc is 0.51  and standard deviation of auc is 0.005

Threshold 0.35
mean of recall score is 1.00 and standard deviation of recall score is  0.006
mean of auc is 0.54  and standard deviation of auc is 0.011

Threshold 0.4
mean of recall score is 0.98  and standard deviation of recall score is  0.012
mean of auc is  0.68 and standard deviation of auc is  0.0156

Threshold 0.45
mean of recall score is 0.96 and standard deviation of recall score is 0.018
mean of auc is 0.84 and standard deviation of auc is 0.0136

Threshold 0.5
mean of recall score is 0.93  and standard deviation of recall score is  0.018
mean of auc is 0.91  and standard deviation of auc is 0.010

From the above output, a similar result as shown on the author joparga3In's blog can be regenerated.

I have added three more ideas on top of joparga3In's model. (1) beside logistic regression model, four more models-- k-nearest nbd, support vector machine using Gaussian radix function kernel, support vector machine using polynomial kernel and random forest-- are used for classification. (2) Using cross-validation, decision boundary threshold in each model except k-nearest nbd are determined by optimizing an objective function of recall score and auc in each model. Each model has it own objective function of recall score and auc. (3) The final classification of activity is based on the vote among all these five models-- k-nearest nbd, support vector machine using Gaussian radix function kernel, support vector machine using polynomial kernel logistic regression and random forest. The pseudo code of our model is as follows:
(1) Data are splitted to train and test sets.
(2) Training set is downsampled so that in the training set the number of normal activities = ratio*the number of fraud activities [ratio = 1 in our case].
(3) Using cross-validaton, some hyperparameters in the following learning models are determined by optimizing the recall score in the model.
	i.	The number of k in the k-nearest nbd. model is determined.
	ii.	The penalty parameter in the support vector machine using Gaussian radix basis function kernel is determined.
	iii.	The penalty parameter and the degree of the polynomial kernel in the support vector machine using polynomial kernel are determined.
	iv.	The penalty parameter in the logistic regression model is determined.
	v.	The min_samples_split in the random forest model is determined
(4) The cross-validation are re-shuffled. 
(5) Using cross-validaton, decision boundary thresholds in the following learning models are determined by optimizing some objective function of auc and recall score in the model.
	i. 	The decision boundary threshold of the support vector machine using Gaussian radix basis function kernel is determined by optimizing the square of the geometric 		mean function auc*(recall score).
	ii.	The decision boundary threshold of the support vector machine using polynomial kernel is determined by optimizing the square of the geometric mean function 			auc*(recall score).
	iii.	The decision boundary threshold of the logistic regression model is determined by optimizing the weighted arithmetic mean function 0.9*(recall score) + 0.1*auc.
	iv.	The decision boundary threshold of the random forest is determined by optimizing the arithmetic mean function 0.5*(recall score) + 0.5*auc.
(6) All five models-- k-nearest nbd, support vector machine using Gaussian radix function kernel, support vector machine using polynomial kernel logistic regression and random forest-- are trained with finely tuned parameters.
(7) Class of activities in a test set are predicted by these five models individually according to their own decision boundary thresholds. [Note that the decision boundary threshold of k-nearest nbd. is a standard value 0.5].
(8) In each test example, fraud activity is outputted by our model if the number of fraud predictions out of these five models is greater than or equal to mode. [mode = 2 in our case].

After implementing the pseudo code 100 times, the result is follows:
ratio:  1  and mode:  2
Result of our model which is a voting models among knn, svm_rbf, svm_poly, lr and rf:
mean of recall score is  0.95  and standard deviation of recall score is  0.021
mean of auc is  0.91  and standard deviation of auc is  0.017

Conclusion:
Our model can boost up the recall score a little bit (0.02 up compared with logistical regression with standard 0.5 decision boundary threshold and just 0.01 down compared with logistical regression with decision boundary threshold being 0.45 ) while the auc value can still maintained to be high (same as logistical regression with standard 0.5 decision boundary threshold and 0.07 up compared with logistical regression with decision boundary threshold being 0.45).

Potential Improvement:
1. Different value of mode is used
2. Some other models in the voting system are used.
3. Different objective function of recall score and auc to determine a decision boundary threshold is used. [It is a very sophisticated subject to find out a very good objective function to determine the decision boundary threshold to fulfill our goal.
4. Some learning models are used to learn the predicted probabilities outputted from all models in the voting system to give the final prediction of class of activities.
5. A deep neural network model is used for classification.

Reference:
1. Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
2. The blog "In depth skewed data classif. (93% recall acc now)", joparga3In, https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now
3. A voting model to keep recall score and auc high, AJ Tong, https://www.kaggle.com/punwai/a-voting-model-to-keep-recall-score-and-auc-high/ 



