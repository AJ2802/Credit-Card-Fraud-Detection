"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Ver2 is to classify fraud transaction by using random forest instead of logistical regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#Read the CSV file
data = pd.read_csv('creditcard.csv')

#Show the contents
#print(data)
#View top 5 column
print(data.head())
#View first 4 features
#print(data.columns[:4])
#print(data.describe())

# Only use the 'Amount' and 'V1',..., 'V28' features
features = ['Amount'] + ['V%d' % number for number in range(1,29)]

# The target variable which we would like to predict, is the 'Class' variable
target = 'Class'

# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = data[features]
Y = data[target]

def normalize(X):
	"""
	Make the distribution of the values of each variable similar by substracting the mean and dividing by the standard deviation.
	"""
	for feature in X.columns:
		X[feature] = X[feature] -X[feature].mean()
		X[feature] = X[feature] / X[feature].std()
		
	return X
	
# Define the model
model = LogisticRegression()

# Define the splitter for splitting the data in a train set and a test set
splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 0)

# Loop through the splits (only one)
for train_indices, test_indices in splitter.split(X,Y):
	# Select the train and test data
	X_train, Y_train = X.iloc[train_indices], Y.iloc[train_indices]
	X_test, Y_test = X.iloc[test_indices], Y.iloc[test_indices]
	
# Normalize the data
X_train = normalize(X_train)
X_test = normalize(X_test)

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators = 10, criterion='gini', max_features = "sqrt",	bootstrap = True, n_jobs = 1, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training Y_train
clf.fit(X_train,Y_train)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
# View the predicted probabilities of the first 10 observations
preds = np.argmax(clf.predict_proba(X_test), axis = 1)

# Create confusion matrix
confusionMatrix = pd.crosstab(Y_test, preds, rownames =['Actual Normal/Fraud'], colnames=['Predicted Normal/Fraud'])
print("confusionMatrix")
print(confusionMatrix.div(confusionMatrix.sum(axis=0), axis=1))
confM = np.matrix(confusionMatrix)

print("Precision of positive predicted value: True Positive/Predicted Positive : %.4f" %(confM[0,0]/np.sum(confM[:,0])))
print("Precision of negative predicted value: True Negative/Predicted Negative : %.4f" %(confM[1,1]/np.sum(confM[:,1])))
print("Total Accuracy: (True Negative + True Positive)/(Predicted Positive + Predicted Negative) : %.4f" %(np.sum(np.diag(confM))/np.sum(confM)))

"""
#View a list of features and their importance scores
print()
print(list(zip(X_train, clf.feature_importances_)))
"""


	 
