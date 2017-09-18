"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Ver1 just follows ideas of the following link 
https://www.data-blogger.com/2017/06/15/fraud-detection-a-simple-machine-learning-approach/
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

#Read the CSV file
data = pd.read_csv('creditcard.csv')

#Show the contents
#print(data)
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
	
# Fit and predict
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
	
# And finally: show the results
print(classification_report(Y_test, Y_pred))

 
	 
