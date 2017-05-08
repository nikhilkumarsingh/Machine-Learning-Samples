from sklearn import datasets, linear_model, metrics
 
# load the digit dataset
digits = datasets.load_digits()
 
# defining feature matrix(X) and response vector(y)
X = digits.data
y = digits.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)
 
# create logistic regression object
reg = linear_model.LogisticRegression()
 
# train the model using the training sets
reg.fit(X_train, y_train)

# making predictions on the testing set
y_pred = reg.predict(X_test)
 
# comparing actual response values (y_test) with predicted response values (y_pred)
print("Logistic Regression model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)