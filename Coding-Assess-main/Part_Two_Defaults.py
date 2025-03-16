import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#Load Data & clean data
def_df = pd.read_csv( 'C:/Dev/Code-Assessment/Coding-Assess-main/data/Part 2. loan_data_final.csv' )
def_df.drop( labels=[ 'Unnamed: 0' ], axis=1, inplace=True )

# One Hot Encode catagorical features 
ohe_df = pd.get_dummies( def_df, 
                        columns = [ 'person_gender', 'person_education', 'person_home_ownership', 
                                    'loan_intent', 'previous_loan_defaults_on_file', 'loan_type' 
                        ] 
)

#Split features from target
y = ohe_df[ 'loan_status' ]
X = ohe_df.drop( labels=[ 'loan_status' ], axis=1 )

#Normalize numeric features
columns = list( X.columns )
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Train logistic regression model
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 )
model = LogisticRegression()
model.fit(X_train, y_train)

#Check results
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')

#Try using a SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy}')