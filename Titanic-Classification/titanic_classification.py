import pandas as pd 
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('Titanic/train.csv')
test_data = pd




encoder = OrdinalEncoder()
features_to_transform = ['Sex','Embarked']
df[features_to_transform] = encoder.fit_transform(df[features_to_transform])



y = np.array(df['Survived'])
X = np.array(df.drop((['PassengerId','Survived','Name','Ticket','Cabin']),axis=1))



imputer = SimpleImputer(strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


model = RandomForestClassifier(n_estimators=700,max_depth=8,max_features=0.6,min_samples_split=3,random_state=42)
model.fit(x_train,y_train)
preds = model.predict(x_test)



print('Accuracy Score:',accuracy_score(y_test,preds))
print('Precision Score:',precision_score(y_test,preds))
print('Recall Score:',recall_score(y_test,preds))
print('F1 Score:',f1_score(y_test,preds))
print('ROC AUC Score:',roc_auc_score(y_test,preds))
print('Classification Report:\n',classification_report(y_test,preds))

