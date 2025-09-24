import pandas
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

file=os.path.join(os.getcwd(),"BMW_Car_Sales_Classification.csv")
dataset=pandas.read_csv(file)

encoder=OrdinalEncoder()
columns=['Model','Year','Region','Color','Fuel_Type','Transmission']
dataset[columns]=encoder.fit_transform(dataset[columns])

encoder_predict=OrdinalEncoder()
dataset['Sales_Classification']=encoder_predict.fit_transform(dataset[['Sales_Classification']])

x_train=dataset.drop('Sales_Classification',axis=1)
y_train=dataset['Sales_Classification']

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=42)

model=xgb.XGBClassifier(
    n_estimators=50,
    num_parallel_tree=2,
    max_depth=5
)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)

print(f"accuracy:{accuracy}")

model.save_model('model.json')
joblib.dump(encoder,'encoder.pkl')
joblib.dump(encoder_predict,'encoder_predict.pkl')
print("Model saved")