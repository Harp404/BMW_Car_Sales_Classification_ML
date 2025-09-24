import xgboost as xgb
import pandas as pd
import questionary
import joblib
import os

model_file=os.path.join(os.getcwd(),"model.json")
dataset=os.path.join(os.getcwd(),"BMW_Car_Sales_Classification.csv")

model=xgb.XGBClassifier()
model.load_model(model_file)
encoder=joblib.load('encoder.pkl')
encoder_predict=joblib.load('encoder_predict.pkl')
df = pd.read_csv(dataset)
df=df.drop('Sales_Classification',axis=1)

columns_encode=['Model','Year','Region','Color','Fuel_Type','Transmission']
columns_all=df.columns

user_input={}

for col in columns_all:
    if col in ['Mileage_KM','Price_USD','Sales_Volume']:
        user_input[col]=0
        continue
    choices=sorted([str(x) for x in df[col].unique()])
    answer=questionary.select(
        f"Select {col}:",
        choices=choices).ask()
    user_input[col]=answer

user_input=pd.DataFrame([user_input])
convert=['Year','Price_USD','Sales_Volume']
user_input[convert]=user_input[convert].astype(int)
user_input['Mileage_KM']=int(input("Enter Mileage: "))
user_input['Price_USD']=int(input("Enter Price_USD: "))
user_input['Sales_Volume']=int(input("Enter Sales_Volume: "))
user_input['Engine_Size_L']=user_input['Engine_Size_L'].astype(float)
user_input[columns_encode]=encoder.transform(user_input[columns_encode])
print(user_input)

output_model=model.predict(user_input)
output=encoder_predict.inverse_transform([output_model])
print(f"The Sales Classification is: {output[0][0]}")