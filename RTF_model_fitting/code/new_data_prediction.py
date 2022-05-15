# importing standard libraries
import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from Model_Save_Load import load_model

def new_data_prediction(input_data):
    RTF_model=load_model(model_name=None,num=None)

    scalar = StandardScaler()
    data_scaled = scalar.fit_transform(input_data)
    y_pre = RTF_model.predict(data_scaled)
    global prediction
    prediction=pandas.DataFrame(columns=['predicted_angle'],data=y_pre)

if __name__=="__main__":
    prediction=pd.DataFrame()
    data=pd.read_csv("metal_ceramic_data_all_with_A_T.csv")
    data=data[900:]
    data.reset_index(drop=True,inplace=True)
    x_data=data.drop(columns=['Wetting angle', 'Metal', 'Substrate'])
    new_data_prediction(x_data)
    data=pd.concat([data,prediction],axis=1)
    data.to_csv("prediction_result.csv")
