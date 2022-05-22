import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

Base_path='D:\Data_sbk_fitting'
Data_path='data_found\\data'
Model_path='SVR_model_fitting\\model'
Result_path='SVR_model_fitting\\result'
All_data=[]

def SVR_model():
    # Fix a random seed  to initialize random no. generator
    np.random.seed(7)

    #data preprocess to adapt the model
    #preparing the data
    data_path=os.path.join(Base_path,Data_path,'wettability_data_sbk_processed.csv')

    # read file and standardize the data
    df=pd.read_csv(data_path)
    df.reset_index(drop=True)
    df=df.loc[:, ~df.columns.str.match('Unnamed')]
    scalar = StandardScaler()
    y = df["receding angle"]
    X = scalar.fit_transform(df.drop(columns=['receding angle']))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

    # Define the support vector regression models for rbf, quadratic, and linear kernels
    regressor_rbf = SVR(kernel = 'rbf', C=1000, gamma=0.5) #create regressor_rbf SVR class object with kernel ‘rbf’ because this follows Gaussian process
    regressor_rbf.fit(x_train, y_train) #fit scaled X and y to the object regressor_rbf
    regressor_quad = SVR(kernel = 'poly', C=1000, degree = 2)
    regressor_quad.fit(x_train, y_train)
    regressor_linear = SVR(kernel = 'linear', C=1000)
    regressor_linear.fit(x_train, y_train)

    # Predict receding angle
    y_pred_rbf= regressor_rbf.predict(x_test) #Predicted y values
    y_pred_rbf=y_pred_rbf.reshape(-1,1)
    y_pred_quad= regressor_quad.predict(x_test) #Predicted y values
    y_pred_quad=y_pred_quad.reshape(-1,1)
    y_pred_linear= regressor_linear.predict(x_test) #Predicted y values
    y_pred_linear=y_pred_linear.reshape(-1,1)

    All_data.append(y_test)
    All_data.append(y_pred_rbf)
    All_data.append(y_pred_linear)
    All_data.append(y_pred_quad)

    df0=pd.DataFrame(data=All_data[0])
    df0.reset_index(drop=True,inplace=True)
    df1=pd.DataFrame(data=All_data[1])
    df1.reset_index(drop=True,inplace=True)
    df2 = pd.DataFrame(data=All_data[2])
    df2.reset_index(drop=True,inplace=True)
    df3 = pd.DataFrame(data=All_data[3])
    df3.reset_index(drop=True,inplace=True)

    df=pd.concat([df0,df1,df2,df3],axis=1,ignore_index=True)
    df.to_csv(os.path.join(Base_path,Result_path,'test_compare_predict.csv'))

def proformance_plot(y_test,y_pre,number):
    # linear regression
    linear = LinearRegression()
    y_test = np.array(y_test).reshape(14, 1)
    y_pre = np.array(y_pre).reshape(14, 1)
    linear.fit(y_test, y_pre)
    plt.scatter(y_test, y_pre, c='blue')
    pred_y = linear.predict(y_test)
    plt.plot(y_test, pred_y, c='red')
    plt.xlabel('Expremental receding angle', fontsize=20)
    plt.ylabel('Predicted receding angle', fontsize=20)
    plt.savefig(os.path.join(Base_path, Result_path, 'Liner_regression_%d.jpg'%number))
    plt.show()

    data = {'y_test': y_test, 'y_pre': y_pre}
    score = r2_score(y_test, pred_y)
    print('test_R2_score:' + str(score))

if __name__=="__main__":
    SVR_model()
    proformance_plot(All_data[0],All_data[1],1)
    proformance_plot(All_data[0], All_data[2],2)
    proformance_plot(All_data[0], All_data[3],3)