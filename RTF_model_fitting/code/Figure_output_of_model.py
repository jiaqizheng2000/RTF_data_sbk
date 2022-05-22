# importing standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_graphviz
import os

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)

numbers_estimators = 300
max_depth_for_RAN = 30

Base_path='D:\Data_sbk_fitting'
Data_path='data_found\\data'
result_path='RTF_model_fitting\\result'

if __name__=="__main__":
    data_path=os.path.join(Base_path,Data_path,'wettability_data_sbk_processed.csv')
    df=pd.read_csv(data_path) #read file
    df.reset_index(drop=True)
    df=df.loc[:, ~df.columns.str.match('Unnamed')]

    x = df.drop(columns=['receding angle'])#x as predictor,y as result
    feature_list = list(x.columns)
    y = df["receding angle"]

    plt.figure(figsize=(20, 30), facecolor='white')
    plotnumber = 1

    for column in x:
        if plotnumber <= 16:
            ax = plt.subplot(4, 4, plotnumber)
            plt.scatter(x[column], y)
            plt.xlabel(column, fontsize=20)
            plt.ylabel('Wetting angle', fontsize=20)
        plotnumber += 1
    plt.tight_layout()
    # plt.savefig(os.path.join(Base_path,result_path,'scattter_pots.jpg'))
    # plt.show()

    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(x)

    vif = pd.DataFrame() # variance_inflation_factor to measure how much the variance of
                        # an estimated regression cofficient is increased because of collinerarity
    vif["VIF"] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
    vif["Features"] = x.columns

    corrl = df.corr() # plot heat map
    cor=pd.DataFrame(data=corrl)
    cor.to_csv(os.path.join(Base_path,result_path,'correlation.csv'))
    plt.figure(figsize=(20, 20))
    sns.heatmap(corrl, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 12}, cmap='twilight_shifted_r')
    # plt.savefig(os.path.join(Base_path,result_path,'heat_map.jpg'))
    # plt.show()

    #Splitting the dataset for train and test
    # x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.10, random_state=470)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.20, random_state=10)
    train_X_column_name = list(x.columns)

    #Importing the models for training the dataset
    dtr = DecisionTreeRegressor()
    #ran = RandomForestRegressor(n_estimators=90)
    ran=RandomForestRegressor(n_estimators=numbers_estimators, max_depth = max_depth_for_RAN, random_state=42)
    lin = LinearRegression()

    models = {"Decision tree": dtr,
              "Random forest": ran,
              "Linear Regression": lin}
    scores = {}

    for key, value in models.items():
        model = value
        model.fit(x_train, y_train)
        # noinspection PyUnresolvedReferences
        scores[key] = model.score(x_test, y_test)

    scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
    scores_frame.sort_values(by=["Accuracy Score"], axis=0, ascending=False, inplace=True)
    print(scores_frame)

    #linear regression
    y_pre = ran.predict(x_test)
    linear = LinearRegression()
    y_test = np.array(y_test).reshape(14,1)
    y_pre  = np.array(y_pre).reshape(14,1)
    linear.fit(y_test,y_pre)
    plt.scatter(y_test,y_pre,c='blue')
    pred_y = linear.predict(y_test)
    plt.plot(y_test,pred_y,c = 'red')
    plt.xlabel('Expremental receding angle', fontsize=20)
    plt.ylabel('Predicted receding angle', fontsize=20)
    # plt.savefig(os.path.join(Base_path,result_path,'Liner_regression.jpg'))
    # plt.show()

    # get a tree in model
    tree = ran.estimators_[5]
    # output as dot 文件
    export_graphviz(tree, out_file=os.path.join(Base_path, result_path, '../result/tree.dot'), feature_names=feature_list, rounded=True, precision=1)

    #Calculate the importance of variables
    plt.figure(3)
    plt.figure(dpi=300,figsize=(9,9))
    plt.clf()

    importance_name_over0=[]
    importance_value_over0=[]
    indicites = np.argsort(tree.feature_importances_)[::-1]
    for i in range(len(train_X_column_name)):
        importance_value_over0.append(tree.feature_importances_[indicites[i]])
        importance_name_over0.append(train_X_column_name[indicites[i]])

    importance_name_over0_2=[]
    importance_value_over0_2=[]
    for i in range(len(importance_value_over0)):
        if importance_value_over0[i]>0.01:
            importance_value_over0_2.append(importance_value_over0[i])
            importance_name_over0_2.append(importance_name_over0[i])
    print(importance_value_over0_2)
    print(importance_name_over0_2)

    plt.bar(importance_name_over0_2, importance_value_over0_2, orientation='vertical')
    plt.xticks(importance_name_over0_2, importance_name_over0_2, rotation='vertical')
    plt.xlabel('Variable')
    plt.ylabel('Importance')
    plt.title('Variable Importances')
    plt.tight_layout()
    # plt.savefig(os.path.join(Base_path,result_path,'Importance.jpg'))
    # plt.show()



