# importing standard libraries
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from Model_Save_Load import load_model
Score=[0.0,0.0,0.0,0.0]
Score_ALL=[]
BASEPATH ='D:\RANDOM_FOREST_TREE_MODEL\data_model_build'
params=[]
random_seed = 1
random_seed_range=[2,5,10,13,21,32,42,53,62]

def model_test():
    # datapath=os.path.join(BASEPATH,'data','metal_ceramic_data_all_with_A_T.csv')
    # datapath = os.path.join(BASEPATH, 'data', 'metal_ceramic_data_all_with_A_T_reduced_MIT.csv')
    datapath = os.path.join(BASEPATH, 'data', 'metal_ceramic_data_all_with_A_T_reduced.csv')
    df = pd.read_csv(datapath)  # read file
    x = df.drop(columns=['Wetting angle', 'Metal', 'Substrate'])  # x as predictor,y as result
    # x = df(columns=['Me_MagpieData mean NdValence', 'Me_MagpieData mean CovalentRadius', 'Me_MagpieData mean GSmagmom', 'Me_MagpieData mean Electronegativity', 'Testing temperature (K)', 'Me_MagpieData mean GSbandgap', 'Ce_MagpieData mean NUnfilled', 'Me_MagpieData mean SpaceGroupNumber', 'Ce_MagpieData mean MeltingT'])
    y = df["Wetting angle"]

    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(x)

    # Splitting the dataset for train and test
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(x_scaled, y, test_size=0.20, random_state=random_seed)
    x_validation,x_test,y_validation,y_test  = train_test_split(x_validation_test,y_validation_test,test_size=0.5,random_state=random_seed)

    random_forest_model_test_base = RandomForestRegressor()
    random_forest_model_test_base.fit(x_train, y_train)
    # print("Random forest Score of train data= %.3f" % random_forest_model_test_base.score(x_train,y_train))
    # print("Random forest base Score test data = %.3f" % random_forest_model_test_base.score(x_test,y_test))
    Score[0]=random_forest_model_test_base.score(x_train,y_train)
    Score[1]=random_forest_model_test_base.score(x_validation_test,y_validation_test)

    random_forest_model_test_random = load_model('RTF_0.961_0.852_0.991_0.769_42',42)
    random_forest_model_test_random.fit(x_validation, y_validation)
    # print("Random forest Score of validation data= %.3f"%random_forest_model_test_random.score(x_validation,y_validation))
    Score[2]=random_forest_model_test_random.score(x_validation,y_validation)
    best_hp_now = random_forest_model_test_random.best_params_
    print(best_hp_now)

    #test data
    score=random_forest_model_test_random.score(x_test,y_test)
    Score[3]=random_forest_model_test_random.score(x_test,y_test)
    # print("Random forest Score of test data = %.3f"%score)

    #save best parmas and model
    params.append(best_hp_now)
    SCORE= {"Train_90_base%": Score[0], "test_%10_base": Score[1], "Train_%90_best": Score[2], "Test_%10_best": Score[3]}
    Score_ALL.append(SCORE)

if __name__=="__main__":
    for j in random_seed_range:
        random_seed=j
        for i in range(10):
            model_test()

        # save params to csv file
        paramters=pd.DataFrame(params)
        # params_path=os.path.join(BASEPATH,'PARAMS','42_features_used_All','params_%d.csv'%random_seed)
        # params_path=os.path.join(BASEPATH,'PARAMS','8_features_used_MIT','params_%d.csv'%random_seed)
        params_path=os.path.join(BASEPATH,'PARAMS','one_model_test','params_9.csv')
        paramters.to_csv(params_path,mode='a')

        #save results to csv file
        results=pd.DataFrame(Score_ALL)
        # results_path=os.path.join(BASEPATH,'RESULTS','42_features_used_All','results_%d.csv'%random_seed)
        # results_path=os.path.join(BASEPATH,'RESULTS','8_features_used_MIT','results_%d.csv'%random_seed)
        results_path=os.path.join(BASEPATH,'RESULTS','one_model_test','results_9.csv')
        results.to_csv(results_path,mode='a')