# importing standard libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from Model_Save_Load import save_model
Score=[0.0,0.0,0.0,0.0]
Score_ALL=[]
BASEPATH ='D:\RANDOM_FOREST_TREE_MODEL\data_model_build'
params=[]
random_seed = 1
random_seed_range=[2,5,10,13,21,32,42,53,62]

def random_tree_models_build():
    random_forest_seed = np.random.randint(low=1, high=230)
    #
    # Search optimal hyperparameter
    n_estimators_range = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features_range = ['auto', 'sqrt']
    max_depth_range = [int(x) for x in np.linspace(1, 200, num=40)]
    min_samples_split_range = [2, 5, 10]
    min_samples_leaf_range = [1, 2, 4, 8,16]
    bootstrap_range = [True, False]

    random_forest_hp_range = {'n_estimators': n_estimators_range,
                              'max_features': max_features_range,
                              'max_depth': max_depth_range,
                              'min_samples_split': min_samples_split_range,
                              'min_samples_leaf': min_samples_leaf_range,
                               'bootstrap':bootstrap_range
                              }

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
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.10, random_state=random_seed)

    random_forest_model_test_base = RandomForestRegressor()
    random_forest_model_test_base.fit(x_train, y_train)
    print("Random forest Score of train data= %.3f" % random_forest_model_test_base.score(x_train,y_train))
    print("Random forest base Score of test data = %.3f" % random_forest_model_test_base.score(x_test,y_test))
    Score[0]=random_forest_model_test_base.score(x_train,y_train)
    Score[1]=random_forest_model_test_base.score(x_test,y_test)

    random_forest_model_test_random = RandomizedSearchCV(estimator=random_forest_model_test_base,
                                                         param_distributions=random_forest_hp_range,
                                                         scoring='r2',
                                                         n_iter=200,
                                                         n_jobs=-1,
                                                         cv=3,
                                                         verbose=1,
                                                         random_state=random_forest_seed
                                                         )
    random_forest_model_test_random.fit(x_train, y_train)
    print("Random forest Score of train data after choosing best params= %.3f"%random_forest_model_test_random.score(x_train,y_train))
    Score[2]=random_forest_model_test_random.score(x_train,y_train)
    best_hp_now = random_forest_model_test_random.best_params_
    print(best_hp_now)

    #test data
    score=random_forest_model_test_random.score(x_test,y_test)
    Score[3]=random_forest_model_test_random.score(x_test,y_test)
    print("Random forest Score of test data after choosing best params= %.3f"%score)

    #save best parmas and model
    params.append(best_hp_now)
    save_model(random_forest_model_test_random,'RTF_%.3f_%.3f_%.3f_%.3f'%(Score[0],Score[1],Score[2],Score[3]),random_seed)
    SCORE= {"Train_90%_base": Score[0], "test_%10_base": Score[1], "Train_%90_after_choosing": Score[2], "Test_%10_after_choosing": Score[3]}
    Score_ALL.append(SCORE)

if __name__=="__main__":
    for j in random_seed_range:
        random_seed=j
        for i in range(10):
            random_tree_models_build()

        # save params to csv file
        paramters=pd.DataFrame(params)
        # params_path=os.path.join(BASEPATH,'PARAMS','42_features_used_All','params_%d.csv'%random_seed)
        # params_path=os.path.join(BASEPATH,'PARAMS','8_features_used_MIT','params_%d.csv'%random_seed)
        params_path=os.path.join(BASEPATH,'PARAMS','9_features_used_Own','params_%d.csv'%random_seed)
        paramters.to_csv(params_path)

        #save results to csv file
        results=pd.DataFrame(Score_ALL)
        # results_path=os.path.join(BASEPATH,'RESULTS','42_features_used_All','results_%d.csv'%random_seed)
        # results_path=os.path.join(BASEPATH,'RESULTS','8_features_used_MIT','results_%d.csv'%random_seed)
        results_path=os.path.join(BASEPATH,'RESULTS','9_features_used_Own','results_%d.csv'%random_seed)
        results.to_csv(results_path)