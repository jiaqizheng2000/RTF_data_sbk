import os
import joblib

BASEPATH ='D:\RANDOM_FOREST_TREE_MODEL\data_model_build'

def save_model(model,model_name,random_seed):
    MODELPATH=os.path.join(BASEPATH,'MODELS','42_features_used_All',str(random_seed))
    # MODELPATH = os.path.join(BASEPATH, 'MODELS', '8_features_used_MIT', str(random_seed))
    # MODELPATH = os.path.join(BASEPATH, 'MODELS', '9_features_used_Own', str(random_seed))
    if not os.path.exists(MODELPATH):
        os.mkdir(MODELPATH)

    savepath=os.path.join(MODELPATH,model_name +'.pkl')
    joblib.dump(model,savepath)

def load_model(model_name,num):
    # MODELPATH=os.path.join(BASEPATH,'MODELS','42_features_used_All')
    MODELPATH=os.path.join(BASEPATH,'MODELS','8_features_used_MIT')
    # MODELPATH=os.path.join(BASEPATH,'MODELS','9_features_used_Own')
    loadpath=os.path.join(MODELPATH,str(num),model_name +'.pkl')
    model=joblib.load(loadpath)
    return model