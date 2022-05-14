import pandas as pd
from matminer.featurizers.conversions import StrToComposition #class to convert str to composition
from matminer.featurizers.composition import ElementProperty #Class to calculate elemental property attributes.
import os

magpie = ElementProperty.from_preset(
    preset_name="magpie")  # Return ElementProperty from a preset string, different kinds

Base_path='D:\RTF_data_sbk'
Data_path='data_found\\data'

def data_supplement(MENTAL,CERAMIC):
    metal = str(MENTAL)
    ceramic = str(CERAMIC)
    substrate = pd.DataFrame([ceramic]) #Two-dimensional, size-mutable, potentially heterogeneous tabular data(表格数据)
                                        # [] for data: Any = None
    metal_matminer = pd.DataFrame([metal], columns=['Metal'])
    metal_matminer = StrToComposition(target_col_id='Me_comp').featurize_dataframe(metal_matminer, 'Metal')#Compute features for all entries contained in input dataframe.
                                                                                                            #Perform the data conversion and set the target column dynamically
    data_Me = magpie.featurize_dataframe(metal_matminer, col_id="Me_comp", ignore_errors=True)  #mental properties
    metal_features = pd.DataFrame(data_Me.values.tolist()*len(substrate), columns = data_Me.columns)
    feature_Me = metal_features.filter(like = 'mean') # keep data that "mean" in label
    feature_Me = feature_Me.drop(columns=['MagpieData mean NfUnfilled']) #drop data of which columns
    feature_Me.columns = ['Me_'+ j for j in feature_Me.columns]

    matrix = pd.DataFrame([metal] * len(substrate))
    sys_cond_0 = pd.concat([matrix, substrate], axis=1)
    sys_cond_0.columns = ['Metal', 'Substrate']
    sys_cond_0 = StrToComposition(target_col_id='Sub_comp').featurize_dataframe(sys_cond_0, 'Substrate')
    data_Sub = magpie.featurize_dataframe(sys_cond_0, col_id="Sub_comp", ignore_errors=True)
    feature_Sub = data_Sub.filter(like='mean')
    feature_Sub.columns = ['Ce_' + j for j in feature_Sub.columns]

    feature_all = pd.concat([feature_Me, feature_Sub], axis=1)

    mental_ceramics_data=pd.concat([metal_matminer.drop(columns=['Me_comp']),sys_cond_0.drop(columns=['Metal','Sub_comp']),feature_all],axis=1)
    return mental_ceramics_data

def data_pre_process():
    data_path=os.path.join(Base_path,Data_path,'wettability_data_sbk_processing.csv')
    df=pd.read_csv(data_path,index_col=0)

    # One-hot encode categorical features
    df = pd.get_dummies(df)
    print(df)
    df.to_csv(os.path.join(Base_path,Data_path,'wettability_data_processed.csv'))

if __name__=='__main__':
    data_pre_process()
