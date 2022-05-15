import pandas as pd
from matminer.featurizers.conversions import StrToComposition #class to convert str to composition
from matminer.featurizers.composition import ElementProperty #Class to calculate elemental property attributes.
import os
import numpy as np

magpie = ElementProperty.from_preset(
    preset_name="magpie")  # Return ElementProperty from a preset string, different kinds

Base_path='D:\Data_sbk_fitting'
Data_path='data_found\\data'

def data_supplement():
    # using method to supplement missing data
    pass

def data_pre_process():
    # one-hot coding to transform english name to number
    data_path=os.path.join(Base_path,Data_path,'wettability_data_sbk_processing.csv')
    df=pd.read_csv(data_path,index_col=0)

    # One-hot encode categorical features
    df = pd.get_dummies(df)
    print(df)
    df.to_csv(os.path.join(Base_path,Data_path,'wettability_data_processed.csv'))

if __name__=='__main__':
    data_pre_process()

    data_supplement()

