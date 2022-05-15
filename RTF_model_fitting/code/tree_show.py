import graphviz
import os
Base_path='D:\RTF_data_sbk'
Data_path='data_found\\data'
result_path='RTF_model_fitting\\result'

if __name__ == '__main__':
    with open(os.path.join(Base_path, result_path, '../result/tree.dot')) as f:
        dot_graph = f.read()
    dot=graphviz.Source(dot_graph)
    dot.view()