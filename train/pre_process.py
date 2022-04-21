
from sklearn.model_selection import train_test_split
import utils_wf
import pandas as pd




def split_data(path=None,out_train_path=None,out_val_path=None):
    """
    split original train data into 90% train and 10% validation, save in the ../data/wf/with_validation folder
    """
    if not path:
        path = '../data/wf/train_NoDef_burst.csv'
    X,Y = utils_wf.load_csv_data(path)
    X_train,X_val,y_trian,y_val = train_test_split(X,Y,test_size=0.1,shuffle=True,stratify=Y)
    df_train = pd.DataFrame(X_train)
    df_train['label'] = y_trian
    df_val = pd.DataFrame(X_val)
    df_val['label'] = y_val
    if not out_train_path:
        out_train_path = '../data/wf/with_validation/train_NoDef_burst.csv'
    if not out_val_path:
        out_val_path = '../data/wf/with_validation/val_NoDef_burst.csv'
    utils_wf.write2csv(df_train,out_train_path)
    utils_wf.write2csv(df_val,out_val_path)
    print(f"spliting data completed.")

if __name__ == '__main__':
    path = '../data/wf_ow/train_NoDef_mix.csv'
    out_train_path = '../data/wf_ow/with_validation/train_NoDef_burst.csv'
    out_val_path = '../data/wf_ow/with_validation/val_NoDef_burst.csv'
    split_data(path,out_train_path,out_val_path)