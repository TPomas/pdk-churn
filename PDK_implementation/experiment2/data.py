# New imports
import os
import shutil

import pachyderm_sdk
from pachyderm_sdk.api.pfs import File, FileType

# Old imports
import pandas as pd
import numpy as np

#from utils import preprocess_dataframe

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

# Apply standard scaling to all numerical columns of df, using mean and std given in json file
def scale_data(df, json_numscale="numscale.json"):
    f = open(json_numscale)
    scale_dict = json.load(f)
    f.close()
    for col in scale_dict:
        df[col] = (df[col] - scale_dict[col]["mean"]) / scale_dict[col]["std"]
    
    return df

# One-hot encode all categorical features, assuming we know all values they may take
def encode_categories(df):
    expected_categories = {}
    expected_categories["new_cell"] = ['U','Y','N']
    expected_categories["asl_flag"] = ['N','Y']
    expected_categories["area"] = ['NORTHWEST/ROCKY MOUNTAIN AREA','GREAT LAKES AREA','CHICAGO AREA',
     'NEW ENGLAND AREA','DALLAS AREA','CENTRAL/SOUTH TEXAS AREA',
     'TENNESSEE AREA','MIDWEST AREA','PHILADELPHIA AREA','OHIO AREA',
     'HOUSTON AREA','SOUTHWEST AREA','NEW YORK CITY AREA',
     'ATLANTIC SOUTH AREA','SOUTH FLORIDA AREA','CALIFORNIA NORTH AREA',
     'DC/MARYLAND/VIRGINIA AREA','NORTH FLORIDA AREA','LOS ANGELES AREA']
    expected_categories["dualband"] = ['Y','N','T']
    expected_categories["refurb_new"] = ['N','R']
    expected_categories["hnd_webcap"] = ['WCMB','UNKW','WC']
    expected_categories["marital"] = ['S','M','A','U','B']
    expected_categories["ethnic"] = ['N','U','I','S','F','J','Z','M','H','G','D','O','R','B','P','X','C']
    expected_categories["kid0_2"] = ['U','Y']
    expected_categories["kid3_5"] = ['U','Y']
    expected_categories["kid6_10"] = ['U','Y']
    expected_categories["kid11_15"] = ['U','Y']
    expected_categories["kid16_17"] = ['U','Y']
    expected_categories["creditcd"] = ['Y','N']
    
    for col in expected_categories:
        categorical_col = pd.Categorical(df[col], categories=expected_categories[col], ordered=False)
        one_hot_cols = pd.get_dummies(categorical_col, prefix=col)
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, one_hot_cols], axis=1)
    
    return df

# Make sure not to pass a scaled "reference_df" as argument, since we use its values to scale df
def preprocess_dataframe(df, reference_df, numerical_cols):
    df = scale_data(df)
    df = encode_categories(df)
    return df

class Churn_Dataset(Dataset):
 
  def __init__(self, df, training_cols, label_col):
 
    self.X = torch.tensor(df[training_cols].values.astype(np.float32), dtype=torch.float32)
    self.y = torch.tensor(df[label_col].values.astype(np.float32), dtype=torch.float32).unsqueeze(-1)
 
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self,idx):
    return self.X[idx], self.y[idx]

def get_train_and_validation_datasets(data_files, test_size=0.2, random_seed=42):
    
    # New - handle a list of files, instead of a single file
    df_list = [pd.read_csv(data_file) for data_file in data_files]
    full_df = pd.concat(df_list)
    #full_df = pd.read_csv(data_file)
    
    train_df, val_df = train_test_split(full_df, test_size=test_size, random_state=random_seed)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    object_cols = list(train_df.columns[train_df.dtypes.values == "object"])
    int_cols = list(train_df.columns[train_df.dtypes.values == "int"])
    float_cols = list(train_df.columns[train_df.dtypes.values == "float"])

    # Churn will be the label, no need to preprocess it
    int_cols.remove("churn")

    numerical_cols = int_cols+float_cols
    
    # Keep an unscaled version of train_df for scaling all dataframes
    unscaled_train_df = train_df.copy()

    train_df = preprocess_dataframe(train_df, unscaled_train_df, numerical_cols)
    val_df = preprocess_dataframe(val_df, unscaled_train_df, numerical_cols)
    
    training_cols = list(train_df.columns)
    label_col = "churn"
    training_cols.remove(label_col)
    
    train_dataset = Churn_Dataset(train_df, training_cols, label_col)
    val_dataset = Churn_Dataset(val_df, training_cols, label_col)
    
    return train_dataset, val_dataset

# New - helper function used below
def safe_open_wb(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')

# New - helper function to download data from Pachyderm repository
def download_pach_repo(
    pachyderm_host,
    pachyderm_port,
    repo,
    branch,
    root,
    token,
    project="default",
    previous_commit=None,
):
    print(f"Starting to download dataset: {repo}@{branch} --> {root}")

    if not os.path.exists(root):
        os.makedirs(root)

    client = pachyderm_sdk.Client(
        host=pachyderm_host, port=pachyderm_port, auth_token=token
    )
    files = []
    if previous_commit is not None:
        for diff in client.pfs.diff_file(new_file=File.from_uri(f"{project}/{repo}@{branch}"),
            old_file=File.from_uri(f"{project}/{repo}@{previous_commit}")
        ):
            src_path = diff.new_file.file.path
            des_path = os.path.join(root, src_path[1:])
            print(f"Got src='{src_path}', des='{des_path}'")

            if diff.new_file.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))
    else:
        for file_info in client.pfs.walk_file(file=File.from_uri(f"{project}/{repo}@{branch}")):
            src_path = file_info.file.path
            des_path = os.path.join(root, src_path[1:])
            print(f"Got src='{src_path}', des='{des_path}'")

            if file_info.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))

    for src_path, des_path in files:
        src_file = client.pfs.pfs_file(file=File.from_uri(f"{project}/{repo}@{branch}:{src_path}"))
        print(f"Downloading {src_path} to {des_path}")

        with safe_open_wb(des_path) as dest_file:
            shutil.copyfileobj(src_file, dest_file)

    print("Download operation ended")
    return files
