import pandas as pd
import numpy as np
import os

#print
# =============================================================================
# SECTION 1 — HIGH-LEVEL PIPELINE: LOAD ALL DATA
# load every modality (cell features + drug features)
# load screening matrix (response labels)
# match them so all share same overlapping cell and drug names
# =============================================================================

def load_all(data_modalities, cell, screen, drug):
    print("loading raw data")
    # this one is to use match data part
    data_dict = {}
    for modality in data_modalities:
        if modality.startswith("cell"):
            folder = cell
        else:
            folder = drug
        df = load_one(folder, modality)
        data_dict[modality] = df

    print("loading screening")
    screening = pd.read_csv(screen, sep="\t", index_col=0)

    print("matching data with screening")
    data_dict, screening = match_data(data_dict, screening)

    return data_dict, screening


# =============================================================================
# SECTION 2 — LOAD ONE MODALITY FILE
# read it, sort rows/columns
# normalize all values to [0,1] because it's a gene expression 
# replace NaN with zero
# =============================================================================

def load_one(folder, modality):
    selected_file = None
    file_list = os.listdir(folder)
    #to ignore useless files in a folder
    for file_name in file_list:
        if file_name.startswith(modality):
            if "_raw.tsv" in file_name:
                selected_file = file_name
                break

    full_path = os.path.join(folder, selected_file)
    #we won't need index
    df = pd.read_csv(full_path, sep="\t", index_col=0)

    #sort rows to join screening with gene and drugs
    df = df.sort_index()
    #aort column
    df = df.sort_index(axis=1)
    print("file completed")
    
    # normalize each column to [0, 1]
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min == 0:
            df[col] = 0
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    # 0 instead of nan
    df = df.fillna(0)
    return df



# =============================================================================
# SECTION 3 — MATCH RAW DATA WITH SCREENING
# find common cells in all cell modalities.
# find common drugs in all drug modalities.
# remove duplicates
# turn into new data frame.
# (if a cell line exists in cell_expr but not in cell_cnv, we cannot use it)
# =============================================================================

def match_data(data_dict, screening):
    #intersection
    # find common cell names across all cell modalities
    cell_names = None
    for key in data_dict:
        if key.startswith("cell"):
            # index is cell line names. we turn into set to find intersections easily
            current_names = set(data_dict[key].index)
            if cell_names is None:
                cell_names = current_names
            else:
                cell_names = cell_names.intersection(current_names)

    # find common drug names in all drug modalities
    drug_names = None
    for key in data_dict:
        if key.startswith("drug"):
            current_names = set(data_dict[key].index)
            if drug_names is None:
                drug_names = current_names
            else:
                drug_names = drug_names.intersection(current_names)

    # intersect with screening rows (cells)
    screening_cell_names = set(screening.index)
    common_cells = cell_names.intersection(screening_cell_names)
    # Pandas .loc[] does not accept a set which we will use later
    common_cells = list(common_cells)

    # intersect with screening columns (drugs)
    screening_drug_names = set(screening.columns)
    common_drugs = drug_names.intersection(screening_drug_names)
    common_drugs = list(common_drugs)

    # make new table with commons and removed duplicates
    for key in data_dict:
        if key.startswith("cell"):
            df = data_dict[key]
            df = df.loc[common_cells]
            df = df.loc[~df.index.duplicated()]
            data_dict[key] = df
        if key.startswith("drug"):
            df = data_dict[key]
            df = df.loc[common_drugs]
            # true means to keep this row so duplicates are deleted
            df = df.loc[~df.index.duplicated()]
            data_dict[key] = df

    # new screening table
    screening = screening.loc[common_cells, common_drugs]
    return data_dict, screening



# =============================================================================
# SECTION 4 — SORT (CELL, DRUG) PAIRS
# this pairs will be used in the last func as index
# sort pairs by drug index first, then by cell index so we can find patterns in all tables
# =============================================================================

def sort_pairs_by_drug_then_cell(pairs_array):
    """
    cell (rows)
    drug (columns)
    """
    pairs = pairs_array
    new = []
    for p in pairs:
        new.append([p[1], p[0]])
    new.sort()
    sorted_pairs = []
    for p in new:
        sorted_pairs.append([p[1], p[0]])
    return np.array(sorted_pairs, dtype=int)


# =============================================================================
# SECTION 5 — PREPARE FINAL TRAINING DATA
# handle resistant (1) and sensitive (-1)
# we need: main_cell, main_drug, label, cell_sizes, drug_sizes
# index: (cell,drug)
# =============================================================================

def train_data(data_dict, screening):
    print("final training data:")
    # find resistant pairs (screening == 1) and we return index
    resistant_array = np.argwhere(screening.values == 1)
    resistant = sort_pairs_by_drug_then_cell(resistant_array)


    # find sensitive pairs (screening == -1)
    sensitive_array = np.argwhere(screening.values == -1)
    sensitive = sort_pairs_by_drug_then_cell(sensitive_array)

    print("resistant count:", len(resistant))
    print("sensitive count:", len(sensitive))

    # build cell matrix (main_cell)
    cell_types = []
    for key in data_dict:
        if key.startswith("cell"):
            cell_types.append(key)
    cell_types.sort()

    cell_df_list = []
    for cell_type in cell_types:
        df = data_dict[cell_type]
        new_cols = []
        for col in df.columns:
            new_cols.append(col + "_" + cell_type)
        df.columns = new_cols
        cell_df_list.append(df)
    cell_data = pd.concat(cell_df_list, axis=1)

    cell_data_sizes = []
    for cell_type in cell_types:
        #(dataframe).shape  →  (number_of_rows , number_of_columns)
        size = data_dict[cell_type].shape[1]
        cell_data_sizes.append(size)

    # build drug matrix (main_drug)
    drug_types = []
    for key in data_dict:
        if key.startswith("drug"):
            drug_types.append(key)
    drug_types.sort()

    drug_df_list = []
    for drug_type in drug_types:
        df = data_dict[drug_type]

        new_cols = []
        for col in df.columns:
            new_cols.append(col + "_" + drug_type)
        df.columns = new_cols
        drug_df_list.append(df)

    drug_data = pd.concat(drug_df_list, axis=1)

    #number of features for each modality
    drug_data_sizes = []
    for drug_type in drug_types:
        #(dataframe).shape  →  (number_of_rows , number_of_columns)
        size = data_dict[drug_type].shape[1]
        drug_data_sizes.append(size)
        
    # to create resistant and sensitive tables (4)
    # cell_data is concated
    # resistant: [row_index , column_index]
    # resistant[:, 0] means is: all row(cell) indices/ resistant[:, 1] is: all column(drug) indices
    # iloc[row, column]
    r_cell = cell_data.iloc[resistant[:, 0], :]
    r_drug = drug_data.iloc[resistant[:, 1], :]
    r_cell = r_cell.reset_index(drop=True)
    r_drug = r_drug.reset_index(drop=True)

    s_cell = cell_data.iloc[sensitive[:, 0], :]
    s_drug = drug_data.iloc[sensitive[:, 1], :]
    s_cell = s_cell.reset_index(drop=True)
    s_drug = s_drug.reset_index(drop=True)

    # combine all resistant and sensitive in one table
    main_cell = pd.concat([r_cell, s_cell], axis=0)
    main_drug = pd.concat([r_drug, s_drug], axis=0)


    new_index = []

    # resistant names
    # resistan = [row_index, drug_index]
    # (cell, drug) as index
    for item in resistant:
        row_index = item[0]
        col_index = item[1]
        cell_name = screening.index[row_index]
        drug_name = screening.columns[col_index]
        pair_name = "(" + cell_name + "," + drug_name + ")"
        new_index.append(pair_name)
    # sensitive index
    for item in sensitive:
        row_index = item[0]
        col_index = item[1]
        cell_name = screening.index[row_index]
        drug_name = screening.columns[col_index]
        pair_name = "(" + cell_name + "," + drug_name + ")"
        new_index.append(pair_name)
    main_cell.index = new_index
    main_drug.index = new_index
    # drug and cell may be repeated alone but but once they are combined, they become unique
    # so no deplicates here 
    
    # create Y which shows resistant\sensitive
    zeros_part = np.zeros(len(resistant))
    ones_part = np.ones(len(sensitive))
    label = np.concatenate([zeros_part, ones_part])
    # ===== Combine all into one table (for viewing only) =====
    complete_table = pd.concat([main_cell, main_drug], axis=1)
    complete_table["Label"] = label
    
    return main_cell, main_drug, label, cell_data_sizes, drug_data_sizes , complete_table
