import pandas as pd

# Load the dataset (update the path to your file)
file_path = 'path_to_your_dataset.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Display dataset columns and the first few rows
print("Dataset columns:\n", data.columns)
print("First few rows:\n", data.head())

# Inspect missing values in the dataset
print("Missing values before cleaning:\n", data.isnull().sum())

# Drop rows or columns with a high number of missing values (optional, based on threshold)
row_threshold = 0.5  # Keep rows with at least 50% non-NaN values
col_threshold = 0.5  # Keep columns with at least 50% non-NaN values
data = data.dropna(axis=0, thresh=len(data.columns) * row_threshold)  # Drop rows
data = data.dropna(axis=1, thresh=len(data) * col_threshold)  # Drop columns

# Recheck missing values after initial cleaning
print("Missing values after initial cleaning:\n", data.isnull().sum())

# Selected features for analysis
features = [
    'dec', 'ra', 'st_age', 'st_ageerr', 'st_bmv', 'st_bmverr', 'st_coronagflag',
    'st_dist', 'st_disterr1', 'st_disterr2', 'st_eeidau', 'st_eeidauerr',
    'st_eeidmas', 'st_eeidmaserr', 'st_exocatflag', 'st_fpbearth', 'st_fpbeartherr',
    'st_fracplxunc', 'st_glat', 'st_glon', 'st_h2m', 'st_h2merr', 'st_h2mlim',
    'st_j2m', 'st_j2merr', 'st_j2mlim', 'st_k2campaign', 'st_k2flag', 'st_ks2m',
    'st_ks2merr', 'st_ks2mlim', 'st_lbol', 'st_lbolerr', 'st_lbtiflag', 'st_logg',
    'st_loggerr', 'st_mass', 'st_masserr', 'st_mbol', 'st_mbolerr', 'st_mbolflag',
    'st_mbolsrc', 'st_metfe', 'st_metfeerr', 'st_metfesrc', 'st_naxa', 'st_nglc',
    'st_nimg', 'st_nplc', 'st_nrvc', 'st_nspec', 'st_nts', 'st_pmdec',
    'st_pmdecerr', 'st_pmra', 'st_pmraerr', 'st_ppnum', 'st_probeflag', 'st_rad',
    'st_raderr', 'st_rvflag', 'st_starshadeflag', 'st_teff', 'st_tefferr',
    'st_vmag', 'st_vmagearth', 'st_vmageartherr', 'st_vmagerr', 'st_vmk',
    'st_vmkerr', 'st_wfirstflag', 'st_wise1', 'st_wise1err', 'st_wise1lim',
    'st_wise2', 'st_wise2err', 'st_wise2lim', 'st_wise3', 'st_wise3err',
    'st_wise3lim', 'st_wise4', 'st_wise4err', 'st_wise4lim', 'wds_deltamag',
    'wds_sep'
]

# Verify if the features exist in the dataset
features = [f for f in features if f in data.columns]
print("Selected Features:", features)

# Analyze missing values in selected features
missing_feature_data = data[features].isnull().sum()
print("Missing values in selected features:\n", missing_feature_data)

# Drop features with all missing values
empty_features = missing_feature_data[missing_feature_data == len(data)].index.tolist()
data = data.drop(columns=empty_features)
features = [f for f in features if f not in empty_features]
print("Features retained after removing empty columns:", features)

# Impute remaining missing values for numeric columns
data[features] = data[features].fillna(data[features].median())

# Drop rows with any remaining missing values in the selected features
data = data.dropna(subset=features)
print("Data shape after dropping rows:", data.shape)

# Raise an error if no valid rows remain
if data.empty:
    raise ValueError("No valid rows remaining after feature selection. Refine feature selection or handle missing data better.")

# Final dataset information
print("Final dataset shape:", data.shape)
print("Final dataset preview:\n", data.head())