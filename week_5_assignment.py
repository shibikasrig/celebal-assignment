
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full_data = pd.concat([train, test], sort=False)

missing = full_data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Fill categorical features with mode
cat_cols = full_data.select_dtypes(include='object').columns
for col in cat_cols:
    full_data[col] = full_data[col].fillna(full_data[col].mode()[0])

# Fill numerical features with median
num_cols = full_data.select_dtypes(include=np.number).columns
for col in num_cols:
    full_data[col] = full_data[col].fillna(full_data[col].median())

# Total Square Footage
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']

# Age of the house
full_data['HouseAge'] = full_data['YrSold'] - full_data['YearBuilt']

# Years since last remodel
full_data['SinceRemodel'] = full_data['YrSold'] - full_data['YearRemodAdd']

# Is the house remodeled?
full_data['IsRemodeled'] = (full_data['YearBuilt'] != full_data['YearRemodAdd']).astype(int)

# Label Encoding for ordinal features
ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC']

# Mapping quality to numbers
qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}
for col in ordinal_features:
    full_data[col] = full_data[col].map(qual_mapping)

# One-hot encoding for nominal features
full_data = pd.get_dummies(full_data)

train_cleaned = full_data[:len(train)]
test_cleaned = full_data[len(train):]
train_labels = train['SalePrice']

print(train_cleaned.shape)
print(test_cleaned.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(train_cleaned, train_labels)

# Predict on training set
train_preds = model.predict(train_cleaned)
rmse = np.sqrt(mean_squared_error(train_labels, train_preds))
print(f"Train RMSE: {rmse:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt

# Check basic info
print(train.info())
print(train.describe())

# Visualize missing values
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values in Training Data")
plt.show()