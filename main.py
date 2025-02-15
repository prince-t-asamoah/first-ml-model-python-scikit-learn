# %%
# 1. Load Libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import model_metrics, three_d_compare, two_d_compare
warnings.filterwarnings("ignore", category=UserWarning)


# %%
# 2. Data Exploration
# Load the dataset
df_churn_data = pd.read_csv('./data/mergedcustomers_missing_values_GENDER.csv')
df_churn_data.head()  # Display the first five rows of the dataset

print('The dataset contains columns of the following dat types: \n' +
      str(df_churn_data.dtypes))
print('The dataset contains following number of records: for each of the columns : \n' +
      str(df_churn_data.count()))
print('Each category within the churndisk column has the following counts: ')
print(df_churn_data.groupby('CHURNRISK').size())

index = ['High', 'Medium', 'Low']
churn_plot = df_churn_data['CHURNRISK'].value_counts(sort=True, ascending=False).plot(kind='bar', figsize=(
    4, 4), title='Total number for occurences of churn risk ' + str(df_churn_data['CHURNRISK'].count()), color=['#BB6B5A', '#8CCB9B', '#E5E88B'])
churn_plot.set_xlabel('Churn Risk')
churn_plot.set_ylabel('Frequency')

plt.show()
# %%
# 3. Data Preprocessing with Scikit-Learn

df_churn_data = df_churn_data.drop(['ID'], axis=1) # Remove columns not required for the analysis
df_churn_data.head()

categorical_columns = ['GENDER', 'STATUS', 'HOMEOWNER'] # Categorical columns in the dataset

print('Categorical columns in the dataset are: ')
print(categorical_columns)
print('\n')

impute_categorical = SimpleImputer(strategy='most_frequent') # Fill missing values with the most frequent value
onehot_categorical = OneHotEncoder(handle_unknown='ignore') # Convert categories of data type string into numbers

categorical_transformer = Pipeline(
    steps=[('impute', impute_categorical), ('onehot', onehot_categorical)])

numerical_columns = df_churn_data.select_dtypes(
    include=[float, int]).columns  # Numerical columns in the dataset
print('Numerical columns in the dataset are: ')
print(numerical_columns)
print('\n')

scaler_numberical = StandardScaler()
numerical_transformer = Pipeline(
    steps=[('scaler', scaler_numberical)])  # Scale the numerical columns

preprocessorForCategoricalColumns = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_columns)])
preprocessorForAllColumns = ColumnTransformer(transformers=[(
    'cat', categorical_transformer, categorical_columns), ('num', numerical_transformer, numerical_columns)])

# The transformation happens in the piepeline. Temporrily done here to show what intermediate value looks like
df_churn_pd_temp_1 = preprocessorForCategoricalColumns.fit_transform(
    df_churn_data)
print('The transformed dataset after preprocessing the categorical columns is: ')
print(df_churn_pd_temp_1)
print('\n')

df_churn_pd_temp_2 = preprocessorForAllColumns.fit_transform(df_churn_data)
print('The transformed dataset after preprocessing all the columns is: ')
print(df_churn_pd_temp_2)
print('\n')

# Prepare data frame for spitting data into train and test datasets
features = []
features  = df_churn_data.drop(['CHURNRISK'], axis=1)

label_churn = pd.DataFrame(df_churn_data, columns=['CHURNRISK'])
label_encoder = LabelEncoder()
label = df_churn_data['CHURNRISK']

label  = label_encoder.fit_transform(label)
print('Encoded value of Churnrisk after applying label encoder: ' + str(label))

# %%
# 4. Spliting Data for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(features, label, random_state=0)
print('Dimensios of datasets that will be used for training : Input features' + str(x_train.shape) + ' Output features' + str(y_train.shape))
print('Dimensios of datasets that will be used for testing : Input features' + str(x_test.shape) + ' Output features' + str(y_test.shape))

# %%
# 5. Preparing a classification model
model_name = 'Random Forest Classifier'
randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# %%
# 6.  Assembling the steps using pipeline
rfc_model = Pipeline(steps=[('preprocessorAll', preprocessorForAllColumns), ('classifier', randomForestClassifier)])
rfc_model.fit(x_train, y_train)

# %%
#7. Running predictions on model
y_pred_rfc = rfc_model.predict(x_test)

# %%
# 8. Model Evaluation and Visualization
two_d_compare(y_test, y_pred_rfc, model_name, x_test)
three_d_compare(y_test, y_pred_rfc, model_name, x_test)

y_test = label_encoder.inverse_transform(y_test)
y_pred_rfc = label_encoder.inverse_transform (y_pred_rfc)
model_metrics(y_test, y_pred_rfc)