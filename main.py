# %%
#1. Load Libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)


#2. Data Exploration

df_churn_data = pd.read_csv('./data/mergedcustomers_missing_values_GENDER.csv')
df_churn_data.head() # Display the first five rows of the dataset

print('The dataset contains columns of the following dat types: \n' + str(df_churn_data.dtypes))
print('The dataset contains following number of records: for each of the columns : \n' + str(df_churn_data.count()))
print('Each category within the churndisk column has the following counts: ')
print(df_churn_data.groupby('CHURNRISK').size())

index = ['High', 'Medium', 'Low']
churn_plot = df_churn_data['CHURNRISK'].value_counts(sort=True, ascending=False).plot(kind='bar', figsize=(4, 4),title='Total number for occurences of churn risk ' + str(df_churn_data['CHURNRISK'].count()), color=['#BB6B5A','#8CCB9B','#E5E88B'])
churn_plot.set_xlabel('Churn Risk')
churn_plot.set_ylabel('Frequency')

plt.show()
