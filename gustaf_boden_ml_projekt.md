

## Loading and Preliminary Data Exploration


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(category=FutureWarning, action='ignore')

df = pd.read_csv('project_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.info()
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12330 entries, 0 to 12329
    Data columns (total 18 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   Administrative           12330 non-null  int64  
     1   Administrative_Duration  12330 non-null  float64
     2   Informational            12330 non-null  int64  
     3   Informational_Duration   12330 non-null  float64
     4   ProductRelated           12330 non-null  int64  
     5   ProductRelated_Duration  12330 non-null  float64
     6   BounceRates              12330 non-null  float64
     7   ExitRates                12330 non-null  float64
     8   PageValues               12330 non-null  float64
     9   SpecialDay               12207 non-null  float64
     10  Month                    12330 non-null  object 
     11  OperatingSystems         12330 non-null  int64  
     12  Browser                  12146 non-null  float64
     13  Region                   12084 non-null  float64
     14  TrafficType              12330 non-null  int64  
     15  VisitorType              12330 non-null  object 
     16  Weekend                  12330 non-null  object 
     17  Revenue                  12183 non-null  object 
    dtypes: float64(9), int64(5), object(4)
    memory usage: 1.7+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Administrative</th>
      <th>Administrative_Duration</th>
      <th>Informational</th>
      <th>Informational_Duration</th>
      <th>ProductRelated</th>
      <th>ProductRelated_Duration</th>
      <th>BounceRates</th>
      <th>ExitRates</th>
      <th>PageValues</th>
      <th>SpecialDay</th>
      <th>Month</th>
      <th>OperatingSystems</th>
      <th>Browser</th>
      <th>Region</th>
      <th>TrafficType</th>
      <th>VisitorType</th>
      <th>Weekend</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>1</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>64.000000</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>2</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>4</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2.666667</td>
      <td>0.05</td>
      <td>0.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>3</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4</td>
      <td>Returning_Visitor</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>10</td>
      <td>627.500000</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Feb</td>
      <td>3</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4</td>
      <td>Returning_Visitor</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Data Summary
from fast_ml.utilities import display_all
from fast_ml import eda

summary_df = eda.df_info(df)
display_all(summary_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_type</th>
      <th>data_type_grp</th>
      <th>num_unique_values</th>
      <th>sample_unique_values</th>
      <th>num_missing</th>
      <th>perc_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Administrative</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>37</td>
      <td>[0, 1, 2, 4, 12, 3, 10, 6, -3, -8]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Administrative_Duration</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>3345</td>
      <td>[0.0, 53.0, 64.6, 6.0, 18.0, 9.0, 56.0, 16.0, ...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Informational</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>17</td>
      <td>[0, 1, 2, 4, 16, 5, 3, 14, 6, 12]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Informational_Duration</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>1258</td>
      <td>[0.0, 120.0, 16.0, 94.0, 93.0, 75.0, 19.0, 22....</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ProductRelated</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>311</td>
      <td>[1, 2, 10, 19, 0, 3, 16, 7, 6, 23]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ProductRelated_Duration</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>9551</td>
      <td>[0.0, 64.0, 2.666666667, 627.5, 154.2166667, 3...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BounceRates</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>2408</td>
      <td>[0.2, 0.0, 0.05, 0.02, 0.015789474, 1.76799007...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ExitRates</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>4777</td>
      <td>[0.2, 0.1, 0.14, 0.05, 0.024561404, 0.02222222...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PageValues</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>2704</td>
      <td>[0.0, 54.17976426, 19.44707913, 38.30849268, 2...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SpecialDay</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>6</td>
      <td>[0.0, 0.4, 0.8, 1.0, 0.2, 0.6, nan]</td>
      <td>123</td>
      <td>0.997567</td>
    </tr>
    <tr>
      <th>Month</th>
      <td>object</td>
      <td>Categorical</td>
      <td>12</td>
      <td>[Feb, Mar, May, Turc, Oct, June, Jul, Aug, Nov...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>OperatingSystems</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>8</td>
      <td>[1, 2, 4, 3, 7, 6, 8, 5]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Browser</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>13</td>
      <td>[nan, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0,...</td>
      <td>184</td>
      <td>1.492295</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>9</td>
      <td>[1.0, nan, 2.0, 3.0, 4.0, 9.0, 5.0, 6.0, 7.0, ...</td>
      <td>246</td>
      <td>1.995134</td>
    </tr>
    <tr>
      <th>TrafficType</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>20</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>VisitorType</th>
      <td>object</td>
      <td>Categorical</td>
      <td>3</td>
      <td>[Returning_Visitor, New_Visitor, Other]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Weekend</th>
      <td>object</td>
      <td>Categorical</td>
      <td>3</td>
      <td>[False, True, Name:Zara]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Revenue</th>
      <td>object</td>
      <td>Categorical</td>
      <td>2</td>
      <td>[False, nan, True]</td>
      <td>147</td>
      <td>1.192214</td>
    </tr>
  </tbody>
</table>
</div>


## Dataset Introduction
The dataset consists of feature vectors belonging to 12,330 sessions.
The dataset was formed so that each session
would belong to a different user in a 1-year period to avoid
any tendency to a specific campaign, special day, user
profile, or period.

| Feature                 | Description                                                                                                                                                                                            | Role     | Type        | Missing Values |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-------------|----------------|
| Administrative          | This is the number of pages of this type (administrative) that the user visited.                                                                                                                       | Feature  | Integer     | no             |
| Administrative_Duration | This is the amount of time spent in this category of pages.                                                                                                                                            | Feature  | Integer     | no             |
| Informational           | This is the number of pages of this type (informational) that the user visited.                                                                                                                         | Feature  | Integer     | no             |
| Informational_Duration  | This is the amount of time spent in this category of pages.                                                                                                                                            | Feature  | Integer     | no             |
| ProductRelated          | This is the number of pages of this type (product related) that the user visited.                                                                                                                       | Feature  | Integer     | no             |
| ProductRelated_Duration | This is the amount of time spent in this category of pages.                                                                                                                                            | Feature  | Continuous  | no             |
| BounceRates             | The percentage of visitors who enter the website through that page and exit without triggering any additional tasks.                                                                                   | Feature  | Continuous  | no             |
| ExitRates               | The percentage of pageviews on the website that end at that specific page.                                                                                                                              | Feature  | Continuous  | no             |
| PageValues              | The average value of the page averaged over the value of the target page and/or the completion of an eCommerce transaction.                                                                            | Feature  | Integer     | no             |
| SpecialDay              | This value represents the closeness of the browsing date to special days or holidays (eg Mother's Day or Valentine's day) in which the transaction is more likely to be finalized.                     | Feature  | Integer     | 123            |
| Month                   | Contains the month the pageview occurred, in string form.                                                                                                                                              | Feature  | Categorical | no             |
| OperatingSystems        | An integer value representing the operating system that the user was on when viewing the page.                                                                                                         | Feature  | Integer     | no             |
| Browser                 | An integer value representing the browser that the user was using to view the page.                                                                                                                    | Feature  | Integer     | 184            |
| Region                  | An integer value representing which region the user is located in.                                                                                                                                     | Feature  | Integer     | 246            |
| TrafficType             | An integer value representing what type of traffic the user is categorized into.                                                                                                                       | Feature  | Integer     | no             |
| VisitorType             | A string representing whether a visitor is New Visitor, Returning Visitor, or Other.                                                                                                                   | Feature  | Categorical | no             |
| Weekend                 | A boolean representing whether the session is on a weekend.                                                                                                                                            | Feature  | Binary      | no             |
| Revenue                 | A boolean representing whether or not the user completed the purchase.                                                                                                                                 | Target   | Binary      | 147            |


## Data Cleaning
We meticulously examine the dataset to identify and rectify missing or invalid values, incorrect data types, and other inconsistencies that may hinder analysis and predictive modeling.


```python
import missingno as msno

# Visualize missing data
msno.matrix(df)
plt.title('Missing Values', fontsize=16)
plt.show()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_8_0.png)
    



```python
# Missing values in target are not feasible and are dropped
missing_values_target = df['Revenue'].isna().sum()
df.dropna(subset=['Revenue'], inplace=True)
print(f"\nDropped {missing_values_target} rows with missing target values.")

# Likewise goes for duplicated rows
duplicate_rows_dropped = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"\nDropped {duplicate_rows_dropped} duplicate rows.")
```

    
    Dropped 147 rows with missing target values.
    
    Dropped 92 duplicate rows.
    


```python
# Calculate the impact of dropping rows with missing values
df_temp = df.copy()
rows_before_drop = len(df_temp)
df_temp.dropna(inplace=True)
rows_dropped = rows_before_drop - len(df_temp)
percentage_dropped = (rows_dropped / rows_before_drop) * 100
print(f"\nImpact of dropping all rows with missing values: {rows_dropped} rows ({percentage_dropped:.2}% of the dataset)")
del df_temp
```

    
    Impact of dropping all rows with missing values: 540 rows (4.5% of the dataset)
    

### Missing values in 'Region', 'Browser', and 'SpecialDay' columns
- The 'Region', 'Browser', and 'SpecialDay' columns contain categorical data
- There are missing values (NaN) in these columns

**Assessment**: As these columns contain categorical data, missing values cannot be easily imputed without additional context. Removing the rows with missing data may result in loss of potentially valuable information.

**Action**: We will keep the missing values for now. During preprocessing, we will consider introducing imputation strategies. Model performance outcomes will inform whether to employ imputation strategies or omit rows with missing values.



```python
# Isolating columns with missing values
df.isna().sum().loc[lambda x: x > 0].sort_values(ascending=False)
```




    Region        244
    Browser       182
    SpecialDay    122
    dtype: int64



### Inspecting Subcategories
When inspecting the subcategories for the object data, we notice some atypical values that will be addressed in the upcoming sections.


```python
# Categorical unique values
object_cols = df.select_dtypes("object")
for i in object_cols:
    print(f"{i}")
    print(df[i].unique())
    print("")
```

    Month
    ['Feb' 'Mar' 'May' 'Turc' 'Oct' 'June' 'Jul' 'Aug' 'Nov' 'Sep' 'Sept'
     'Dec']
    
    VisitorType
    ['Returning_Visitor' 'New_Visitor' 'Other']
    
    Weekend
    ['False' 'True' 'Name:Zara']
    
    Revenue
    [False True]
    
    

### Unexpected Values in 'Month'
- There is an unexpected value "Turc" in the 'Month' column
- September is represented under two seperate categories: "Sep" and "Sept"
- The months "Jan" and "April" are missing

**Assessment**: Without additional context, we cannot make assumptions about what "Turc" represents. These rows could belong to any month, including January, February, or others.

**Action:** To prevent erroneous interpretations, these "Turc" entries in the month will be replaced with NaN values. Similar to the previous section, we may consider relabeling "Turc" or imputing it as, for example, "Unknown", if it demonstrates predictive significance. "Sep" and "Sept" are consolidated.


```python
month_counts = df['Month'].value_counts()
for month, count in month_counts.items():
    print(f'Month "{month}" count: {count}')
```

    Month "May" count: 3131
    Month "Nov" count: 2951
    Month "Mar" count: 1852
    Month "Dec" count: 1689
    Month "Oct" count: 545
    Month "Aug" count: 427
    Month "Jul" count: 426
    Month "Sep" count: 353
    Month "June" count: 283
    Month "Feb" count: 179
    Month "Turc" count: 168
    Month "Sept" count: 87
    


```python
# Relabel 'Turc' to NaN
df['Month'] = df['Month'].replace('Turc', np.nan)

# Consolidate "Sept" with "Sep"
df['Month'] = df['Month'].replace('Sept', 'Sep')
```


```python
print(f"There is now {df['Month'].isna().sum()} additional missing values.")
```

    There is now 168 additional missing values.
    

### Weekend: Handling Unexpected 'Name:Zara' Value
The 'Weekend' feature should be binary, but 189 rows contain the unexpected value 'Name:Zara'.

**Assessment**: The presence of 'Name:Zara' in a binary feature is concerning and may indicate erroneous entries.

**Preprocessing Options**:
- Drop the 'Weekend' feature if it lacks predictive value.
- If valuable we can consider imputation or dropping rows with unknown 'Weekend' values.

**Action**: Replace 'Name:Zara' with NaN in the 'Weekend' feature for now, allowing further investigation and preprocessing decisions.


```python
df['Weekend'].value_counts()
```




    Weekend
    False        9081
    True         2826
    Name:Zara     184
    Name: count, dtype: int64




```python
# Relabel 'Name:Zara' to NaN
df['Weekend'] = df['Weekend'].replace('Name:Zara', np.nan)
```

### Investigation of Anomalies in Continuous Features
Descriptive statistics reveal anomalies in the following features:
- 'Administrative' contains 122 negative values.
    - Trivia: 'Administrative_Duration' max value 989493 indicates some 'forget' to log out from their accounts
- 'BounceRates' contains: 
    - 357 negative values
    - 243 inaccurate values above 1

These anomalies may stem from data entry errors, measurement or tracking issues, or system bugs.

Unfortunately, within the confines of this dataset alone, it's impossible to determine the intended values for these anomalies. Therefore, they will be treated as NaN values.

The handling of these now additional 722 missing values will be explored further during the preprocessing pipeline.


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Administrative</th>
      <td>12091.0</td>
      <td>2.254983</td>
      <td>3.418783</td>
      <td>-10.00000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>4.000000</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>Administrative_Duration</th>
      <td>12091.0</td>
      <td>1293.862319</td>
      <td>34406.147068</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>9.00000</td>
      <td>94.650000</td>
      <td>989493.000000</td>
    </tr>
    <tr>
      <th>Informational</th>
      <td>12091.0</td>
      <td>0.507071</td>
      <td>1.271568</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>Informational_Duration</th>
      <td>12091.0</td>
      <td>34.699421</td>
      <td>141.381512</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2549.375000</td>
    </tr>
    <tr>
      <th>ProductRelated</th>
      <td>12091.0</td>
      <td>31.944256</td>
      <td>44.552279</td>
      <td>0.00000</td>
      <td>7.000000</td>
      <td>18.00000</td>
      <td>38.000000</td>
      <td>705.000000</td>
    </tr>
    <tr>
      <th>ProductRelated_Duration</th>
      <td>12091.0</td>
      <td>1203.140449</td>
      <td>1920.334676</td>
      <td>0.00000</td>
      <td>191.000000</td>
      <td>606.20000</td>
      <td>1474.500000</td>
      <td>63973.522230</td>
    </tr>
    <tr>
      <th>BounceRates</th>
      <td>12091.0</td>
      <td>0.043189</td>
      <td>0.240376</td>
      <td>-0.49868</td>
      <td>0.000000</td>
      <td>0.00274</td>
      <td>0.017188</td>
      <td>2.098952</td>
    </tr>
    <tr>
      <th>ExitRates</th>
      <td>12091.0</td>
      <td>0.041943</td>
      <td>0.046898</td>
      <td>0.00000</td>
      <td>0.014285</td>
      <td>0.02500</td>
      <td>0.050000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>PageValues</th>
      <td>12091.0</td>
      <td>5.921890</td>
      <td>18.584783</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>361.763742</td>
    </tr>
    <tr>
      <th>SpecialDay</th>
      <td>11969.0</td>
      <td>0.061826</td>
      <td>0.199533</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>OperatingSystems</th>
      <td>12091.0</td>
      <td>2.124225</td>
      <td>0.907893</td>
      <td>1.00000</td>
      <td>2.000000</td>
      <td>2.00000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>Browser</th>
      <td>11909.0</td>
      <td>2.358972</td>
      <td>1.713439</td>
      <td>1.00000</td>
      <td>2.000000</td>
      <td>2.00000</td>
      <td>2.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>11847.0</td>
      <td>3.147970</td>
      <td>2.402297</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>3.00000</td>
      <td>4.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>TrafficType</th>
      <td>12091.0</td>
      <td>4.081217</td>
      <td>4.027246</td>
      <td>1.00000</td>
      <td>2.000000</td>
      <td>2.00000</td>
      <td>4.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summary of invalid values
admin_bad_data_neg = df.loc[df['Administrative'] < 0, 'Administrative'].count()
bounce_bad_data_neg = df.loc[df['BounceRates'] < 0, 'BounceRates'].count()
bounce_bad_data_pos = df.loc[df['BounceRates'] > 1, 'BounceRates'].count()

numerical_feat_total_bad = admin_bad_data_neg + bounce_bad_data_neg + bounce_bad_data_pos

print(f"'Administrative' negative values: {admin_bad_data_neg}")
print(f"'BounceRates' negative values: {bounce_bad_data_neg}")
print(f"'BounceRates' values above 1: {bounce_bad_data_pos}")
print(f"Total invalid values for numerical features: {numerical_feat_total_bad}")
```

    'Administrative' negative values: 122
    'BounceRates' negative values: 357
    'BounceRates' values above 1: 243
    Total invalid values for numerical features: 722
    


```python
# Replace negative and irrational values with NaN
df.loc[df['Administrative'] < 0, 'Administrative'] = np.nan
df.loc[df['BounceRates'] < 0, 'BounceRates'] = np.nan
df.loc[df['BounceRates'] > 1, 'BounceRates'] = np.nan
```

### 'SpecialDay' as Categorical Ordinal
Although currently represented as floating-point numbers, the 'SpecialDay' feature is inherently categorical ordinal data. These values denote the proximity of browsing dates to special occasions. Converting them to string values would be more appropriate, as the floating-point decimals are unnecessary and make EDA more cumbersome. This will be done in a subsequent section. Moreover, the feature will ultimately be encoded into distinct categories.


```python
# Print unique values of 'SpecialDay', disclosing unnecessary decimals
for x in df['SpecialDay'].unique():
    print('{:.20f}'.format(x))
```

    0.00000000000000000000
    0.40000000000000002220
    0.80000000000000004441
    1.00000000000000000000
    0.20000000000000001110
    0.59999999999999997780
    nan
    

### Data Cleaning Summary
After thorough data cleaning, we have encountered several inconsistencies within the dataset. The initial missing value count of 540 (4.5%), not accounting for duplicated rows and missing values in the target, has more than doubled, resulting in a total of 1546 values (12.8% of the dataset).


```python
summary_df = eda.df_info(df)
display_all(summary_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_type</th>
      <th>data_type_grp</th>
      <th>num_unique_values</th>
      <th>sample_unique_values</th>
      <th>num_missing</th>
      <th>perc_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Administrative</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>27</td>
      <td>[0.0, 1.0, 2.0, 4.0, 12.0, 3.0, 10.0, 6.0, nan...</td>
      <td>122</td>
      <td>1.009015</td>
    </tr>
    <tr>
      <th>Administrative_Duration</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>3317</td>
      <td>[0.0, 53.0, 64.6, 6.0, 18.0, 9.0, 56.0, 16.0, ...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Informational</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>17</td>
      <td>[0, 1, 2, 4, 16, 5, 3, 6, 12, 7]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Informational_Duration</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>1242</td>
      <td>[0.0, 120.0, 16.0, 94.0, 93.0, 75.0, 19.0, 22....</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ProductRelated</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>310</td>
      <td>[1, 2, 10, 19, 0, 3, 16, 7, 6, 23]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ProductRelated_Duration</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>9447</td>
      <td>[0.0, 64.0, 2.666666667, 627.5, 154.2166667, 3...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BounceRates</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>1781</td>
      <td>[0.2, 0.0, 0.05, 0.02, 0.015789474, nan, 0.018...</td>
      <td>600</td>
      <td>4.962369</td>
    </tr>
    <tr>
      <th>ExitRates</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>4727</td>
      <td>[0.2, 0.1, 0.14, 0.05, 0.024561404, 0.02222222...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PageValues</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>2672</td>
      <td>[0.0, 54.17976426, 19.44707913, 38.30849268, 2...</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SpecialDay</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>6</td>
      <td>[0.0, 0.4, 0.8, 1.0, 0.2, 0.6, nan]</td>
      <td>122</td>
      <td>1.009015</td>
    </tr>
    <tr>
      <th>Month</th>
      <td>object</td>
      <td>Categorical</td>
      <td>10</td>
      <td>[Feb, Mar, May, nan, Oct, June, Jul, Aug, Nov,...</td>
      <td>168</td>
      <td>1.389463</td>
    </tr>
    <tr>
      <th>OperatingSystems</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>8</td>
      <td>[1, 2, 4, 3, 7, 6, 8, 5]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Browser</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>13</td>
      <td>[nan, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0,...</td>
      <td>182</td>
      <td>1.505252</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>float64</td>
      <td>Numerical</td>
      <td>9</td>
      <td>[1.0, nan, 2.0, 3.0, 4.0, 9.0, 5.0, 6.0, 7.0, ...</td>
      <td>244</td>
      <td>2.01803</td>
    </tr>
    <tr>
      <th>TrafficType</th>
      <td>int64</td>
      <td>Numerical</td>
      <td>20</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>VisitorType</th>
      <td>object</td>
      <td>Categorical</td>
      <td>3</td>
      <td>[Returning_Visitor, New_Visitor, Other]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Weekend</th>
      <td>object</td>
      <td>Categorical</td>
      <td>2</td>
      <td>[False, True, nan]</td>
      <td>184</td>
      <td>1.521793</td>
    </tr>
    <tr>
      <th>Revenue</th>
      <td>object</td>
      <td>Categorical</td>
      <td>2</td>
      <td>[False, True]</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Calculate the impact of dropping rows with missing values
df_temp = df.copy()
rows_before_drop = len(df_temp)
df_temp.dropna(inplace=True)
rows_dropped = rows_before_drop - len(df_temp)
percentage_dropped = (rows_dropped / rows_before_drop) * 100
print(f"\nImpact of dropping all rows with missing values: {rows_dropped} rows ({percentage_dropped:.3}% of the dataset)")
```

    
    Impact of dropping all rows with missing values: 1546 rows (12.8% of the dataset)
    

## Optimizing Memory Usage
By converting columns to appropriate data types and downcasting them, we significantly reduced the dataset's memory footprint from 4.3 MB to 0.45 MB (90% reduction). The key steps include:

1. Dropping missing values
   > Note: The handling of missing values here might differ from the approach used in the modeling phase.
2. Converting specific columns to integer, float, and categorical types
3. Downcasting float, integer, and object columns using the `downcast_dtypes()` function

This optimization leads to:
- Reduced memory consumption
- Improved code performance
- Enhanced data understanding

The memory-optimized dataset facilitates **more efficient and effective exploratory data analysis**.


```python
# Bring out true datatypes of boolean features
print(type(df['Weekend'][1]))
print(type(df['Revenue'][1]))
```

    <class 'str'>
    <class 'bool'>
    


```python
# Get memory usage in bytes
memory_usage_bytes = df.memory_usage(deep=True).sum()

# Convert to megabytes (MB)
memory_usage_mb = memory_usage_bytes / (1024 ** 2)
print(f"Memory usage: {memory_usage_mb:.2f} MB")
```

    Memory usage: 4.29 MB
    


```python
# Drop missing values
df.dropna(inplace=True)

# Reset df index
df.reset_index(drop=True, inplace=True)

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    object_cols = [c for c in df if df[c].dtype == "object"]
    
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    df[object_cols] = df[object_cols].astype("category")
    
    return df

# Convert columns to integer type
int_columns = ['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
df[int_columns] = df[int_columns].astype(int)

# Convert columns to float type
float_columns = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
df[float_columns] = df[float_columns].astype(float)

# Convert object 'Weekend' and boolean 'Revenue' to numeric representation
df['Weekend'] = df['Weekend'].map({'True': 1, 'False': 0})
df['Revenue'] = df['Revenue'].map({True: 1, False: 0})

# Round 'SpecialDay' to 1 decimal place and convert to categorical
df['SpecialDay'] = df['SpecialDay'].round(1).astype('object')

# Downcast integer and float columns
df = downcast_dtypes(df)

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10545 entries, 0 to 10544
    Data columns (total 18 columns):
     #   Column                   Non-Null Count  Dtype   
    ---  ------                   --------------  -----   
     0   Administrative           10545 non-null  int16   
     1   Administrative_Duration  10545 non-null  float32 
     2   Informational            10545 non-null  int16   
     3   Informational_Duration   10545 non-null  float32 
     4   ProductRelated           10545 non-null  int16   
     5   ProductRelated_Duration  10545 non-null  float32 
     6   BounceRates              10545 non-null  float32 
     7   ExitRates                10545 non-null  float32 
     8   PageValues               10545 non-null  float32 
     9   SpecialDay               10545 non-null  category
     10  Month                    10545 non-null  category
     11  OperatingSystems         10545 non-null  int16   
     12  Browser                  10545 non-null  int16   
     13  Region                   10545 non-null  int16   
     14  TrafficType              10545 non-null  int16   
     15  VisitorType              10545 non-null  category
     16  Weekend                  10545 non-null  int16   
     17  Revenue                  10545 non-null  int16   
    dtypes: category(3), float32(6), int16(9)
    memory usage: 464.2 KB
    


```python
# Get memory usage in bytes
memory_usage_bytes = df.memory_usage(deep=True).sum()

# Convert to megabytes (MB)
memory_usage_mb = memory_usage_bytes / (1024 ** 2)
print(f"Memory usage: {memory_usage_mb:.2f} MB")
```

    Memory usage: 0.45 MB
    

## Exploratory Data Analysis (EDA)
### Feature Type Analysis
The table below presents the results of the feature type analysis, classifying features into appropriate types. Two key points to consider for the upcoming modeling phase:

- Categorical features must be encoded during preprocessing, even if they are numerical in data type:
  - Some features, like 'SpecialDay', appears as decimal numbers but represent ordinal categorical data, indicating the closeness of browsing date to special days.
  - Other numeric categorical features, such as 'OperatingSystems', are placeholders for nominal categories without inherent order.
- Sparse numeric discrete features with few unique values may perform poorly. Encoding these features could potentially improve their predictive power.


```python
from great_tables import GT, md, html

# Feature Classification and Metadata Summary
def create_feature_lists(data, custom_features=None):
    '''Generate lists of features based on their types and metadata'''
    
    NUM_FEAT = [feature for feature in data.select_dtypes(include=np.number).columns]
    CONT_FEAT = [feature for feature in NUM_FEAT if data[feature].dtype in ['float64', 'float32', 'float16'] and len(data[feature].unique()) >= 25]
    DISC_FEAT = [feature for feature in NUM_FEAT if data[feature].dtype in ['int64', 'int32', 'int16', 'int8']]
    CAT_FEAT = [feature for feature in data.columns if data[feature].dtype in ['object', 'category', 'bool'] 
            and len(data[feature].unique()) < 13] + [feature for feature in NUM_FEAT if data[feature].dtype in ['float64', 'float32', 'float16'] and len(data[feature].unique()) < 25]
    BINARY_FEAT = [feature for feature in CAT_FEAT if len(data[feature].unique()) == 2] + \
              [feature for feature in NUM_FEAT if len(data[feature].unique()) == 2]

    return NUM_FEAT, CONT_FEAT, DISC_FEAT, CAT_FEAT, BINARY_FEAT, custom_features

# Custom features dictionary
custom_features = {
    'Administrative': {'FeatureType': 'Numeric', 'SubType': 'Discrete'},
    'SpecialDay': {'FeatureType': 'Categorical', 'SubType': 'Ordinal'},
    'Month': {'FeatureType': 'Categorical', 'SubType': 'Ordinal'},
    'Browser': {'FeatureType': 'Categorical', 'SubType': 'Nominal'},
    'OperatingSystems': {'FeatureType': 'Categorical', 'SubType': 'Nominal'},
    'Region': {'FeatureType': 'Categorical', 'SubType': 'Nominal'},
    'TrafficType': {'FeatureType': 'Categorical', 'SubType': 'Nominal'},
    'Region': {'FeatureType': 'Categorical', 'SubType': 'Nominal'},
    'VisitorType': {'FeatureType': 'Categorical', 'SubType': 'Nominal'}
}

# Run Create Feature List Function
num_feat, cont_feat, disc_feat, cat_feat, binary_feat, custom_features = create_feature_lists(df, custom_features)

# Creating a dataframe to store the feature metadata
feature_df = pd.DataFrame(index=df.columns, 
                          columns=['Feature', 'DataType', 'UniqueValues', 'FeatureType', 'SubType'])

# Overview of unique values and feature type
for col in df.columns:
    feature_df.loc[col, 'Feature'] = col
    feature_df.loc[col, 'DataType'] = df[col].dtype
    
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        unique_values = df[col].cat.categories.tolist()[:6] + ['...']
    else:
        unique_values = df[col].unique()
        if len(unique_values) > 6:
            unique_values = unique_values[:6].tolist() + ['...']
    
    if col in cont_feat:
        feature_df.loc[col, 'UniqueValues'] = len(df[col].unique())
    else:
        feature_df.loc[col, 'UniqueValues'] = unique_values
    
    if col in custom_features:
        feature_df.loc[col, 'FeatureType'] = custom_features[col]['FeatureType']
        feature_df.loc[col, 'SubType'] = custom_features[col]['SubType']
    elif col in num_feat:
        feature_df.loc[col, 'FeatureType'] = "Numeric"
        if col in binary_feat:
            feature_df.loc[col, 'SubType'] = "Binary"
        elif col in cont_feat:
            feature_df.loc[col, 'SubType'] = "Continuous"
        elif col in disc_feat:
            feature_df.loc[col, 'SubType'] = "Discrete"
    elif col in cat_feat:
        feature_df.loc[col, 'FeatureType'] = "Categorical"
        if col in binary_feat:
            feature_df.loc[col, 'SubType'] = "Binary"
        else:
            feature_df.loc[col, 'SubType'] = "Nominal_or_Ordinal"

# Create a display table
(
    GT(feature_df, rowname_col="Feature")
    .tab_header(
        title="Feature Metadata Summary",
        subtitle="Descriptive Overview of Dataset Features and Types"
    )
 
)
```




<div id="ufnsmngidj" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#ufnsmngidj table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#ufnsmngidj thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#ufnsmngidj p { margin: 0 !important; padding: 0 !important; }
 #ufnsmngidj .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #ufnsmngidj .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #ufnsmngidj .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #ufnsmngidj .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #ufnsmngidj .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #ufnsmngidj .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #ufnsmngidj .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #ufnsmngidj .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #ufnsmngidj .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #ufnsmngidj .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #ufnsmngidj .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #ufnsmngidj .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #ufnsmngidj .gt_spanner_row { border-bottom-style: hidden !important; }
 #ufnsmngidj .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #ufnsmngidj .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #ufnsmngidj .gt_from_md> :first-child { margin-top: 0 !important; }
 #ufnsmngidj .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #ufnsmngidj .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #ufnsmngidj .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #ufnsmngidj .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #ufnsmngidj .gt_row_group_first td { border-top-width: 2px !important; }
 #ufnsmngidj .gt_row_group_first th { border-top-width: 2px !important; }
 #ufnsmngidj .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #ufnsmngidj .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #ufnsmngidj .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #ufnsmngidj .gt_left { text-align: left !important; }
 #ufnsmngidj .gt_center { text-align: center !important; }
 #ufnsmngidj .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #ufnsmngidj .gt_font_normal { font-weight: normal !important; }
 #ufnsmngidj .gt_font_bold { font-weight: bold !important; }
 #ufnsmngidj .gt_font_italic { font-style: italic !important; }
 #ufnsmngidj .gt_super { font-size: 65% !important; }
 #ufnsmngidj .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #ufnsmngidj .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead class="gt_header">
  <tr>
    <th colspan="5" class="gt_heading gt_title gt_font_normal">Feature Metadata Summary</th>
  </tr>
  <tr>
    <th colspan="5" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Descriptive Overview of Dataset Features and Types</th>
  </tr>
</thead>
<tr class="gt_col_headings">
  <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id=""></th>
  <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="DataType">DataType</th>
  <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="UniqueValues">UniqueValues</th>
  <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="FeatureType">FeatureType</th>
  <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="SubType">SubType</th>
</tr>
<tbody class="gt_table_body">
<tr>
  <th class="gt_row gt_left gt_stub">Administrative</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[0, 1, 4, 2, 12, 3, '...']</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Discrete</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Administrative_Duration</th>
  <td class="gt_row gt_right">float32</td>
  <td class="gt_row gt_right">2983</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Continuous</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Informational</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[0, 1, 2, 4, 16, 5, '...']</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Discrete</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Informational_Duration</th>
  <td class="gt_row gt_right">float32</td>
  <td class="gt_row gt_right">1127</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Continuous</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">ProductRelated</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[2, 10, 19, 1, 0, 3, '...']</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Discrete</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">ProductRelated_Duration</th>
  <td class="gt_row gt_right">float32</td>
  <td class="gt_row gt_right">8387</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Continuous</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">BounceRates</th>
  <td class="gt_row gt_right">float32</td>
  <td class="gt_row gt_right">1672</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Continuous</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">ExitRates</th>
  <td class="gt_row gt_right">float32</td>
  <td class="gt_row gt_right">4211</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Continuous</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">PageValues</th>
  <td class="gt_row gt_right">float32</td>
  <td class="gt_row gt_right">2324</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Continuous</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">SpecialDay</th>
  <td class="gt_row gt_right">category</td>
  <td class="gt_row gt_right">[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Ordinal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Month</th>
  <td class="gt_row gt_right">category</td>
  <td class="gt_row gt_right">['Aug', 'Dec', 'Feb', 'Jul', 'June', 'Mar', '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Ordinal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">OperatingSystems</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[3, 2, 1, 4, 7, 6, '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Nominal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Browser</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[2, 3, 4, 1, 5, 6, '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Nominal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Region</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[2, 1, 3, 4, 5, 9, '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Nominal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">TrafficType</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[4, 3, 5, 2, 1, 6, '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Nominal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">VisitorType</th>
  <td class="gt_row gt_right">category</td>
  <td class="gt_row gt_right">['New_Visitor', 'Other', 'Returning_Visitor', '...']</td>
  <td class="gt_row gt_left">Categorical</td>
  <td class="gt_row gt_left">Nominal</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Weekend</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[0 1]</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Binary</td>
</tr>
<tr>
  <th class="gt_row gt_left gt_stub">Revenue</th>
  <td class="gt_row gt_right">int16</td>
  <td class="gt_row gt_right">[0 1]</td>
  <td class="gt_row gt_left">Numeric</td>
  <td class="gt_row gt_left">Binary</td>
</tr>
</tbody>


</table>

</div>




### Continuous Feature Distributions Analysis
All continuous features exhibit right-skewed distributions with a significant number of outliers. The distributions are far from Gaussian. Having a fairly symmetric predictor, devoid of skewness, would greatly facilitate predictions on the data. Below, we present the original distributions. Next, we will apply standard transformations and compare the distributions of these engineered features.


```python
# Import helper functions from 'goose_helpers.py'
import goose_helpers as goose

# Display distributions using custom function
continuous_features = goose.plot_continuous_features(df)
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_0.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_1.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_2.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_3.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_4.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_5.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_39_6.png)
    


### Data Anomalies in 'BounceRates' and 'ExitRates'
The strip plot analysis revealed anomalies in the 'BounceRates' and 'ExitRates' features, indicating a mix of continuous values and binned categories. Solid-like lines and rounded numbers in value counts suggest the presence of categorical data. For instance, occurrences of exactly 20% without decimals occurring 491 times imply categorical rather than continuous data. Without clarification from the dataset's author, accurately categorizing these bins is challenging. Additionally, the presence of seemingly continuous "categories" like 0.066667 in 'ExitRates' adds complexity. These observations highlight potential irregularities, though definitive explanations await further information.


```python
df['BounceRates'].value_counts().sort_values(ascending=False).head(10)
```




    BounceRates
    0.000000    4768
    0.200000     491
    0.066667     117
    0.028571     102
    0.050000     101
    0.025000      91
    0.033333      87
    0.040000      86
    0.100000      84
    0.020000      82
    Name: count, dtype: int64




```python
df['ExitRates'].value_counts().sort_values(ascending=False).head(10)
```




    ExitRates
    0.200000    498
    0.100000    294
    0.050000    285
    0.033333    248
    0.066667    235
    0.025000    190
    0.040000    183
    0.016667    152
    0.020000    147
    0.022222    133
    Name: count, dtype: int64



### Exploring Feature Transformations
For this analysis, we will primarily focus on using logarithmic and square root transformations. While the Box-Cox and Yeo-Johnson transformations are powerful techniques for handling skewed data, they encountered issues with our dataset due to the presence of extremely large values and complex distributions.

Both the Box-Cox and Yeo-Johnson transformations involve raising values to a power determined by an estimated parameter (λ). In some cases, the transformed values may exceed the maximum representable value for the data type, resulting in overflow issues.

To avoid these problems, we will stick with logarithmic and square root transformations, which are less sensitive to extreme values. For more information on the Yeo-Johnson transformation and its comparison to the Box-Cox transformation, please refer to [this resource](https://feaz-book.com/numeric-yeojohnson).

Refer to the plot below for a feature-by-feature comparison, illustrating the distributions after applying robust scaling, logarithmic transformation (with robust scaling), and square root transformation (with robust scaling).


```python
from sklearn.preprocessing import RobustScaler, FunctionTransformer, PowerTransformer

#warnings.filterwarnings(category=RuntimeWarning, action='ignore')

# Create log transformer
log_transformer = FunctionTransformer(np.log1p)

# Create square root transformer
sqrt_transformer = FunctionTransformer(np.sqrt)

# Create a RobustScaler object
robust_scaler = RobustScaler()

# Create a figure with subplots
fig, axes = plt.subplots(len(continuous_features), 3, figsize=(20, 5 * len(continuous_features)))

df_temp = df.copy()

for i, feature in enumerate(continuous_features):
    # Original distribution
    df_temp[f'Scaled_{feature}'] = robust_scaler.fit_transform(df_temp[[feature]])
    sns.histplot(df_temp[f'Scaled_{feature}'], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'{feature} - Original (Scaled)')
    
    # Log transformation
    df_temp[f'Log_{feature}'] = log_transformer.transform(df_temp[[feature]])
    df_temp[f'Scaled_Log_{feature}'] = robust_scaler.fit_transform(df_temp[[f'Log_{feature}']])
    sns.histplot(df_temp[f'Scaled_Log_{feature}'], kde=True, ax=axes[i, 1])
    axes[i, 1].set_title(f'{feature} - Log Transformed')
    
    # Square root transformation
    df_temp[f'Sqrt_{feature}'] = sqrt_transformer.transform(df_temp[[feature]])
    df_temp[f'Scaled_Sqrt_{feature}'] = robust_scaler.fit_transform(df_temp[[f'Sqrt_{feature}']])
    sns.histplot(df_temp[f'Scaled_Sqrt_{feature}'], kde=True, ax=axes[i, 2])
    axes[i, 2].set_title(f'{feature} - Square Root Transformed')

plt.tight_layout()
plt.show()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_44_0.png)
    


#### Optimizing Features for Better Data Distribution

We've developed a targeted feature handling strategy based on thorough analysis of each feature's distribution, outliers, and characteristics. Here's a summary:

| Feature                 | Distribution   | Outliers      | Transformation | Scaler        |
|-------------------------|----------------|---------------|----------------|---------------|
| Administrative_Duration | Right-skewed   | Significant   | Logarithmic    | RobustScaler  |
| Informational_Duration  | Right-skewed   | Significant   | Logarithmic    | RobustScaler  |
| ProductRelated          | Right-skewed   | Significant   | None           | RobustScaler  |
| ProductRelated_Duration | Right-skewed   | Significant   | None           | RobustScaler  |
| BounceRates             | Right-skewed   | Significant   | Logarithmic    | RobustScaler  |
| ExitRates               | Right-skewed   | Significant   | Logarithmic    | RobustScaler  |
| PageValues              | Right-skewed   | Significant   | Logarithmic    | RobustScaler  |

By comparing the transformed features in the visual below with the originals, we can observe several notable improvements, despite the remaining challenges:

- Reduced skewness
- Revealed data clusters
- Improved outlier handling
- Better interpretability

Some previously unworkable features are now 'usable'. RobustScaler is the clear choice for handling our many outliers while preserving data structure.

These optimizations will lead to more accurate and meaningful results in our modeling and analysis.

Explore the detailed outcomes of our feature transformations in the visualization below!


```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set the color palette
sns.set_palette(sns.color_palette(["steelblue"]))

engineered_features = {
    'Administrative_Duration': 'Scaled_Log_Administrative_Duration',
    'Informational_Duration': 'Scaled_Log_Informational_Duration',
    'ProductRelated': 'Scaled_ProductRelated',
    'ProductRelated_Duration': 'Scaled_ProductRelated_Duration',
    'BounceRates': 'Scaled_Log_BounceRates',
    'ExitRates': 'Scaled_Log_ExitRates',
    'PageValues': 'Scaled_Log_PageValues'
}

for feature, engineered_feature in engineered_features.items():
    if len(df_temp[engineered_feature].unique()) > 25:  # Condition for continuous features
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        # Plots
        sns.histplot(data=df_temp[engineered_feature], ax=ax[0], kde=True, alpha=0.7)
        sns.stripplot(x=df_temp[engineered_feature], ax=ax[1], alpha=0.7)
        sns.boxplot(x=df_temp[engineered_feature], ax=ax[2])

        # Stripplot median and mean line
        median = df_temp[engineered_feature].median()
        mean = df_temp[engineered_feature].mean()
        ax[1].axvline(x=median, color='#6AA2D8', linestyle='-', linewidth=3, ymin=0.35, ymax=0.65, zorder=3)  # median
        ax[1].axvline(x=mean, color='#6AA2D8', linestyle='--', linewidth=3, ymin=0.35, ymax=0.65, zorder=3)   # mean

        # Title entire figure
        fig.suptitle(f"{feature} - {engineered_feature}", fontsize=18, y=1)

        # Hide ticks while keeping labels
        ax[0].tick_params(bottom=False)
        ax[1].tick_params(bottom=False)
        ax[2].tick_params(bottom=False)
        ax[0].tick_params(left=False)
        ax[1].tick_params(left=False)
        ax[2].tick_params(left=False)

        # Hide x and y labels
        ax[0].set(xlabel=None, ylabel=None)
        ax[1].set(xlabel=None, ylabel=None)
        ax[2].set(xlabel=None, ylabel=None)

        sns.despine()

# Delete df_temp to free memory
del df_temp
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_0.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_1.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_2.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_3.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_4.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_5.png)
    



    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_46_6.png)
    


### Categorical Features Frequencies
Several categorical features contain sparse categories, limiting the utility of these specific subcategories for predictions. Notably, the target variable and many features display significant imbalance, underscoring the need for strategic handling in modeling.

| Feature                 | Category Frequencies                                    | Sparse Subcategories    | Encoding Technique                    |
|-------------------------|---------------------------------------------------------|-------------------------|---------------------------------------|
| SpecialDay              | Highly imbalanced, majority have 0.0 special day value  | No                      | One-Hot Encoding or Ordinal Encoding  |
| Month                   | Relatively balanced, higher in Mar, May and Nov         | No                      | One-Hot Encoding or Ordinal Encoding  |
| VisitorType             | Highly imbalanced, majority are Returning_Visitor       | No                      | One-Hot Encoding                      |
| Administrative          | Imbalanced, majority in 0                               | Yes                     | One-Hot Encoding                      |
| Informational           | Highly imbalanced, majority have 0 informational pages  | Yes                     | One-Hot Encoding                      |
| OperatingSystems        | Imbalanced, majority use OS 2                           | Yes                     | One-Hot Encoding                      |
| Browser                 | Imbalanced, majority use Browser 2 and 1                | Yes                     | One-Hot Encoding                      |
| Region                  | Imbalanced, majority from Region 1                      | No                      | One-Hot Encoding                      |
| TrafficType             | Imbalanced, majority are TrafficType 1, 2 or 3          | Yes                     | One-Hot Encoding                      |
| Weekend                 | Imbalanced, majority are not during weekend             | No                      | One-Hot Encoding                      |
| Revenue                 | Imbalanced, majority did not generate revenue           | No                      | Target variable                       |


```python
categorical_features = goose.plot_categorical_features(df)
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_48_0.png)
    


### Correlation Analysis
#### Correlation Matrix
- Independent variable `PageValues`is moderately correlated with dependent variable `Revenue`.
- The high positive correlation between the following features suggests redundancy, indicating potential collinearity:
    - `ProductRelated` and `ProductRelated_Duration`
    - `ExitRates` and `BounceRates`
- For `Month`, `SpecialDay`, and `Weekend`, one would presume a correlation. The absence of correlation suggests a missed opportunity to leverage special holidays, such as Black Friday and Christmas, by offering discounts and other promotions.


##### Regarding 'PageValues'
- Kaggle information states: *"The average value of the page averaged over the value of the target page and/or the completion of an eCommerce transaction."*
    - It's unclear whether this pertains to historic or current session data, which could introduce data leakage issues.
However, the "Uppdragsbeskrivning" clearly states: *"Till din hjälp har du historisk data..."*, confirming that it's historic data, thus justifying the retention of 'PageValues'.



```python
# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True).round(3), mask=np.triu(df.corr(numeric_only=True)), annot=True, cmap="coolwarm_r", cbar=None, vmin=-1, vmax=+1, center=0, linewidths=.5)
plt.title('Correlation Matrix')
plt.tick_params(axis='both', length=0)
plt.show()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_50_0.png)
    


#### Feature Correlation and Conversion Analysis
- The strong correlation between `ExitRates` and `BounceRates`, highlighted in the correlation matrix, persists in this analysis.
- Additionally, there's a noticeable trend: as `PageValues` increase, the proportion of customers bouncing decreases. This suggests successful conversion rate optimization (CRO) efforts, positively impacting user engagement and site performance. In other words, as users engage more with the content by clicking around, their likelihood of bouncing decreases.
- However, a cluster of high `PageValues` coincides with lower `ProductRelated` counts, suggesting potential for improving conversion rates. This trend implies that as users explore more product pages, they tend to move away from those driving revenue. Enhancing user engagement to align with higher `PageValues` could involve transitioning to a scrolling-based interface or making better recommendations. However, further investigation is warranted to fully understand this trend's implications for optimizing conversion rates.


```python
# Set the color palette for two categories represented in 'Churn'
sns.set_palette(sns.color_palette(["#4796C8", "#FF5733"]))

# Create the pairplot with 'Churn' as the hue
pairplot = sns.pairplot(data=df[continuous_features + ['Revenue']], corner=True, hue='Revenue', plot_kws={'alpha': 0.5})

# Loop through axes; remove the ticks
for ax in pairplot.axes.flatten():
    if ax:
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_52_0.png)
    


### Conversions by Visitor Type
New visitors convert at 25%, higher than the 14% for returning visitors, indicating room for improving engagement and conversion among the latter. While returning customers may not convert at every visit, we acknowledge that their loyalty drives long-term revenue growth. Implementing targeted campaigns and offers could boost engagement, as suggested earlier. However, previous analysis suggests ineffective implementation of these strategies.


```python
# Group by 'VisitorType' and calculate the count of 'Revenue' for each category
revenue_counts = df.groupby(['VisitorType', 'Revenue']).size().unstack()

# Plotting the stacked bar plot
ax = revenue_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette("Paired")[1:3], width=0.7)
plt.title('Distribution of Revenue by Visitor Type')
plt.xlabel('Visitor Type')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Revenue', labels=['No Revenue', 'Revenue'])

# Annotate with percentage of revenue
total_per_category = revenue_counts.sum(axis=1)
for i in range(len(revenue_counts)):
    total = total_per_category[i]
    count = revenue_counts.iloc[i, 1]  # Revenue count
    percentage = count / total * 100 if total > 0 else 0
    ax.annotate(f'{percentage:.1f}%', xy=(i, total), xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', color='black', fontsize=10)
sns.despine()


```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_54_0.png)
    


### Special Days and Conversion Trends
`SpecialDays` indicates proximity with notable dates such as holidays, Mother's Day and similar occasions. Interestingly, a significant portion of conversions happens exactly on these special days, as "0.0" in the graph illustrates this immediacy. Concurrently, the majority of visits also transpire on these occasions, underscoring a pattern where customers prefer to shop on the day itself rather than beforehand. This insight suggests an area ripe for strategy refinement.

Many retailers use special days for promotions, often beginning well in advance. Despite "Black Friday" marking a peak in sales, it's the preceding "Black Week" and subsequent "Cyber Monday" that have become increasingly profitable. Our analysis suggests a substantial opportunity for boosting pre-event engagement. By tailoring marketing strategies to extend offers before these key dates, businesses stand to gain considerably.


```python
# Group by 'SpecialDay' and calculate the count of 'Revenue' for each category
revenue_counts = df.groupby(['SpecialDay', 'Revenue']).size().unstack()

# Plotting the stacked bar plot
ax = revenue_counts.plot(kind='barh', stacked=True, figsize=(10, 6), color=sns.color_palette("Paired")[1:3], width=0.7)
plt.title('Distribution of Revenue by Special Day')
plt.xlabel('Special Day')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Revenue', labels=['No Revenue', 'Revenue'])
sns.despine()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_56_0.png)
    


### Revenue by Weekend
There are three times as many visitors during weekdays compared to weekends; however, the conversion rate is slightly higher during weekends.


```python
# Group by 'Weekend' and calculate the count of 'Revenue' for each category
revenue_counts = df.groupby(['Weekend', 'Revenue']).size().unstack()

# Plotting the stacked bar plot
ax = revenue_counts.plot(kind='bar', stacked=True, figsize=(8, 6), color=sns.color_palette("Paired")[1:3], width=0.7)
plt.title('Distribution of Revenue by Weekend')
plt.xlabel('Weekend')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.legend(title='Revenue', labels=['No Revenue', 'Revenue'])

# Calculate the percentage of revenue for each category
total_per_category = revenue_counts.sum(axis=1)
for i in range(len(revenue_counts)):
    total = total_per_category[i]
    for j in range(len(revenue_counts.columns)):
        count = revenue_counts.iloc[i, j]
        percentage = count / total * 100 if total > 0 else 0
        ax.annotate(f'{percentage:.1f}%', xy=(i, revenue_counts.iloc[i, :j].sum() + count / 2),
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', color='white', fontsize=10)

sns.despine()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_58_0.png)
    


## Metric Selection for Understanding Converting Customers

### Goal
The primary objective, as stated in the assignment, is to understand what constitutes a good user experience by identifying the customers who actually convert and make purchases. The aim is to analyze historical data to profile the characteristics and behaviors of converting customers.

In short, the task at hand is to build a model that accurately identifies the converting customers based on historical data.

### Metric Considerations

Given the goal of understanding the profile of converting customers, the following metrics are crucial:

1. **Precision for Class 1 (Buyers)**: Optimizing precision ensures that the model correctly identifies the customers who are most likely to convert, minimizing false positives. This helps in building an accurate profile of the converting customers.

2. **Recall for Class 1 (Buyers)**: While precision is the primary focus, recall is still important to ensure that the model captures a sufficient number of converting customers for analysis. A balance between precision and recall is necessary to obtain a comprehensive understanding of the converting customer profile.

Accuracy, while important, may not be the most critical metric in this scenario, especially since the classes are imbalanced.

### Prioritizing Precision for Customer Profiling

In the context of understanding the converting customer profile, precision takes precedence over other metrics. By prioritizing precision, the model focuses on correctly identifying the customers who are most likely to convert, minimizing the inclusion of non-converting customers in the profile.

High precision ensures that the insights gained from the analysis are based on a reliable and accurate understanding of the converting customer profile. This allows the company to:

1. Identify the key characteristics, behaviors, and factors that differentiate converting customers from non-converting ones.
2. Gain a clear understanding of what constitutes a good user experience for their specific customer base.
3. Make data-driven decisions on how to optimize their website, products, and marketing strategies to cater to the preferences and needs of the converting customers.

### Balancing Precision and Recall

While precision is the primary focus, it's important to strike a balance with recall to ensure that the model captures a sufficient number of converting customers for analysis. Monitoring the recall-score helps in achieving this balance.

The company should evaluate the trade-off between precision and recall and determine an acceptable level of false positives that still allows for meaningful insights. Iterating and refining the model based on the results and business objectives can help in finding the right balance.

### Recommendation

The recommendation is to prioritize precision when building the model to understand the converting customer profile. By focusing on precision, the company can gain accurate insights into the characteristics and behaviors of customers who are most likely to convert.

However, it's important to monitor the recall-score and ensure that the model also captures a sufficient number of converting customers for a comprehensive analysis. Striking the right balance between precision and recall will enable the company to make informed decisions and develop strategies that cater to the needs and preferences of their high-value customers, ultimately improving the overall user experience and driving more conversions.

## Data Split
To prepare for modeling, we'll divide the dataset into distinct subsets for training, validation, and testing purposes.


```python
# Splits data into train, validation, test and combined train/validation sets for modeling.
X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val = goose.split_data(df, "Revenue", verbose=True)
```

    Data split complete
    
    Training set: (7385, 17) (7385,)
    Validation set: (1578, 17) (1578,)
    Test set: (1582, 17) (1582,)
    Combined Training and Validation set: (8963, 17) (8963,)
    

## Establishing a Baseline Model

To set a benchmark for future model comparisons, we'll begin with Logistic Regression—a simple and interpretable model. Here's why:

- **Quick Implementation**: It's fast to train, providing an immediate benchmark for performance.
- **Clarity of Insights**: The model's interpretability helps us understand which features are most influential in predicting outcomes.
- **Benchmark for Comparison**: This initial model will act as a reference, helping to quantify the benefits of more advanced models explored later.

Starting with Logistic Regression ensures any additional model complexity is warranted by significant performance gains.

### Baseline Model Architecture
When developing our baseline model, we aim to simplify the process somewhat. We'll avoid employing the entire array of techniques such as sampling methods or imputing missing values. However, we will allow for the application of transformations to ensure more manageable distributions and to address outliers to a certain extent. This strategy allows us to assess the effects of feature transformations before advancing to model selection, intricate hyperparameter tuning, and experimenting with the aforementioned preprocessing techniques.

### Baseline Model Performance

Implementing feature transformations have resulted in a markedly improved model on the validation data, boosting the recall score by 22 percentage points from 0.36 to 0.58, while precision remains identical at 0.71. With some quick tuning inherent to Logistic Regression, we managed to increase the recall to 0.59. Hence, we will define our baseline as:

- Precision-score: 0.71
- Recall-score: 0.59

Given our goal of precisely identifying and profiling converting customers, our primary emphasis will be on enhancing precision. Subsequently, we will work to improve recall, ensuring we capture a broad enough group of actual converters. This strategy is designed to refine our dataset for a more accurate analysis of customer conversion patterns.

### Baseline Model Hypothesis

- **Null Hypothesis (H0):** The baseline model (Logistic Regression) performs equally well compared to the Enhanced Boosting Model (EBM) and LightGBM.
- **Alternative Hypothesis (H1):** There is a significant difference in performance between the baseline model and at least one of the alternative models.

#### Logistic Regression with Basic Scalers
The key distinction here is that we are not employing log transformations; instead, we are applying a robust scaler to all continuous features.


```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PowerTransformer, FunctionTransformer

# Instantiate scaler
scaler = RobustScaler()

# Numerical transformers
robust_transformer = make_pipeline(scaler)
log_transformer = make_pipeline(FunctionTransformer(np.log1p), scaler)
sqrt_transformer = make_pipeline(FunctionTransformer(np.sqrt), scaler)

# Categorical transformers
onehot_transformer = make_pipeline(OneHotEncoder(drop='if_binary', handle_unknown='ignore', dtype=int))

# Feature specific transformers
transformers = [
    (robust_transformer, ['Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration', 'Informational_Duration', 'BounceRates', 'ExitRates', 'PageValues']),
    (log_transformer, []),
    (sqrt_transformer, []),
    (onehot_transformer, ['VisitorType', 'Informational', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend', 'Month', 'SpecialDay']),
]

# Combine transformers
basic_preprocessor = make_column_transformer(*transformers)

# Set the seed for reproducibility
random_state = 42

# Instantiate estimator
log_model = LogisticRegression(random_state=random_state)

# Pipeline creation
log_pipe_basic = make_pipeline(basic_preprocessor, log_model)

# Train model
log_pipe_basic.fit(X_train, y_train)

# Display report
goose.display_classification_report(log_pipe_basic, X_train, y_train, X_val, y_val, data_to_display='both')
goose.display_roc_curve(log_pipe_basic, X_train, y_train, X_val, y_val, data_to_display='validation')
```

    Classification Report (Training):
                  precision    recall  f1-score   support
    
               0       0.90      0.98      0.94      6234
               1       0.76      0.40      0.53      1151
    
        accuracy                           0.89      7385
       macro avg       0.83      0.69      0.73      7385
    weighted avg       0.88      0.89      0.87      7385
    
    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.89      0.97      0.93      1332
               1       0.71      0.36      0.48       246
    
        accuracy                           0.88      1578
       macro avg       0.80      0.67      0.70      1578
    weighted avg       0.86      0.88      0.86      1578
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_64_1.png)
    


#### Logistic Regression with Feature Transformations
The updated baseline model, leveraging feature transformations, demonstrates a consistent precision of 0.71 and an enhanced recall of 0.59 on validation data.


```python
# Feature specific transformers
transformers = [
    (robust_transformer, ['ProductRelated', 'ProductRelated_Duration']),
    (log_transformer, ['Administrative_Duration', 'Informational_Duration', 'BounceRates', 'ExitRates', 'PageValues']),
    (sqrt_transformer, []),
    (onehot_transformer, ['VisitorType', 'Informational', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend', 'Month', 'SpecialDay']),
]

# Combine transformers
preprocessor = make_column_transformer(*transformers)

# Instantiate estimator
log_model = LogisticRegression(solver='saga', penalty='l1', C=0.15, random_state=random_state)
#log_model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.8, C=0.15, random_state=random_state)

# Pipeline creation
log_pipe = make_pipeline(preprocessor, log_model)

# Train model
log_pipe.fit(X_train, y_train)

# Display report and roc-curve
goose.display_classification_report(log_pipe, X_train, y_train, X_val, y_val, data_to_display='both')
goose.display_roc_curve(log_pipe, X_train, y_train, X_val, y_val, data_to_display='validation')
```

    Classification Report (Training):
                  precision    recall  f1-score   support
    
               0       0.92      0.96      0.94      6234
               1       0.71      0.54      0.62      1151
    
        accuracy                           0.89      7385
       macro avg       0.82      0.75      0.78      7385
    weighted avg       0.89      0.89      0.89      7385
    
    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.93      0.96      0.94      1332
               1       0.71      0.59      0.64       246
    
        accuracy                           0.90      1578
       macro avg       0.82      0.77      0.79      1578
    weighted avg       0.89      0.90      0.89      1578
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_66_1.png)
    


## Model Selection

After applying cross-validation across a carefully selected list of classifiers and analyzing the resulting F1-scores, we deliberated on which models optimally balance explainability and performance. This process led us to select two models for further hyperparameter tuning. Below are their introductions:

**Explainable Boosting Classifier (EBM)**  
EBM stands out as a glass-box model, offering a unique blend of high performance, akin to similar boosting and bagging techniques, while maintaining unparalleled interpretability and explainability. Although it didn't outperform the top contender from the model selection phase, the LightGBM Classifier, EBM emerges as a prime candidate for developing an initial model. This model can be presented to stakeholders with clarity and detailed explanation. Furthermore, EBM's comprehensive toolkit aids in providing deeper insights into customer behavior and the driving factors behind conversions, facilitating a better understanding for the board.

**LightGBM Classifier (LGBM)**  
LGBM emerged as the top-performing model during our model selection phase. While our primary objective revolves around comprehending the essence of a superior customer experience and the factors influencing their decisions, LGBM presents significant advantages. It offers a robust framework for exploring alternative solutions for future deployments, particularly when enhancing performance—and consequently boosting revenue—becomes paramount. This makes LGBM a valuable asset in our toolkit, balancing between understanding customer behavior and achieving business objectives.


```python
# Import necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import get_scorer

# Model libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

# Initialize preprocessor with feature-specific transformations
preprocessor = make_column_transformer(
    (robust_transformer, ['ProductRelated', 'ProductRelated_Duration']),
    (log_transformer, ['Administrative_Duration', 'Informational_Duration', 'BounceRates', 'ExitRates', 'PageValues']),
    (sqrt_transformer, []),
    (onehot_transformer, ['VisitorType', 'Informational', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend', 'Month', 'SpecialDay']),
)

# Combine preprocessing with classifier
classifiers = [LogisticRegression(random_state=random_state),
               RandomForestClassifier(random_state=random_state),
               SVC(random_state=random_state),
               KNeighborsClassifier(),
               GradientBoostingClassifier(random_state=random_state),
               XGBClassifier(random_state=random_state),
               LGBMClassifier(random_state=random_state, verbose=-1),
               CatBoostClassifier(allow_writing_files=False, logging_level='Silent', verbose=None, silent=None, random_seed=random_state),
               ExplainableBoostingClassifier(random_state=random_state)
]

# Define the number of folds
num_folds = 5

# Define scoring method
scoring = 'f1'

# Get the scoring function based on the scoring variable
scorer = get_scorer(scoring)

# Create a table for the results
final_results_table = PrettyTable()

# Create lists to store classifier names and mean scores (for plot, next code block)
classifier_names = []
mean_scores = []

# Dynamically create field names based on the number of folds
field_names = ["Classifier"] + [f"F{i+1}" for i in range(num_folds)] + ["Std", "Avg Score"]
final_results_table.field_names = field_names

# Loop through each classifier
for classifier in tqdm(classifiers, desc="Processing classifiers", dynamic_ncols=True):
    # Create a pipeline with preprocessing and classifier
    pipe = make_pipeline(preprocessor, classifier)

    # Perform cross-validation
    scores = cross_val_score(pipe, X_train_val, y_train_val, cv=num_folds, scoring=scoring)
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    # Add classifier name and mean score to their respective lists
    classifier_names.append(classifier.__class__.__name__)
    mean_scores.append(mean_score)

    # Format the results
    scores_formatted = ["{:.3f}".format(score) for score in scores]
    mean_score_formatted = "{:.3f}".format(mean_score)
    std_dev_formatted = "{:.3f}".format(std_dev)

    # Add the results to the table
    final_results_table.add_row([classifier.__class__.__name__] + scores_formatted + [std_dev_formatted, mean_score_formatted])

# Sort results by mean score in descending order
final_results_table.sortby = "Avg Score"
final_results_table.reversesort = True

# Print the results
print(final_results_table)
```

    Processing classifiers:  11%|█         | 1/9 [00:00<00:03,  2.36it/s]

    Processing classifiers: 100%|██████████| 9/9 [01:14<00:00,  8.30s/it]

    +-------------------------------+-------+-------+-------+-------+-------+-------+-----------+
    |           Classifier          |   F1  |   F2  |   F3  |   F4  |   F5  |  Std  | Avg Score |
    +-------------------------------+-------+-------+-------+-------+-------+-------+-----------+
    |         LGBMClassifier        | 0.654 | 0.644 | 0.633 | 0.619 | 0.669 | 0.017 |   0.644   |
    |   GradientBoostingClassifier  | 0.661 | 0.641 | 0.620 | 0.623 | 0.650 | 0.016 |   0.639   |
    | ExplainableBoostingClassifier | 0.659 | 0.644 | 0.592 | 0.612 | 0.667 | 0.029 |   0.635   |
    |       CatBoostClassifier      | 0.659 | 0.635 | 0.617 | 0.613 | 0.649 | 0.018 |   0.635   |
    |              SVC              | 0.635 | 0.646 | 0.614 | 0.604 | 0.654 | 0.019 |   0.631   |
    |         XGBClassifier         | 0.621 | 0.624 | 0.604 | 0.607 | 0.625 | 0.009 |   0.616   |
    |       LogisticRegression      | 0.650 | 0.609 | 0.596 | 0.561 | 0.641 | 0.032 |   0.611   |
    |     RandomForestClassifier    | 0.577 | 0.620 | 0.578 | 0.566 | 0.614 | 0.022 |   0.591   |
    |      KNeighborsClassifier     | 0.592 | 0.584 | 0.579 | 0.543 | 0.616 | 0.024 |   0.583   |
    +-------------------------------+-------+-------+-------+-------+-------+-------+-----------+
    

    
    

## Model Tuning

### Explainable Boosting Classifier (EBM)
Following meticulous tuning, we've increased the generalizability of the model while also improving the raw performance and enhancing both precision and recall by two percentage points over the model's default hyperparameters, leading to the ensuing scores on the validation dataset:
- Precision: 0.69
- Recall: 0.61
- AUC: 93

While EBM shows a minor dip in precision compared to the baseline model, there's an uptick in recall. Although any reduction in our key metric isn't ideal, it's important to note that we haven't yet explored sampling techniques to address the target variable's imbalance or implemented imputation for missing values. These forthcoming strategies may further refine our model's performance, offering room for optimization in both precision and recall.


```python
# Hyperparameters for EBMClassifier
param_grid = {
    'explainableboostingclassifier__cyclic_progress': [0.7], # ✅
    'explainableboostingclassifier__early_stopping_rounds': [50], # ✅
    'explainableboostingclassifier__greedy_ratio': [1.5], # ✅
    'explainableboostingclassifier__inner_bags': [50], # [0, 50] 50x 🔥
    'explainableboostingclassifier__interaction_smoothing_rounds': [0], # ✅
    'explainableboostingclassifier__interactions': [0.95],  # ✅
    'explainableboostingclassifier__learning_rate': [0.01], # ✅
    'explainableboostingclassifier__max_bins': [5000],  # ✅
    'explainableboostingclassifier__max_interaction_bins': [32],  # ✅
    'explainableboostingclassifier__max_leaves': [4], # ✅
    'explainableboostingclassifier__max_rounds': [1000000000], # ✅
    'explainableboostingclassifier__min_hessian': [0.0001], # ✅
    'explainableboostingclassifier__min_samples_leaf': [2], # ✅
    'explainableboostingclassifier__outer_bags': [14], # <50 ✅
    'explainableboostingclassifier__smoothing_rounds': [75], # ✅
    'explainableboostingclassifier__validation_size': [0.2], # ✅
}

# Initialize preprocessor with feature-specific transformations
preprocessor = make_column_transformer(
    (robust_transformer, ['ProductRelated', 'ProductRelated_Duration']),
    (log_transformer, ['Administrative_Duration', 'Informational_Duration', 'BounceRates', 'ExitRates', 'PageValues']),
    (sqrt_transformer, []),
    (onehot_transformer, ['VisitorType', 'Informational', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend', 'Month', 'SpecialDay']),
)

ebm_model = ExplainableBoostingClassifier(random_state=random_state)

pipe = make_pipeline(preprocessor, ebm_model)

grid_ebm = GridSearchCV(estimator=pipe,
                        param_grid=param_grid,
                        cv=3,
                        scoring='f1',
                        n_jobs=-1)

grid_ebm.fit(X_train, y_train)

goose.display_search_results(grid_ebm, X_train, y_train, X_val, y_val)
goose.display_roc_curve(grid_ebm, X_train, y_train, X_val, y_val, data_to_display='validation')

```

    +--------------------------------------------------------------------------+
    |               ExplainableBoostingClassifier Tuning Results               |
    +-------------------------------------------------------------+------------+
    |                          Parameter                          |   Value    |
    +-------------------------------------------------------------+------------+
    |        explainableboostingclassifier__cyclic_progress       |    0.7     |
    |     explainableboostingclassifier__early_stopping_rounds    |     50     |
    |         explainableboostingclassifier__greedy_ratio         |    1.5     |
    |          explainableboostingclassifier__inner_bags          |     50     |
    | explainableboostingclassifier__interaction_smoothing_rounds |     0      |
    |         explainableboostingclassifier__interactions         |    0.95    |
    |         explainableboostingclassifier__learning_rate        |    0.01    |
    |           explainableboostingclassifier__max_bins           |    5000    |
    |     explainableboostingclassifier__max_interaction_bins     |     32     |
    |          explainableboostingclassifier__max_leaves          |     4      |
    |          explainableboostingclassifier__max_rounds          | 1000000000 |
    |          explainableboostingclassifier__min_hessian         |   0.0001   |
    |       explainableboostingclassifier__min_samples_leaf       |     2      |
    |          explainableboostingclassifier__outer_bags          |     14     |
    |       explainableboostingclassifier__smoothing_rounds       |     75     |
    |        explainableboostingclassifier__validation_size       |    0.2     |
    +-------------------------------------------------------------+------------+
    Classification Report - Training Set:
                  precision    recall  f1-score   support
    
               0       0.93      0.97      0.95      6234
               1       0.77      0.64      0.70      1151
    
        accuracy                           0.91      7385
       macro avg       0.85      0.80      0.82      7385
    weighted avg       0.91      0.91      0.91      7385
    
    Classification Report - Validation Set:
                  precision    recall  f1-score   support
    
               0       0.93      0.95      0.94      1332
               1       0.69      0.61      0.65       246
    
        accuracy                           0.90      1578
       macro avg       0.81      0.78      0.79      1578
    weighted avg       0.89      0.90      0.89      1578
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_71_1.png)
    


```python
Round 7: 4 min
+--------------------------------------------------------------------------+
|               ExplainableBoostingClassifier Tuning Results               |
+-------------------------------------------------------------+------------+
|                          Parameter                          |   Value    |
+-------------------------------------------------------------+------------+
|        explainableboostingclassifier__cyclic_progress       |    0.7     |
|     explainableboostingclassifier__early_stopping_rounds    |     50     |
|         explainableboostingclassifier__greedy_ratio         |    1.5     |
|          explainableboostingclassifier__inner_bags          |     50     |
| explainableboostingclassifier__interaction_smoothing_rounds |     0      |
|         explainableboostingclassifier__interactions         |    0.95    |
|         explainableboostingclassifier__learning_rate        |    0.01    |
|           explainableboostingclassifier__max_bins           |    5000    |
|     explainableboostingclassifier__max_interaction_bins     |     32     |
|          explainableboostingclassifier__max_leaves          |     4      |
|          explainableboostingclassifier__max_rounds          | 1000000000 |
|          explainableboostingclassifier__min_hessian         |   0.0001   |
|       explainableboostingclassifier__min_samples_leaf       |     2      |
|          explainableboostingclassifier__outer_bags          |     14     |
|       explainableboostingclassifier__smoothing_rounds       |     75     |
|        explainableboostingclassifier__validation_size       |    0.2     |
+-------------------------------------------------------------+------------+
Classification Report - Training Set:
              precision    recall  f1-score   support

           0       0.93      0.97      0.95      6234
           1       0.77      0.64      0.70      1151

    accuracy                           0.91      7385
   macro avg       0.85      0.80      0.82      7385
weighted avg       0.91      0.91      0.91      7385

Classification Report - Validation Set:
              precision    recall  f1-score   support

           0       0.93      0.95      0.94      1332
           1       0.69      0.61      0.65       246

    accuracy                           0.90      1578
   macro avg       0.81      0.78      0.79      1578
weighted avg       0.89      0.90      0.89      1578


Round 1: 10 hrs
param_grid = {
    'explainableboostingclassifier__cyclic_progress': [0.25, 0.5, 1.0],
    'explainableboostingclassifier__early_stopping_rounds': [50],
    'explainableboostingclassifier__greedy_ratio': [1.25, 1.5, 1.75],
    'explainableboostingclassifier__inner_bags': [0], # [0, 50]
    'explainableboostingclassifier__interaction_smoothing_rounds': [0], # 50 default
    'explainableboostingclassifier__interactions': [0.95],  # [0.5, 0.75, 0.95]
    'explainableboostingclassifier__learning_rate': [0.005, 0.01, 0.02], 
    'explainableboostingclassifier__max_bins': [1024],  # [1024, 4096, 16384]
    'explainableboostingclassifier__max_interaction_bins': [32],  # [16, 32, 64]
    'explainableboostingclassifier__max_leaves': [3, 4], # [3, 4] 
    'explainableboostingclassifier__max_rounds': [25000], # [1000000000] ideal
    'explainableboostingclassifier__min_hessian': [0.0001], # [1.0, 0.01, 0.0001, 0.000001]
    'explainableboostingclassifier__min_samples_leaf': [2, 3, 4], # [2, 3, 4] 
    'explainableboostingclassifier__outer_bags': [14], # [14, 20, 30] <50 
    'explainableboostingclassifier__smoothing_rounds': [100, 200, 300], 
    'explainableboostingclassifier__validation_size': [0.1, 0.15, 0.2],      
}
+----------------------------------------------------------------------+
|             ExplainableBoostingClassifier Tuning Results             |
+-------------------------------------------------------------+--------+
|                          Parameter                          | Value  |
+-------------------------------------------------------------+--------+
|        explainableboostingclassifier__cyclic_progress       |  1.0   |
|     explainableboostingclassifier__early_stopping_rounds    |   50   |
|         explainableboostingclassifier__greedy_ratio         |  1.5   |
|          explainableboostingclassifier__inner_bags          |   0    |
| explainableboostingclassifier__interaction_smoothing_rounds |   0    |
|         explainableboostingclassifier__interactions         |  0.95  |
|         explainableboostingclassifier__learning_rate        |  0.01  |
|           explainableboostingclassifier__max_bins           |  1024  |
|     explainableboostingclassifier__max_interaction_bins     |   32   |
|          explainableboostingclassifier__max_leaves          |   3    |
|          explainableboostingclassifier__max_rounds          | 25000  |
|          explainableboostingclassifier__min_hessian         | 0.0001 |
|       explainableboostingclassifier__min_samples_leaf       |   2    |
|          explainableboostingclassifier__outer_bags          |   14   |
|       explainableboostingclassifier__smoothing_rounds       |  100   |
|        explainableboostingclassifier__validation_size       |  0.2   |
+-------------------------------------------------------------+--------+
Classification Report - Training Set:
              precision    recall  f1-score   support

           0       0.94      0.97      0.95      6234
           1       0.79      0.65      0.71      1151

    accuracy                           0.92      7385
   macro avg       0.86      0.81      0.83      7385
weighted avg       0.91      0.92      0.91      7385

Classification Report - Validation Set:
              precision    recall  f1-score   support

           0       0.92      0.95      0.94      1332
           1       0.68      0.58      0.63       246

    accuracy                           0.89      1578
   macro avg       0.80      0.77      0.78      1578
weighted avg       0.89      0.89      0.89      1578
```

#### Re-train with Best Parameters
We refit the model using the best parameters identified from grid-search and incorporated these settings into a new pipeline, equipped for future tasks like imputation.


```python
# Re-train the tuned model with the optimal parameters
ebm_params = {k.replace('explainableboostingclassifier__', ''): v 
              for k, v in grid_ebm.best_params_.items()}
ebm_model = ExplainableBoostingClassifier(**ebm_params, random_state=random_state)
ebm_pipe = make_pipeline(preprocessor, ebm_model)
ebm_pipe.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;pipeline-1&#x27;,
                                                  Pipeline(steps=[(&#x27;robustscaler&#x27;,
                                                                   RobustScaler())]),
                                                  [&#x27;ProductRelated&#x27;,
                                                   &#x27;ProductRelated_Duration&#x27;]),
                                                 (&#x27;pipeline-2&#x27;,
                                                  Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                                  (&#x27;robustscaler&#x27;,
                                                                   RobustScaler())]),
                                                  [&#x27;Administrative_Duration&#x27;,
                                                   &#x27;Informational_...
                                                                                 handle_unknown=&#x27;ignore&#x27;))]),
                                                  [&#x27;VisitorType&#x27;,
                                                   &#x27;Informational&#x27;,
                                                   &#x27;OperatingSystems&#x27;,
                                                   &#x27;Browser&#x27;, &#x27;Region&#x27;,
                                                   &#x27;TrafficType&#x27;, &#x27;Weekend&#x27;,
                                                   &#x27;Month&#x27;, &#x27;SpecialDay&#x27;])])),
                (&#x27;explainableboostingclassifier&#x27;,
                 ExplainableBoostingClassifier(cyclic_progress=0.7,
                                               inner_bags=50,
                                               interaction_smoothing_rounds=0,
                                               max_bins=5000, max_leaves=4,
                                               max_rounds=1000000000,
                                               smoothing_rounds=75,
                                               validation_size=0.2))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;pipeline-1&#x27;,
                                                  Pipeline(steps=[(&#x27;robustscaler&#x27;,
                                                                   RobustScaler())]),
                                                  [&#x27;ProductRelated&#x27;,
                                                   &#x27;ProductRelated_Duration&#x27;]),
                                                 (&#x27;pipeline-2&#x27;,
                                                  Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                                  (&#x27;robustscaler&#x27;,
                                                                   RobustScaler())]),
                                                  [&#x27;Administrative_Duration&#x27;,
                                                   &#x27;Informational_...
                                                                                 handle_unknown=&#x27;ignore&#x27;))]),
                                                  [&#x27;VisitorType&#x27;,
                                                   &#x27;Informational&#x27;,
                                                   &#x27;OperatingSystems&#x27;,
                                                   &#x27;Browser&#x27;, &#x27;Region&#x27;,
                                                   &#x27;TrafficType&#x27;, &#x27;Weekend&#x27;,
                                                   &#x27;Month&#x27;, &#x27;SpecialDay&#x27;])])),
                (&#x27;explainableboostingclassifier&#x27;,
                 ExplainableBoostingClassifier(cyclic_progress=0.7,
                                               inner_bags=50,
                                               interaction_smoothing_rounds=0,
                                               max_bins=5000, max_leaves=4,
                                               max_rounds=1000000000,
                                               smoothing_rounds=75,
                                               validation_size=0.2))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;columntransformer: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for columntransformer: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;pipeline-1&#x27;,
                                 Pipeline(steps=[(&#x27;robustscaler&#x27;,
                                                  RobustScaler())]),
                                 [&#x27;ProductRelated&#x27;, &#x27;ProductRelated_Duration&#x27;]),
                                (&#x27;pipeline-2&#x27;,
                                 Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)),
                                                 (&#x27;robustscaler&#x27;,
                                                  RobustScaler())]),
                                 [&#x27;Administrative_Duration&#x27;,
                                  &#x27;Informational_Duration&#x27;, &#x27;BounceRates&#x27;,
                                  &#x27;ExitRates&#x27;, &#x27;...
                                 Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(func=&lt;ufunc &#x27;sqrt&#x27;&gt;)),
                                                 (&#x27;robustscaler&#x27;,
                                                  RobustScaler())]),
                                 []),
                                (&#x27;pipeline-4&#x27;,
                                 Pipeline(steps=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(drop=&#x27;if_binary&#x27;,
                                                                dtype=&lt;class &#x27;int&#x27;&gt;,
                                                                handle_unknown=&#x27;ignore&#x27;))]),
                                 [&#x27;VisitorType&#x27;, &#x27;Informational&#x27;,
                                  &#x27;OperatingSystems&#x27;, &#x27;Browser&#x27;, &#x27;Region&#x27;,
                                  &#x27;TrafficType&#x27;, &#x27;Weekend&#x27;, &#x27;Month&#x27;,
                                  &#x27;SpecialDay&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">pipeline-1</label><div class="sk-toggleable__content fitted"><pre>[&#x27;ProductRelated&#x27;, &#x27;ProductRelated_Duration&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RobustScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>RobustScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">pipeline-2</label><div class="sk-toggleable__content fitted"><pre>[&#x27;Administrative_Duration&#x27;, &#x27;Informational_Duration&#x27;, &#x27;BounceRates&#x27;, &#x27;ExitRates&#x27;, &#x27;PageValues&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;FunctionTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RobustScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>RobustScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">pipeline-3</label><div class="sk-toggleable__content fitted"><pre>[]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;FunctionTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(func=&lt;ufunc &#x27;sqrt&#x27;&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RobustScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>RobustScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">pipeline-4</label><div class="sk-toggleable__content fitted"><pre>[&#x27;VisitorType&#x27;, &#x27;Informational&#x27;, &#x27;OperatingSystems&#x27;, &#x27;Browser&#x27;, &#x27;Region&#x27;, &#x27;TrafficType&#x27;, &#x27;Weekend&#x27;, &#x27;Month&#x27;, &#x27;SpecialDay&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;if_binary&#x27;, dtype=&lt;class &#x27;int&#x27;&gt;, handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">ExplainableBoostingClassifier</label><div class="sk-toggleable__content fitted"><pre>ExplainableBoostingClassifier(cyclic_progress=0.7, inner_bags=50,
                              interaction_smoothing_rounds=0, max_bins=5000,
                              max_leaves=4, max_rounds=1000000000,
                              smoothing_rounds=75, validation_size=0.2)</pre></div> </div></div></div></div></div></div>



### Light Gradient-Boosting Machine (LightGBM)
In comparison to EBM, the pipeline detailed below was significantly faster in determining the optimal hyperparameters, taking only 20 seconds as opposed to 10 hours—a substantial improvement, thanks in large part to an excellent resource available [here](https://macalusojeff.github.io/post/HyperparameterTuningLGBM/). Even without any preprocessing, LightGBM achieved the following scores on validation data:
- Precision: 0.70
- Recall: 0.59

Adding the optimal number of estimators, as determined from the early stopping rounds, did not result in any noticeable improvement. However, when also integrating the preprocessor with our selected transformers led to enhanced performance:
- Precision: 0.71
- Recall: 0.60

We've increased precision by 2 percentage points and recall by 3 percentage points through tuning parameters. It's worth noting that LightGBM was more overfit compared to EBM for the out-of-the-box (OOB) model. When compared to the baseline model, LightGBM maintains the same level of precision while achieving a 2-point increase in recall, improving from 0.58 to 0.60.


```python
import scipy.stats
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

# Defining a new model object with a large number of estimators for early stopping
lgbm_model = lgb.LGBMClassifier(n_estimators=10000, n_jobs=-1, random_state=random_state, verbose=-1)

# Define the parameter distributions for hyperparameter tuning
param_distributions = {
    "colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Also known as feature_fraction
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),
    "max_depth": np.append(-1, np.arange(3, 16)),
    "min_child_samples": scipy.stats.randint(5, 300),  # Also known as min_data_in_leaf
    "num_leaves": scipy.stats.randint(8, 256),
    "reg_alpha": [0, 0.01, 1, 2, 5, 7, 10, 50, 100],  # Also known as lambda_l1
    "reg_lambda": [0, 0.01, 1, 5, 10, 20, 50, 100],  # Also known as lambda_l2
    "subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Also known as bagging_fraction
    #"is_unbalance": [True],  # https://stackoverflow.com/questions/71838896/how-to-use-is-unbalance-and-scale-pos-weight-parameters-in-lightgbm-for-a-bi
    #"scale_pos_weight": [0.4],
}

# Configure the randomized search
random_search = RandomizedSearchCV(lgbm_model,
                                   param_distributions=param_distributions,
                                   n_iter=40,
                                   cv=3,
                                   scoring="f1",  # Use accuracy for classification tasks
                                   n_jobs=-1,
                                   random_state=random_state)

# Perform the randomized search with early stopping
random_search.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20)])

# Functions to display search results and roc-curve
goose.display_search_results(random_search, X_train, y_train, X_val, y_val)
goose.display_roc_curve(random_search, X_train, y_train, X_val, y_val, data_to_display='validation')
```

    Training until validation scores don't improve for 20 rounds
    Early stopping, best iteration is:
    [144]	valid_0's binary_logloss: 0.226971
    +-----------------------------------------+
    |      LGBMClassifier Tuning Results      |
    +-------------------+---------------------+
    |     Parameter     |        Value        |
    +-------------------+---------------------+
    |  colsample_bytree |  0.9047505230698577 |
    |   learning_rate   | 0.06924653758542858 |
    |     max_depth     |          8          |
    | min_child_samples |         132         |
    |     num_leaves    |          46         |
    |     reg_alpha     |         0.01        |
    |     reg_lambda    |         100         |
    |     subsample     |  0.5687604720729966 |
    +-------------------+---------------------+
    Classification Report - Training Set:
                  precision    recall  f1-score   support
    
               0       0.93      0.96      0.95      6234
               1       0.76      0.62      0.68      1151
    
        accuracy                           0.91      7385
       macro avg       0.85      0.79      0.82      7385
    weighted avg       0.91      0.91      0.91      7385
    
    Classification Report - Validation Set:
                  precision    recall  f1-score   support
    
               0       0.93      0.95      0.94      1332
               1       0.70      0.59      0.64       246
    
        accuracy                           0.90      1578
       macro avg       0.82      0.77      0.79      1578
    weighted avg       0.89      0.90      0.89      1578
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_76_1.png)
    


#### Refit with Best Parameters Including n_estimators and Preprocessor
To effectively use the best number of estimators identified through early stopping, we initialize the model with this key parameter and the other optimal hyperparameters from the grid search. Re-training the pipeline with these settings, together with the preprocessor, results in a recall improvement of one percentage point.


```python
# Extract the parameters from the best model to re-train the model
# Update the number of estimators to the best iteration from early stopping
best_model = random_search.best_estimator_
optimal_params = best_model.get_params()
optimal_params["n_estimators"] = best_model.best_iteration_

# Re-train the tuned model with the optimal parameters
lgbm_model = lgb.LGBMClassifier(**optimal_params)  # Inherits n_jobs and random_state from above

# lgbm_model.fit(X_train, y_train)
lgbm_pipe = make_pipeline(preprocessor, lgbm_model)

lgbm_pipe.fit(X_train, y_train)

# y_train_pred = lgbm_pipe.predict(X_train)
# y_val_pred = lgbm_pipe.predict(X_val)

goose.display_classification_report(lgbm_pipe, X_train, y_train, X_val, y_val, data_to_display='validation')

goose.display_roc_curve(lgbm_pipe, X_train, y_train, X_val, y_val, data_to_display='validation')
```

    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.93      0.95      0.94      1332
               1       0.71      0.60      0.65       246
    
        accuracy                           0.90      1578
       macro avg       0.82      0.78      0.80      1578
    weighted avg       0.89      0.90      0.90      1578
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_78_1.png)
    


## Sampling Techniques

We explore the impact of various sampling techniques on model performance:

- **Subsampling**: Reducing the majority class to balance the dataset.
- **Oversampling**: Increasing the minority class to achieve balance.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generating synthetic samples for the minority class.

Our goal is to determine whether these techniques enhance the predictive accuracy of our models. We will assess their effectiveness using classification reports and ROC-AUC metrics.

### Applying Sampling Techniques
Utilizing the `imblearn` library to generate new sampled versions of the dataset through various techniques.


```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

################## Subsampling ##################
# Subsampling the majority class
undersampler = RandomUnderSampler(random_state=random_state)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

################## Oversampling ##################
# Oversampling the minority class
oversampler = RandomOverSampler(random_state=random_state)
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

################## SMOTE ##################
# Fit and transform the training data with the preprocessor
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

# Apply SMOTE to the preprocessed training data
smote = SMOTE(random_state=random_state)
X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)
```

### EBM on Sampled Datasets
The fitting times for EBM were quite aggressive, yet the results did not yield any noticeable improvements compared to the unsampled dataset.


```python
from sklearn.base import clone
from sklearn.metrics import classification_report

# Fit model on sampled training data for random under sampling
ebm_pipe_undersampled = clone(ebm_pipe)
ebm_pipe_undersampled.fit(X_train_resampled, y_train_resampled)

# Fit model on sampled training data for random over sampling
ebm_pipe_oversampled = clone(ebm_pipe)
ebm_pipe_oversampled.fit(X_train_oversampled, y_train_oversampled)

# Fit model on sampled training data for SMOTE
ebm_model_smote = clone(ebm_model)
ebm_model_smote.fit(X_train_smote, y_train_smote)

# Display classification report for random under sampling
y_val_pred_undersampled = ebm_pipe_undersampled.predict(X_val)
print("Undersampling EBM:")
print(classification_report(y_val, y_val_pred_undersampled))

# Display classification report for random over sampling
y_val_pred_oversampled = ebm_pipe_oversampled.predict(X_val)
print("Oversampling EBM:")
print(classification_report(y_val, y_val_pred_oversampled))

# Display classification report for SMOTE
y_val_pred_smote = ebm_model_smote.predict(X_val_preprocessed)
print("SMOTE EBM:")
print(classification_report(y_val, y_val_pred_smote))

```

    Undersampling EBM:
                  precision    recall  f1-score   support
    
               0       0.97      0.87      0.92      1332
               1       0.55      0.86      0.67       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.86      0.79      1578
    weighted avg       0.90      0.87      0.88      1578
    
    Oversampling EBM:
                  precision    recall  f1-score   support
    
               0       0.93      0.91      0.92      1332
               1       0.58      0.64      0.61       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.78      0.77      1578
    weighted avg       0.88      0.87      0.87      1578
    
    SMOTE EBM:
                  precision    recall  f1-score   support
    
               0       0.94      0.94      0.94      1332
               1       0.68      0.66      0.67       246
    
        accuracy                           0.90      1578
       macro avg       0.81      0.80      0.80      1578
    weighted avg       0.90      0.90      0.90      1578
    
    

#### EBM Threshold Comparison (ROC-AUC)
Visual comparison of ROC curves shows no threshold improvements, with AUC mirroring that of the unsampled dataset model, except for random oversampling, which results in a lower AUC.


```python
################## Helper Function ##################
from sklearn.metrics import roc_curve, auc

# Define a function to plot ROC curve
def plot_roc_curve(y_true, y_proba, method_name, ax):
    # Extract probabilities for the positive class
    y_proba_positive = y_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_positive)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'{method_name} (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='lightslategray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve ({method_name})')
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
```


```python
# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 4))

# Plot ROC curve for No Sampling
plot_roc_curve(y_val, ebm_pipe.predict_proba(X_val), "No Sampling", axs[0])

# Plot ROC curve for Random Under Sampling
plot_roc_curve(y_val, ebm_pipe_undersampled.predict_proba(X_val), "Random Under Sampling", axs[1])

# Plot ROC curve for Random Over Sampling
plot_roc_curve(y_val, ebm_pipe_oversampled.predict_proba(X_val), "Random Over Sampling", axs[2])

# Plot ROC curve for SMOTE
plot_roc_curve(y_val, ebm_model_smote.predict_proba(X_val_preprocessed), "SMOTE", axs[3])

plt.tight_layout()
plt.show()


```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_86_0.png)
    


### LightGBM on Sampled Datasets
The fitting time for LightGBM was almost instant. However, analysis suggests no significant enhancement in metrics for the LightGBM model post-application of sampling techniques.


```python
from sklearn.base import clone

# Re-fit and Display Metrics for..

################## Subsampled Data ##################
# Fit model on sampled training data
lgm_pipe_undersampled = clone(lgbm_pipe)
lgm_pipe_undersampled.fit(X_train_resampled, y_train_resampled)

# Display classification report
y_val_pred_undersampled = lgm_pipe_undersampled.predict(X_val)
print("Subsampling LightGBM:")
print(classification_report(y_val, y_val_pred_undersampled))

################## Oversampled Data ##################
# Fit model on sampled training data
lgbm_pipe_oversampled = clone(lgbm_pipe)
lgbm_pipe_oversampled.fit(X_train_oversampled, y_train_oversampled)

# Display classification report
y_val_pred_oversampled = lgbm_pipe_oversampled.predict(X_val)
print("Oversampling LightGBM:")
print(classification_report(y_val, y_val_pred_oversampled))

################## SMOTE Sampled Data ##################
# Fit model on sampled training data
lgbm_model_smote = clone(lgbm_model)
lgbm_model_smote.fit(X_train_smote, y_train_smote)

# Predict on the validation set
y_val_pred_smote = lgbm_model_smote.predict(X_val_preprocessed)
print("SMOTE LightGBM:")
print(classification_report(y_val, y_val_pred_smote))
```

    Subsampling LightGBM:
                  precision    recall  f1-score   support
    
               0       0.97      0.87      0.92      1332
               1       0.55      0.87      0.68       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.87      0.80      1578
    weighted avg       0.91      0.87      0.88      1578
    
    Oversampling LightGBM:
                  precision    recall  f1-score   support
    
               0       0.97      0.87      0.92      1332
               1       0.55      0.86      0.67       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.87      0.79      1578
    weighted avg       0.91      0.87      0.88      1578
    
    SMOTE LightGBM:
                  precision    recall  f1-score   support
    
               0       0.95      0.92      0.94      1332
               1       0.64      0.75      0.69       246
    
        accuracy                           0.90      1578
       macro avg       0.80      0.84      0.81      1578
    weighted avg       0.90      0.90      0.90      1578
    
    

#### LigthGBM Threshold Comparison (ROC-AUC)
Comparative assessment indicates that oversampling marginally increased the AUC score by 1 point for the LightGBM model. However, the curve, where the False Positives are lower—which is of interest—is not of any significant improvement compared to the unsampled dataset.


```python
# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 4))

# Plot ROC curve for No Sampling
plot_roc_curve(y_val, lgbm_pipe.predict_proba(X_val), "No Sampling", axs[0])

# Plot ROC curve for Random Under Sampling
plot_roc_curve(y_val, lgm_pipe_undersampled.predict_proba(X_val), "Random Under Sampling", axs[1])

# Plot ROC curve for Random Over Sampling
plot_roc_curve(y_val, lgbm_pipe_oversampled.predict_proba(X_val), "Random Over Sampling", axs[2])

# Plot ROC curve for SMOTE
plot_roc_curve(y_val, lgbm_model_smote.predict_proba(X_val_preprocessed), "SMOTE", axs[3])

plt.tight_layout()
plt.show()
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_90_0.png)
    


### Baseline Model on Sampled Data
The absence of marked improvements in EBM and LightEBM prompts us to explore sampling techniques for our baseline model, i.e., Logistic Regression. The performance slightly lags behind that of the aforementioned models, EBM and LightEBM.


```python
print("No Sampling Logistic Regression:")
goose.display_classification_report(log_pipe, X_train, y_train, X_val, y_val, data_to_display='validation')

# Re-fit and Display Metrics for..

################## Subsampled Data ##################
# Fit model on sampled training data
log_pipe_undersampled = clone(log_pipe)
log_pipe_undersampled.fit(X_train_resampled, y_train_resampled)

# Display classification report
y_val_pred_undersampled = log_pipe_undersampled.predict(X_val)
print("Subsampling Logistic Regression:")
print(classification_report(y_val, y_val_pred_undersampled))

################## Oversampled Data ##################
# Fit model on sampled training data
log_pipe_oversampled = clone(log_pipe)
log_pipe_oversampled.fit(X_train_oversampled, y_train_oversampled)

# Display classification report
y_val_pred_oversampled = log_pipe_oversampled.predict(X_val)
print("Oversampling Logistic Regression:")
print(classification_report(y_val, y_val_pred_oversampled))

################## SMOTE Sampled Data ##################
# Fit model on sampled training data
log_model_smote = clone(log_model)
log_model_smote.fit(X_train_smote, y_train_smote)

# Predict on the validation set
y_val_pred_smote = log_model_smote.predict(X_val_preprocessed)
print("SMOTE Logistic Regression:")
print(classification_report(y_val, y_val_pred_smote))

################## Plot Roc Curves ##################
# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 4))

# Plot ROC curve for No Sampling
plot_roc_curve(y_val, log_pipe.predict_proba(X_val), "No Sampling", axs[0])

# Plot ROC curve for Random Under Sampling
plot_roc_curve(y_val, log_pipe_undersampled.predict_proba(X_val), "Random Under Sampling", axs[1])

# Plot ROC curve for Random Over Sampling
plot_roc_curve(y_val, log_pipe_oversampled.predict_proba(X_val), "Random Over Sampling", axs[2])

# Plot ROC curve for SMOTE
plot_roc_curve(y_val, log_model_smote.predict_proba(X_val_preprocessed), "SMOTE", axs[3])

plt.tight_layout()
plt.show()
```

    No Sampling Logistic Regression:
    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.93      0.96      0.94      1332
               1       0.71      0.59      0.64       246
    
        accuracy                           0.90      1578
       macro avg       0.82      0.77      0.79      1578
    weighted avg       0.89      0.90      0.89      1578
    
    Subsampling Logistic Regression:
                  precision    recall  f1-score   support
    
               0       0.96      0.88      0.92      1332
               1       0.56      0.83      0.67       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.85      0.79      1578
    weighted avg       0.90      0.87      0.88      1578
    
    Oversampling Logistic Regression:
                  precision    recall  f1-score   support
    
               0       0.97      0.88      0.92      1332
               1       0.56      0.83      0.67       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.85      0.79      1578
    weighted avg       0.90      0.87      0.88      1578
    
    

    SMOTE Logistic Regression:
                  precision    recall  f1-score   support
    
               0       0.97      0.87      0.92      1332
               1       0.55      0.83      0.66       246
    
        accuracy                           0.87      1578
       macro avg       0.76      0.85      0.79      1578
    weighted avg       0.90      0.87      0.88      1578
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_92_2.png)
    


### Summarizing Sampling Techniques
Despite different sampling methods showing varied precision and recall for the majority class, the virtually identical ROC-AUC across subsampling, oversampling, and SMOTE indicates uniform model discrimination. The sampling affects class balance and threshold-dependent metrics, but not the underlying discriminative capacity of the model. Adjusting the decision threshold can reconcile precision-recall disparities without affecting AUC. In simple terms: **Sampling techniques did not improve the estimators predictions.**

## Revisiting Model Selection: A Comparative Analysis at Specific Thresholds
Thus far, it's not entirely clear whether EBM and/or LightGBM have meaningfully outperformed the baseline model. Adhering to our task of building an effective initial model for accurately profiling our converting customers, in this section, we will examine the precision and recall scores at various thresholds. This analysis aims to determine which model is most feasible for this task and, potentially, for future implementations of a predictive model in post-profiling scenarios.

### EBM Profiling Model Analysis:
EBM notably surpasses the alternative models in striking a balance between high precision and practical recall. Here are the specifics at a threshold of 0.73:

- **Precision**: 0.86

- **Recall**: 0.34

- **Confusion Matrix Highlights**:
    - True Positives: 89
    - True Negatives: 1313
    - False Positives: 19
    - False Negatives: 157

**EBM Profiling Model Example**: 
In profiling 1578 customers, our model predicts 108 would convert. Among these, 19 are false positives—incorrectly predicted to convert—and 89 genuinely do. It's important to note that this model's application is intended for historical data analysis, not real-time decision-making. We propose that analyzing a group of 1578 customers can yield insightful understandings of customer conversion behaviors. Additionally, we consider the potential for real-time model deployment, for which LightGBM emerges as a more suitable candidate.

### LightGBM Potential Deployment Model Insights:
The exact future application for forecasting customer conversions in real-time remains to be determined. The LightGBM model's effectiveness hinges on a recall rate surpassing that of the profiling model to identify a sufficient number of conversions for any applied strategy to be successful. While precision remains crucial—to mitigate the repercussions of excessive false positives on cost-efficiency or brand reputation—we posit that model interpretability is less critical in this context. This stance is predicated on the nature of the business, which does not demand the level of interpretability required in sectors like finance, healthcare or criminal justice. Opting for a threshold of 0.48 with LightGBM, we achieve:

- **Precision**: 0.71

- **Recall**: 0.64

- **Confusion Matrix Highlights**:
    - True Positives: 144
    - True Negatives: 1266
    - False Positives: 66
    - False Negatives: 102

**LightGBM Real-Time Deployment Example**: 
Website visitors deemed likely to convert by the model could receive intensified follow-up marketing efforts. Utilizing the model in real-time over a month on 1,578 users, we anticipate 144 correct predictions of conversion, triggering additional marketing initiatives. Conversely, 66 users would be incorrectly targeted, and 102 would be overlooked as false negatives. The strategy's future adjustments—increasing the threshold to reduce negative feedback or lowering it to capture more potential conversions—will depend on the marketing campaign's impact on customer satisfaction and company reputation.

#### Logistic Regression Across Thresholds
After reviewing various thresholds, we conclude that the default threshold of 0.5 is optimal. This is because any further improvement in precision, albeit slight from 0.71 to 0.72, would cause a significant drop in recall, from 0.59 to 0.34. This trade-off leads to the loss of too many True Positives (TP), with the minimal increase in precision failing to justify the substantial decrease in captured TPs, thereby rendering it unsuitable for a viable model.
  
Consequently, we reject the null hypothesis positing that Logistic Regression's performance is equivalent to that of alternative models. Instead, we accept the alternative hypothesis indicating a significant performance discrepancy between the baseline model and at least one of the alternative models.


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_thresholds(model, X, y):
    # Probabilities for target
    pred_proba = model.predict_proba(X)
    pred_proba_pos = pred_proba[:, 1]
    threshold_values = np.linspace(0.2, 0.8, 25)

    # Prepare an empty list to collect metrics for each threshold
    metrics = []

    for threshold in threshold_values:
        y_pred = (pred_proba_pos >= threshold).astype(int)
        accuracy = round(accuracy_score(y, y_pred), 2)
        precision = round(precision_score(y, y_pred), 2)
        recall = round(recall_score(y, y_pred), 2)

        # Append the metrics for the current threshold to the list
        metrics.append([round(threshold, 2), precision, recall])

    # Convert the list of metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics, columns=['Threshold', 'Precision', 'Recall'])

    return metrics_df

evaluate_thresholds(log_model, X_val_preprocessed, y_val)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.60</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.22</td>
      <td>0.63</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.64</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.64</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.30</td>
      <td>0.65</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.33</td>
      <td>0.65</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.35</td>
      <td>0.66</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.38</td>
      <td>0.66</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.40</td>
      <td>0.67</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.43</td>
      <td>0.67</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.68</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.48</td>
      <td>0.69</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.50</td>
      <td>0.71</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.53</td>
      <td>0.70</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.55</td>
      <td>0.70</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.58</td>
      <td>0.69</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.60</td>
      <td>0.69</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.63</td>
      <td>0.70</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.65</td>
      <td>0.69</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.68</td>
      <td>0.69</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.70</td>
      <td>0.72</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.73</td>
      <td>0.74</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.75</td>
      <td>0.72</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.78</td>
      <td>0.73</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.80</td>
      <td>0.74</td>
      <td>0.24</td>
    </tr>
  </tbody>
</table>
</div>




```python
################## Helper Function ##################
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix_with_threshold(model, X_val, y_val, threshold=0.5, normalize=True):
    sns.set_style("white")
    
    # Predict probabilities for the positive class
    pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Convert predicted probabilities to binary predictions based on the threshold
    y_pred = (pred_proba >= threshold).astype(int)

    # Generate the confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    if normalize:
        # Normalize the confusion matrix to get percentages of the total observations
        cm = cm.astype('float') / np.sum(cm)
    
    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix at Threshold {threshold}')
    plt.show()
```

#### LightGBM Across Thresholds
Upon a closer examination, we observe that increasing the threshold allows us to significantly improve precision without sacrificing too much relative recall, up to a point where we achieve a precision of 0.78 and a recall of 0.37. This adjustment yields a reasonable sample size while marking a substantial increase in precision compared to the default threshold.


```python
evaluate_thresholds(lgbm_model, X_val_preprocessed, y_val)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.59</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.22</td>
      <td>0.61</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.64</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.65</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.30</td>
      <td>0.66</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.33</td>
      <td>0.67</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.35</td>
      <td>0.67</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.38</td>
      <td>0.66</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.40</td>
      <td>0.68</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.43</td>
      <td>0.68</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.69</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.48</td>
      <td>0.71</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.50</td>
      <td>0.71</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.53</td>
      <td>0.71</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.55</td>
      <td>0.72</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.58</td>
      <td>0.72</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.60</td>
      <td>0.75</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.63</td>
      <td>0.78</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.65</td>
      <td>0.79</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.68</td>
      <td>0.78</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.70</td>
      <td>0.80</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.73</td>
      <td>0.82</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.75</td>
      <td>0.89</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.78</td>
      <td>0.89</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.80</td>
      <td>0.87</td>
      <td>0.19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confusion Matrix with percentages
plot_confusion_matrix_with_threshold(lgbm_model, X_val_preprocessed, y_val, threshold=0.48, normalize=False)
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_100_0.png)
    


#### EBM Across Thresholds
The analysis becomes particularly intriguing with EBM. By adjusting the thresholds, we reach a precision of 0.86 and a recall of 0.34. This development clearly demonstrates that EBM outperforms LightGBM, marking excellent news for our project. Given these results, EBM can now be considered the optimal model for our current task. We require high precision to minimize False Negatives, ensuring a sufficient sample size. Furthermore, the necessity for a model that offers clear interpretability aligns with what Explainable Boosting Machines offer.


```python
evaluate_thresholds(ebm_model, X_val_preprocessed, y_val)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.59</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.22</td>
      <td>0.60</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.62</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.63</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.30</td>
      <td>0.65</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.33</td>
      <td>0.65</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.35</td>
      <td>0.65</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.38</td>
      <td>0.66</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.40</td>
      <td>0.66</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.43</td>
      <td>0.66</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.68</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.48</td>
      <td>0.68</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.50</td>
      <td>0.69</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.53</td>
      <td>0.70</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.55</td>
      <td>0.72</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.58</td>
      <td>0.75</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.60</td>
      <td>0.77</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.63</td>
      <td>0.77</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.65</td>
      <td>0.80</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.68</td>
      <td>0.81</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.70</td>
      <td>0.82</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.73</td>
      <td>0.86</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.75</td>
      <td>0.85</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.78</td>
      <td>0.86</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.80</td>
      <td>0.87</td>
      <td>0.24</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confusion Matrix
print(len(y_val))
plot_confusion_matrix_with_threshold(ebm_model, X_val_preprocessed, y_val, threshold=0.7, normalize=False)
```

    1578
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_103_1.png)
    


## Evaluation of the Imputation Pipeline
Considering the importance of an imputation strategy not only in profiling but also, and perhaps more critically, in real-time applications, we chose to evaluate the efficacy of the imputation pipeline on our preferred model for such applications, namely, LightGBM.
  
Our investigations into imputation strategies reveal no significant improvement in the model's predictive performance. With a chosen threshold of 0.45, the imputation-enhanced model yields a precision of 0.71 and a recall of 0.57. This is comparable to the model's precision of 0.71 and recall of 0.64, without imputation. 
  
It's important to emphasize the potential relevance of the imputation pipeline in a production environment. Especially if missing values persist and predictions are needed for a broad user base, the value of this pipeline becomes apparent. Originally, missing data represented 13% of our dataset. Should such missing data issues continue, the imputation pipeline proves its merit for real-time application by maintaining model performance without compromising generalizability. In this context, achieving metric consistency, even if only for precision, is a substantial benefit that preserves the model's reliability.


```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

############### Load and Setup Dataset ###############
df2 = pd.read_csv('project_data.csv')
df2.drop('Unnamed: 0', axis=1, inplace=True)

# Missing values in target are not feasible and are dropped
df2.dropna(subset=['Revenue'], inplace=True)

# Likewise goes for duplicated rows
df2.drop_duplicates(inplace=True)

# Relabel 'Turc' to NaN
df2['Month'] = df2['Month'].replace('Turc', np.nan)

# Consolidate "Sept" with "Sep"
df2['Month'] = df2['Month'].replace('Sept', 'Sep')

# Relabel 'Name:Zara' to NaN
df2['Weekend'] = df2['Weekend'].replace('Name:Zara', np.nan)

# Replace negative and irrational values with NaN
df2.loc[df2['Administrative'] < 0, 'Administrative'] = np.nan
df2.loc[df2['BounceRates'] < 0, 'BounceRates'] = np.nan
df2.loc[df2['BounceRates'] > 1, 'BounceRates'] = np.nan

# Apply mapping to clear up 0 for constant imputing
special_day_mapping = {0.0: 1, 0.2: 2, 0.4: 3, 0.6: 4, 0.8: 5, 1.0: 6}
df2['SpecialDay'] = df2['SpecialDay'].map(special_day_mapping)

# Convert object 'Weekend' and boolean 'Revenue' to numeric representation
df2['Weekend'] = df2['Weekend'].map({'True': 1, 'False': 0})
df2['Revenue'] = df2['Revenue'].map({True: 1, False: 0})

X_train2, y_train2, X_val2, y_val2, X_test2, y_test2, X_train_val2, y_train_val2 = goose.split_data(df2, "Revenue", verbose=False)

####################### Pipeline #######################
# Scaling and transformation
scaler = RobustScaler()
robust_scaler = make_pipeline(scaler)
log_transform = make_pipeline(FunctionTransformer(np.log1p), scaler)
sqrt_transform = make_pipeline(FunctionTransformer(np.sqrt), scaler)

# Encoding
onehot_encode = make_pipeline(OneHotEncoder(drop='if_binary', handle_unknown='ignore', dtype=int))

# Imputation
knn_impute = make_pipeline(KNNImputer(n_neighbors=2), log_transform)
specialday_impute = make_pipeline(SimpleImputer(strategy='constant', fill_value=0), onehot_encode) # SimpleImputer(strategy='most_frequent')
weekend_impute = make_pipeline(SimpleImputer(strategy='most_frequent'), onehot_encode)
onehot_impute = make_pipeline(SimpleImputer(strategy='most_frequent'), onehot_encode) # SimpleImputer(strategy='constant', fill_value=0)

# Iterative Imputer transformer
# iterative_imputer_transformer = make_pipeline(
#     FunctionTransformer(np.log1p, validate=False),  # Applying log transformation
#     IterativeImputer(random_state=random_state, ),
#     scaler
# )

# Month ordinal encoding setup
months_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_ordinal = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder(categories=[months_order]))

# Preprocessor setup
preprocessor2 = ColumnTransformer(
    transformers=[
        ('bouncerate_knn', knn_impute, ['BounceRates']),
        ('month_ordinal_encode', month_ordinal, ['Month']),
        ('specialday_impute_encode', specialday_impute, ['SpecialDay']),
        ('weekend_impute_encode', weekend_impute, ['Weekend']),
        ('missing_category_impute_encode', onehot_impute, ['Browser', 'Region']),
        ('product_related_robust', robust_scaler, ['ProductRelated', 'ProductRelated_Duration']),
        ('log_transform_scale', log_transform, ['Administrative_Duration', 'Informational_Duration', 'ExitRates', 'PageValues']),
        ('sqrt_transform_scale', sqrt_transform, []),
        ('categorical_encode', onehot_encode, ['VisitorType', 'Administrative', 'Informational', 'OperatingSystems', 'TrafficType'])
    ],
    remainder='passthrough'
)

# Model pipeline
model_pipeline = make_pipeline(preprocessor2, lgbm_model)

# Model training
model_pipeline.fit(X_train2, y_train2)

goose.display_classification_report(model_pipeline, X_train2, y_train2, X_val2, y_val2, data_to_display='both')
goose.display_roc_curve(model_pipeline, X_train2, y_train2, X_val2, y_val2, data_to_display='validation')
```

    Classification Report (Training):
                  precision    recall  f1-score   support
    
               0       0.94      0.96      0.95      7151
               1       0.77      0.64      0.70      1317
    
        accuracy                           0.91      8468
       macro avg       0.85      0.80      0.83      8468
    weighted avg       0.91      0.91      0.91      8468
    
    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.92      0.96      0.94      1528
               1       0.71      0.53      0.61       281
    
        accuracy                           0.89      1809
       macro avg       0.82      0.74      0.77      1809
    weighted avg       0.89      0.89      0.89      1809
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_105_1.png)
    



```python
evaluate_thresholds(model_pipeline, X_val2, y_val2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.56</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.22</td>
      <td>0.58</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.59</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.60</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.30</td>
      <td>0.63</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.33</td>
      <td>0.65</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.35</td>
      <td>0.66</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.38</td>
      <td>0.67</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.40</td>
      <td>0.68</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.43</td>
      <td>0.70</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.69</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.48</td>
      <td>0.71</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.50</td>
      <td>0.71</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.53</td>
      <td>0.74</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.55</td>
      <td>0.76</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.58</td>
      <td>0.77</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.60</td>
      <td>0.78</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.63</td>
      <td>0.80</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.65</td>
      <td>0.79</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.68</td>
      <td>0.83</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.70</td>
      <td>0.84</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.73</td>
      <td>0.86</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.75</td>
      <td>0.89</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.78</td>
      <td>0.90</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.80</td>
      <td>0.93</td>
      <td>0.22</td>
    </tr>
  </tbody>
</table>
</div>



## Final Model Evaluation
Our model assessment will utilize the unseen test set to gain insights into what drives customer conversions. This critical analysis will underpin the profiling of user behavior for stakeholder review and motivate the development of a deployable, real-time model for the website.

Step-by-step:
1. Extract feature names for EBM's upcoming profiling anlysis
2. Instantiate the model with extracted feature names and optimal parameters from the grid search.
3. Refit the model using both train and validation data sets combined.
4. Evaluate the model by displaying and analysing metrics, confusion matrices and ROC-AUC on the hold-out test set (unseen data).
5. Summarize the findings.

### Extract Feature Names
In our preprocessing pipeline, the feature names get obscured due to feature engineering. This code segment serves to remedy that by fetching and storing them. This allows us to retrain the model with the original feature names intact, enhancing our understanding of the model's behavior.


```python
# Define transformation rules for clarity.
transformation_rules = {
    "OneHotEncoder": ['VisitorType', 'Informational', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend', 'Month', 'SpecialDay'],
    "log_transformer": ['Administrative_Duration', 'Informational_Duration', 'BounceRates', 'ExitRates', 'PageValues'],
    "Direct": ['ProductRelated', 'ProductRelated_Duration']
}

# Fit preprocessor
preprocessor.fit(X_train_val, y_train_val)

# Access fitted OneHotEncoder
onehot_encoder = preprocessor.named_transformers_['pipeline-4'].named_steps['onehotencoder']

# Get categories from OneHotEncoder
categories = onehot_encoder.categories_

# Create feature names for one-hot encoded features
onehot_feature_names = [f"{feature}_{cat}" for feature, cats in zip(transformation_rules["OneHotEncoder"], categories) for cat in cats]

# Drop-first in binary
onehot_feature_names.remove('Weekend_0')

# Define log-transformed feature names
log_features = [f'log_{col}' for col in transformation_rules["log_transformer"]]

# Define directly used features
direct_features = transformation_rules["Direct"]

# Combine all feature names
all_transformed_features = onehot_feature_names + log_features + direct_features

# Print transformed feature names
print("All Transformed Feature Names:", all_transformed_features)

```

    All Transformed Feature Names: ['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor', 'Informational_0', 'Informational_1', 'Informational_2', 'Informational_3', 'Informational_4', 'Informational_5', 'Informational_6', 'Informational_7', 'Informational_8', 'Informational_9', 'Informational_10', 'Informational_11', 'Informational_12', 'Informational_13', 'Informational_14', 'Informational_16', 'Informational_24', 'OperatingSystems_1', 'OperatingSystems_2', 'OperatingSystems_3', 'OperatingSystems_4', 'OperatingSystems_5', 'OperatingSystems_6', 'OperatingSystems_7', 'OperatingSystems_8', 'Browser_1', 'Browser_2', 'Browser_3', 'Browser_4', 'Browser_5', 'Browser_6', 'Browser_7', 'Browser_8', 'Browser_9', 'Browser_10', 'Browser_11', 'Browser_12', 'Browser_13', 'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5', 'Region_6', 'Region_7', 'Region_8', 'Region_9', 'TrafficType_1', 'TrafficType_2', 'TrafficType_3', 'TrafficType_4', 'TrafficType_5', 'TrafficType_6', 'TrafficType_7', 'TrafficType_8', 'TrafficType_9', 'TrafficType_10', 'TrafficType_11', 'TrafficType_12', 'TrafficType_13', 'TrafficType_14', 'TrafficType_15', 'TrafficType_16', 'TrafficType_17', 'TrafficType_18', 'TrafficType_19', 'TrafficType_20', 'Weekend_1', 'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep', 'SpecialDay_0.0', 'SpecialDay_0.2', 'SpecialDay_0.4', 'SpecialDay_0.6', 'SpecialDay_0.8', 'SpecialDay_1.0', 'log_Administrative_Duration', 'log_Informational_Duration', 'log_BounceRates', 'log_ExitRates', 'log_PageValues', 'ProductRelated', 'ProductRelated_Duration']
    

### Model for Customer Profiling: Explainable Boosting Machine (EBM)
EBM is performing as well as we could have hoped. At the default threshold, our precision is at 0.73 (+4 points), with a recall of 0.6 (-1 point). The ROC-AUC is impressive, boasting an area under the curve of 0.94 (+1 point). We will perform a more thorough analysis of feature importance and customer conversion drivers in the upcoming section on interpretation and profiling.


```python
# Feature specific transformers
transformers = [
    (robust_transformer, ['ProductRelated', 'ProductRelated_Duration']),
    (log_transformer, ['Administrative_Duration', 'Informational_Duration', 'BounceRates', 'ExitRates', 'PageValues']),
    (sqrt_transformer, []),
    (onehot_transformer, ['VisitorType', 'Informational', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend', 'Month', 'SpecialDay']),
]

# Combine transformers
preprocessor = make_column_transformer(*transformers)

# Best params from grid search
ebm_params = {k.replace('explainableboostingclassifier__', ''): v 
              for k, v in grid_ebm.best_params_.items()}

# Instantiate estimator
ebm_model = ExplainableBoostingClassifier(**ebm_params, random_state=random_state, feature_names=all_transformed_features)

# Pipeline creation
ebm_pipe = make_pipeline(preprocessor, ebm_model)

# Train model
ebm_pipe.fit(X_train_val, y_train_val)

# Display report and roc-curve
goose.display_classification_report(ebm_pipe, X_train, y_train, X_test, y_test, data_to_display='validation')
goose.display_roc_curve(ebm_pipe, X_train, y_train, X_test, y_test, data_to_display='validation')
#goose.display_confusion_matrix(ebm_pipe, X_train, y_train, X_test, y_test, data_to_display='validation')
```

    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.93      0.96      0.94      1336
               1       0.73      0.60      0.66       246
    
        accuracy                           0.90      1582
       macro avg       0.83      0.78      0.80      1582
    weighted avg       0.90      0.90      0.90      1582
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_111_1.png)
    



```python
evaluate_thresholds(ebm_pipe, X_test, y_test)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.60</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.22</td>
      <td>0.61</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.63</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.65</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.30</td>
      <td>0.66</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.33</td>
      <td>0.66</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.35</td>
      <td>0.68</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.38</td>
      <td>0.68</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.40</td>
      <td>0.70</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.43</td>
      <td>0.70</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.71</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.48</td>
      <td>0.72</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.50</td>
      <td>0.73</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.53</td>
      <td>0.74</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.55</td>
      <td>0.75</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.58</td>
      <td>0.78</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.60</td>
      <td>0.80</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.63</td>
      <td>0.80</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.65</td>
      <td>0.81</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.68</td>
      <td>0.82</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.70</td>
      <td>0.84</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.73</td>
      <td>0.86</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.75</td>
      <td>0.88</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.78</td>
      <td>0.90</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.80</td>
      <td>0.94</td>
      <td>0.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confusion Matrix
plot_confusion_matrix_with_threshold(ebm_pipe, X_test, y_test, threshold=0.5, normalize=False)
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_113_0.png)
    


### Model for Deployment: Light Gradient-Boosting Machine (LightGBM)
The purpose of evaluating the model on the test set is to simulate its performance in a real-life scenario. Hence, this serves as our final evaluation and a projection of its potential real-world performance.

We are thrilled with LightGBM's generalizability, which showed even better performance on the test set than on the validation set. The performance at the standard threshold includes a precision of 0.75 (+3 points) and a recall of 0.63 (+2 points).

We recommend starting with a threshold of 0.43 for real-time deployment, yielding the following metrics and outcomes for a sample size of 1582:

- **Precision**: 0.72

- **Recall**: 0.72

- **Confusion Matrix Highlights**:
    - True Positives: 174
    - True Negatives: 1267
    - False Positives: 69
    - False Negatives: 72

The threshold can be adjusted based on performance and specific requirements.


```python
# Extract the parameters from the best model to re-train the model
# Update the number of estimators to the best iteration from early stopping
best_model = random_search.best_estimator_
optimal_params = best_model.get_params()
optimal_params["n_estimators"] = best_model.best_iteration_

# Retrain the tuned model with the optimal parameters
lgbm_model = lgb.LGBMClassifier(**optimal_params)  # Inherits n_jobs and random_state from above

lgbm_pipe = make_pipeline(preprocessor, lgbm_model)

lgbm_pipe.fit(X_train_val, y_train_val)

goose.display_classification_report(lgbm_pipe, X_train_val, y_train_val, X_test, y_test, data_to_display='validation')
goose.display_roc_curve(lgbm_pipe, X_train_val, y_train_val, X_test, y_test, data_to_display='validation')
```

    
    Classification Report (Validation):
                  precision    recall  f1-score   support
    
               0       0.93      0.96      0.95      1336
               1       0.74      0.62      0.68       246
    
        accuracy                           0.91      1582
       macro avg       0.84      0.79      0.81      1582
    weighted avg       0.90      0.91      0.90      1582
    
    


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_115_1.png)
    



```python
evaluate_thresholds(lgbm_pipe, X_test, y_test)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.20</td>
      <td>0.58</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.22</td>
      <td>0.60</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.62</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.28</td>
      <td>0.63</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.30</td>
      <td>0.66</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.33</td>
      <td>0.67</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.35</td>
      <td>0.68</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.38</td>
      <td>0.69</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.40</td>
      <td>0.70</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.43</td>
      <td>0.72</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.73</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.48</td>
      <td>0.74</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.50</td>
      <td>0.74</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.53</td>
      <td>0.76</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.55</td>
      <td>0.77</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.58</td>
      <td>0.79</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.60</td>
      <td>0.81</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.63</td>
      <td>0.83</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.65</td>
      <td>0.84</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.68</td>
      <td>0.85</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.70</td>
      <td>0.88</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.73</td>
      <td>0.91</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.75</td>
      <td>0.91</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.78</td>
      <td>0.91</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.80</td>
      <td>0.92</td>
      <td>0.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confusion Matrix
plot_confusion_matrix_with_threshold(lgbm_pipe, X_test, y_test, threshold=0.43, normalize=False)
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_117_0.png)
    


## Interpretation and Customer Profiling
We will now analyze the most important predictors that illuminates the incentives for users to convert. The chart below offers significant insights into customer behavior and the key factors that influence conversion. It highlights the most impactful individual sub-categories alongside interaction terms, demonstrating the combined effect of dual-feature interactions and their respective scores.

Key Insights:
- 'Informational_3' stands out as the predominant predictor. A user's frequent visits to this page significantly influence conversion likelihood.

- The interaction terms of 'Informational_3' with 'SpecialDay' values of 0.4, 0.6, and 0.8 indicate strong predictive power. This pattern may suggest that users are more inclined to visit certain informational pages as special days approach, possibly due to the company's strategic customer engagement.

- While 'Informational_2' isn't as impactful as 'Informational_3', its interaction with 'Informational_3' indicates a potential for joint influence on increasing conversions. This interplay suggests that 'Informational_2' deserves closer investigation for its possible role in boosting conversion effectiveness.

- 'The 'VisitorType_Returning_Visitor' is a robust predictor on its own and demonstrates even greater predictive strength when combined with 'Informational_3'. Similarly, 'VisitorType_Other' shows strong predictive power individually and forms a valuable interaction term with 'Informational_2'.

- 'SpecialDay_0.8' is notable as an individual predictor, signifying a strategic opportunity for targeted marketing initiatives during specific periods leading up to special days.

- The presence of 'TrafficType_10' among the top predictors is intriguing. The relevance of this feature depends on its definition, which, if clarified, could yield valuable insights into its significance for conversion.


```python
from interpret import show
from IPython.display import Image, display

# Display feature importances from file
display(Image(filename='assets/feature_importances_global.png'))

# Isolate the ebm classifier from pipeline
ebm = ebm_pipe.named_steps['explainableboostingclassifier']

# Display global explanations
show(ebm.explain_global())
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_119_0.png)
    



<!-- http://127.0.0.1:7001/2218686320656/ -->
<iframe src="http://127.0.0.1:7001/2218686320656/" width=100% height=800 frameBorder="0"></iframe>


### Local Explanations Insight
The image illustrates a single prediction where 'Informational_3' significantly influenced the model's correct prediction of class 1. Features contributing negatively are shown in blue, offering insight into decision drivers for each visitor classification.


```python
# Preprocess test data
X_test_preprocessed = preprocessor.transform(X_test)

# Display feature importances from file
display(Image(filename='assets/feature_importances_local_example.png'))

# Display interpret dashboard
#show(ebm.explain_global())
show([ebm.explain_global(), ebm.explain_local(X_test_preprocessed, y_test)], 0)
```


    
![png](gustaf_boden_ml_projekt_files/gustaf_boden_ml_projekt_121_0.png)
    



<!-- http://127.0.0.1:7001/2218759687872/ -->
<a href="http://127.0.0.1:7001/2218759687872/" target="_new">Open in new window</a><iframe src="http://127.0.0.1:7001/2218759687872/" width=100% height=800 frameBorder="0"></iframe>


# Conclusion

## Strategic Insights: Enhancing Conversion and User Experience

### **Presenting our Model of Choice**
The Explainable Boosting Machine Classifier (EBM) has emerged as a frontrunner, rivaling the powerful LightGBM. This success is credited not only to thorough feature engineering and hyperparameter tuning but also to a rigorous comparative analysis of thresholds and imputation techniques such as SMOTE, among other strategies. 
A key advantage of the EBM over LightGBM is its transparent 'glassbox' approach, facilitating in-depth model analysis and understanding, both currently and for future applications. This transparency, even if LightGBM had superior performance, might lead many experts to favor EBM.

### **Challenges**
#### Data Quality
- The data exhibited an excessive number of missing entries, inaccuracies, and inconsistencies.
    - Issues ranged from misclassified data types to misplaced sub-category values.
    - There were signs suggestive of data integrity issues—possible unauthorized alterations to the dataset.
    - A discussion with the data engineering team is imperative to ascertain the company's data maturity level. Although some problems may stem from data collection methods, others require technical resolutions.

#### Tuning Times
The EBM model's optimization process was time-consuming. The 'inner_bags' parameter, while resource-intensive, bolstered the model's performance and could be omitted for efficiency at the expense of a slight performance drop. The exact maintenance needs for the model remain unclear, but given that it performs exceptionally out-of-the-box (OOB), with only marginal gains from further hyperparameter tuning - future maintanance is likely to be straigthforward and may not necessitate extensive ongoing adjustments.

#### Feature Distribution and Skewness
Addressing substantial variance in continuous features through transformations has enhanced our model's performance. This step proved crucial for the comparative and baseline models, where transformations were key to achieving competitive metrics.

#### Beating our Baseline Model
Logistic Regression served as our baseline. Through targeted feature engineering, we lifted its recall from 0.36 to 0.59 while maintaining precision at 0.71. Initially, it appeared our null hypothesis—stating the baseline's equal performance to EBM and LightGBM—couldn't be rejected. However, thorough evaluation at varying thresholds confirmed the superior performance of our selected EBM model.

### **Performance**
For initial deployment, we recommend the EBM model with a default threshold at 0.5, which accurately identifies 73% of positive predictions (precision) and correctly detects 60% of actual positives (recall). Should false positives prove problematic, increasing the threshold to 0.8 yields a precision of 0.94 and a recall of 0.31, significantly reducing misclassifications in larger samples. For example, this threshold would result in only 60 False Positives in a sample size of a 1000.

### **Customer Profiling: Conversion Drivers and Improvement Suggestions**
Our main goal has been profiling based on historical data to decipher the drivers behind customer conversions. The model's insights reveal 'Informational' pages as significant conversion factors, especially types "1", "2", and "3". Moreover, 'SpecialDay' and 'VisitorType' display strong interaction effects with these pages. Notably, conversions are not maximized on special days themselves but in the preceding periods, suggesting that early marketing efforts could substantially boost revenues.

#### Maximizing the Pre-Special Day Period
Exploratory data analysis (EDA) revealed that current trends show untapped potential in the run-up to special days like Black Friday. While high activity and revenue occur on the day, pre-event engagement is lacking. Our model indicates higher conversion predictions during this lead-up, presenting a strategic opportunity for marketing before these days to enhance revenue.

#### Leveraging Returning Visitors
With a 25% conversion rate for new visitors and 14% for returnees, a tailored machine learning model could help improve the latter's conversion rates. Interaction patterns between returning visitors and their engagement with 'Informational_3' are promising areas for targeted conversion strategies.

#### Traffic Type 10
Lastly, 'TrafficType_10' stands out as a predictor for conversions. Understanding this traffic's source may aid in refining the customer profile and tailoring marketing efforts accordingly.

### Future Directions: Enhancing Model Efficacy
We have recommended deploying our tuned Explainable Boosting Machine (EBM) model initially, due to its comparable performance with LightGBM in precision-recall ratio and its superior interpretability. This interpretability is vital in the early deployment stages for understanding customer behavior and the impact of business strategy changes. However, we suggest considering the LightGBM for future scenarios where enhancing the True Positive Rate (recall) becomes a priority. LightGBM has shown to achieve a higher recall than EBM, albeit with slightly lower precision. It's recommended to periodically review performance and strategy alignment to determine if switching to LightGBM is beneficial, especially in scenarios where false positives have minimal business impact and the need for EBM's explainability is less critical.

### **Summary**
Our in-depth analysis has provided a thorough understanding of the factors driving conversions. This insight affords us the chance to refine our marketing approaches, ultimately boosting business efficacy. Moreover, it lays the foundation for implementing a real-time predictive model poised to identify customers with the highest conversion potential.

**Questions?**

Feel free to ask any questions or provide feedback.

<div style="text-align: left;">
<a href="https://freeimage.host/i/HXsIQUv"><img src="https://iili.io/HXsIQUv.md.jpg" alt="HXsIQUv.md.jpg" border="0" width="450"></a>
</div>


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
