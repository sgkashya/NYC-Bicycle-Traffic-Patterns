import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
# High Temperature: The highest temperature in one day in ˚F.
# Low Temperature: The lowest temperature in one day in ˚F.
# Precipitation: rain drop height (in inch).
# Total: The total for bike usage on four bridges in each day.

dataset_2 = pd.read_csv('nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pd.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pd.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pd.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pd.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_2.head()) #This line will print out your data

# Analysis Question 1
print('Analysis Question 1: Sensor Installation on Bridges')
dataset_2['Total'] = pd.to_numeric(dataset_2['Total'].replace(',','', regex=True))
corr_with_traffic = dataset_2[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']].corr()['Total'].sort_values(ascending=False)
print(f'{corr_with_traffic}\n')

# Analysis Question 2
print('Analysis Question 2: Predicting Traffic Based on Weather Forecast')
corr_with_weather = dataset_2[['High Temp', 'Low Temp', 'Precipitation', 'Total']].corr()['Total'].sort_values(ascending=False)
print(f'{corr_with_weather}\n')

# Analysis Question 3
print('Analysis Question 3: Predicting the Day of the Week Based on Bicycle Counts')
label_encoder = LabelEncoder()
dataset_2['Day'] = label_encoder.fit_transform(dataset_2['Day'])
cols = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
X = dataset_2[cols]
y = dataset_2['Day']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy for predicting the day of the week based on the # of bicyclists on the bridges: {round(accuracy, 4)} ({round(accuracy, 4)*100}%)")
