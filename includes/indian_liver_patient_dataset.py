import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'datasets/indian_liver_patient_dataset.csv')
data = pd.read_csv(file_path)
#print(data.info())
#print(data.isna().sum())
data_using = data.copy()
"""
data_nan = data_using.loc[data["Albumin_and_Globulin_Ratio"].isnull()]
data_nonan = data_using.dropna()
test_data = data_nan
#print(test_data)
df = data_nonan
y_train_data = df["Albumin_and_Globulin_Ratio"]
x_train_data = df.drop("Albumin_and_Globulin_Ratio", axis=1)
#print(x_train_data.shape)
x_test_data = test_data.drop("Albumin_and_Globulin_Ratio", axis=1)
#print(x_test_data.shape)

x_train_data = x_train_data.to_numpy()
for index, value in enumerate(x_train_data.flat):
    if value == "Female":
        x_train_data.flat[index] = 0
    elif value == "Male":
        x_train_data.flat[index] = 1
        
x_test_data = x_test_data.to_numpy()
for index, value in enumerate(x_test_data.flat):
    if value == "Female":
        x_test_data.flat[index] = 0
    elif value == "Male":
        x_test_data.flat[index] = 1

regLog = LinearRegression()
regLog.fit(x_train_data, y_train_data)
y_pred_data = regLog.predict(x_test_data)
#print(y_pred_data)
data_using.loc[data_using.Albumin_and_Globulin_Ratio.isnull(), 'Albumin_and_Globulin_Ratio'] = y_pred_data
print(data_using.info())
"""

data_using["Age"] = (data_using["Age"] / data_using["Age"].max()).round(2)
data_using["Total_Bilirubin"] = (data_using["Total_Bilirubin"] / data_using["Total_Bilirubin"].max()).round(2)
data_using["Direct_Bilirubin"] = (data_using["Direct_Bilirubin"] / data_using["Direct_Bilirubin"].max()).round(2)
data_using["Alkaline_Phosphotase"] = (data_using["Alkaline_Phosphotase"] / data_using["Alkaline_Phosphotase"].max()).round(2)
data_using["Alamine_Aminotransferase"] = (data_using["Alamine_Aminotransferase"] / data_using["Alamine_Aminotransferase"].max()).round(2)
data_using["Aspartate_Aminotransferase"] = (data_using["Aspartate_Aminotransferase"] / data_using["Aspartate_Aminotransferase"].max()).round(2)
data_using["Total_Protiens"] = (data_using["Total_Protiens"] / data_using["Total_Protiens"].max()).round(2)
data_using["Albumin"] = (data_using["Albumin"] / data_using["Albumin"].max()).round(2)

#print(data_using.head(10))

x = data_using.drop(columns=["Albumin_and_Globulin_Ratio", "Liver_Problem"])
y = data_using["Liver_Problem"]

#Convertir le tableau X en tableau numPy
x = x.to_numpy()

#Modification de toutes les valeurs Gender en 0 et en 1
for index, value in enumerate(x.flat):
    if value == "Female":
        x.flat[index] = 0
    elif value == "Male":
        x.flat[index] = 1
x = x.astype(float)
#x_train_indian_liver_patient = x[:350,:]
x_train_indian_liver_patient = x
x_test_indian_liver_patient = x[350:,:]
#print(x_train)

#Convertir le tableau Y en tableau numPy
y = y.to_numpy()
y = y.astype(float)

#Modification de toutes les valeurs y = 0 en -1
i=0
for index, value in enumerate(y.flat):
    if value == 2:
        y.flat[index] = -1

#y_train_indian_liver_patient = y[:350].reshape(-1, 1)
y_train_indian_liver_patient = y.reshape(-1, 1)
y_test_indian_liver_patient = y[350:]
