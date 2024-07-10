import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'datasets/diabetes2.csv')
data = pd.read_csv(file_path)
#print(data.head(10))
#print(data.isna().sum())
#print(data.describe())
#print(data.info())
data_using = data.copy()
#sns.countplot(data=data_using, x="Outcome")
#print(data_using.corr())
#sns.heatmap(data_using.corr())
#sns.displot(data_using["Age"], bins=20, kde=True)

#print(data_using.columns)

data_using["Pregnancies"] = (data_using["Pregnancies"] / data_using["Pregnancies"].max()).round(2)
data_using["Glucose"] = (data_using["Glucose"] / data_using["Glucose"].max()).round(2)
data_using["BloodPressure"] = (data_using["BloodPressure"] / data_using["BloodPressure"].max()).round(2)
data_using["SkinThickness"] = (data_using["SkinThickness"] / data_using["SkinThickness"].max()).round(2)
data_using["Insulin"] = (data_using["Insulin"] / data_using["Insulin"].max()).round(2)
data_using["BMI"] = (data_using["BMI"] / data_using["BMI"].max()).round(2)
data_using["DiabetesPedigreeFunction"] = (data_using["DiabetesPedigreeFunction"] / data_using["DiabetesPedigreeFunction"].max()).round(2)
data_using["Age"] = (data_using["Age"] / data_using["Age"].max()).round(2)


#print(data_using.info())
x = data_using.drop(columns=["Outcome"])
y = data_using["Outcome"]
#print(x.info())
#print(x.head())
#print("\n\n")
#print(y.head())

#Convertir le tableau X en tableau numPy
x = x.to_numpy()
i = 0
#x = np.delete(x, 4, axis=1)
#print(x[:10,:])
#x_train_diabete2 = x[:538,:]
x_train_diabete2 = x
x_test_diabete2 = x[538:,:]

#Convertir le tableau Y en tableau numPy
y = y.to_numpy()

#Modification de toutes les valeurs y = 0 en y = -1
for index, value in enumerate(y.flat):
    if value == 0:
        y.flat[index] = 1
    else:
        y.flat[index] = -1

y = y.astype(float)
#y_train_diabete2 = y[:538].reshape(-1, 1)
y_train_diabete2 = y.reshape(-1, 1)
y_test_diabete2 = y[538:]
