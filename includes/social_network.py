import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'datasets/Social_Network_Ads.csv')
data = pd.read_csv(file_path)
#print(data.info())
data_using = data.copy()

data_using["Age"] = (data_using["Age"] / data_using["Age"].max()).round(2)
data_using["EstimatedSalary"] = (data_using["EstimatedSalary"] / data_using["EstimatedSalary"].max()).round(2)
#print(data_using.head(10))

x = data_using.drop(columns=["Purchased", "User ID"])
y = data_using["Purchased"]

#Convertir le tableau X en tableau numPy
x = x.to_numpy()
#Modification de toutes les valeurs Gender en 0 et en 1
for index, value in enumerate(x.flat):
    if value == "Female":
        x.flat[index] = 0
    elif value == "Male":
        x.flat[index] = 1
x = x.astype(float)
#x_train_social_network = x[:300,:]
x_train_social_network = x
x_test_social_network = x[300:,:]

#Convertir le tableau Y en tableau numPy
y = y.to_numpy()
y = y.astype(float)

#Modification de toutes les valeurs y = 0 en -1
for index, value in enumerate(y.flat):
    if value == 0:
        y.flat[index] = 1
    else:
        y.flat[index] = -1

#y_train_social_network = y[:300].reshape(-1, 1)
y_train_social_network = y.reshape(-1, 1)
y_test_social_network = y[300:]
