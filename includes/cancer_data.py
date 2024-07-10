import pandas as pd
import numpy as np
import os

file_path = os.path.join(os.path.dirname(__file__), 'datasets/Cancer_Data.csv')
data = pd.read_csv(file_path)
#print(data.info())
data_using = data.copy()

data_using["radius_mean"] = (data_using["radius_mean"] / data_using["radius_mean"].max()).round(2)
data_using["texture_mean"] = (data_using["texture_mean"] / data_using["texture_mean"].max()).round(2)
data_using["perimeter_mean"] = (data_using["perimeter_mean"] / data_using["perimeter_mean"].max()).round(2)
data_using["area_mean"] = (data_using["area_mean"] / data_using["area_mean"].max()).round(2)
data_using["radius_se"] = (data_using["radius_se"] / data_using["radius_se"].max()).round(2)
data_using["texture_se"] = (data_using["texture_se"] / data_using["texture_se"].max()).round(2)
data_using["perimeter_se"] = (data_using["perimeter_se"] / data_using["perimeter_se"].max()).round(2)
data_using["area_se"] = (data_using["area_se"] / data_using["area_se"].max()).round(2)
data_using["radius_worst"] = (data_using["radius_worst"] / data_using["radius_worst"].max()).round(2)
data_using["texture_worst"] = (data_using["texture_worst"] / data_using["texture_worst"].max()).round(2)
data_using["perimeter_worst"] = (data_using["perimeter_worst"] / data_using["perimeter_worst"].max()).round(2)
data_using["area_worst"] = (data_using["area_worst"] / data_using["area_worst"].max()).round(2)
#data_using["id"] = (data_using["id"] / data_using["id"].max()).round(2)
#data_using["id"].round(2)

x = data_using.drop(columns=["id", "diagnosis"])
y = data_using["diagnosis"]


x = np.delete(x, 568, axis=0) # Suppression de la dernière ligne pour avoir un nombre paire de ligne
x = np.delete(x, 30, axis=1) # Suppression de la dernière colonne puisqu'elle était un NAN
y = np.delete(y, 568, axis=0) # Suppression de la dernière ligne pour avoir un nombre paire de ligne
#print(x[:10, :])

#Modification de toutes les valeurs de diagnosis en 0 et en 1
for index, value in enumerate(y.flat):
    if value == "M":
        y.flat[index] = -1
    elif value == "B":
        y.flat[index] = 1

y = y.astype(float)

#x_train_cancer = x[:398, :]
x_train_cancer = x
x_test_cancer = x[398:, :]

#y_train_cancer = y[:398:].reshape(-1, 1)
y_train_cancer = y.reshape(-1, 1)
y_test_cancer = y[398:]
