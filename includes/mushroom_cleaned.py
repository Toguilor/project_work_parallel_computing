import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'datasets/mushroom_cleaned.csv')
data = pd.read_csv(file_path)
#print(data.info())
data_using = data.copy()
#print(data_using.head(10))

data_using["cap-diameter"] = (data_using["cap-diameter"] / data_using["cap-diameter"].max()).round(2)
data_using["cap-shape"] = (data_using["cap-shape"] / data_using["cap-shape"].max()).round(2)
data_using["gill-attachment"] = (data_using["gill-attachment"] / data_using["gill-attachment"].max()).round(2)
data_using["gill-color"] = (data_using["gill-color"] / data_using["gill-color"].max()).round(2)
data_using["stem-height"] = (data_using["stem-height"] / data_using["stem-height"].max()).round(2)
data_using["stem-width"] = (data_using["stem-width"] / data_using["stem-width"].max()).round(2)
data_using["stem-color"] = (data_using["stem-color"] / data_using["stem-color"].max()).round(2)
data_using["season"] = (data_using["season"] / data_using["season"].max()).round(2)

x = data_using.drop(columns=["class"])
y = data_using["class"]

#Convertir le tableau X en tableau numPy
x = x.to_numpy()

x = x.astype(float)
x_train_mushroom = x[:10000,:]
#x_train_mushroom = x
x_test_mushroom = x[37825:,:]
#print(x_train)


#Convertir le tableau Y en tableau numPy
y = y.to_numpy()
y = y.astype(float)

#Modification de toutes les valeurs y = 0 en -1
i=0
for index, value in enumerate(y.flat):
    if value == 0:
        y.flat[index] = -1

y_train_mushroom = y[:10000].reshape(-1, 1)
#y_train_mushroom = y.reshape(-1, 1)
y_test_mushroom = y[37825:]
