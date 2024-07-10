import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'datasets/taxi_fare.csv')
data = pd.read_csv(file_path)
#print(data.info())
data_using = data.copy()

data_using["trip_duration"] = (data_using["trip_duration"] / data_using["trip_duration"].max()).round(2)
data_using["distance_traveled"] = (data_using["distance_traveled"] / data_using["distance_traveled"].max()).round(2)
data_using["num_of_passengers"] = (data_using["num_of_passengers"] / data_using["num_of_passengers"].max()).round(2)
data_using["fare"] = (data_using["fare"] / data_using["fare"].max()).round(2)
data_using["tip"] = (data_using["tip"] / data_using["tip"].max()).round(2)
data_using["miscellaneous_fees"] = (data_using["miscellaneous_fees"] / data_using["miscellaneous_fees"].max()).round(2)
data_using["total_fare"] = (data_using["total_fare"] / data_using["total_fare"].max()).round(2)
#print(data_using.head(10))

x = data_using.drop(columns=["surge_applied"])
y = data_using["surge_applied"]

#Convertir le tableau X en tableau numPy
x = x.to_numpy()

x = x.astype(float)
x_train_taxi_fare = x[:50000,:]
#x_train_taxi_fare = x
x_test_taxi_fare = x[146771:,:]
#print(x_train)

#Convertir le tableau Y en tableau numPy
y = y.to_numpy()
y = y.astype(float)

#Modification de toutes les valeurs y = 0 en -1
for index, value in enumerate(y.flat):
    if value == 0:
        y.flat[index] = -1

y_train_taxi_fare = y[:50000].reshape(-1, 1)
#y_train_taxi_fare = y.reshape(-1, 1)
y_test_taxi_fare = y[146771:]
