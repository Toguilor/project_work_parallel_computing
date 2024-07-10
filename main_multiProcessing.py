import time
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt
from optimization import descente_gradient, nesterov_optimizer
from includes.diabete2 import x_train_diabete2, y_train_diabete2
from includes.cancer_data import x_train_cancer, y_train_cancer
from includes.indian_liver_patient_dataset import x_train_indian_liver_patient, y_train_indian_liver_patient
from includes.mushroom_cleaned import x_train_mushroom, y_train_mushroom
from includes.social_network import x_train_social_network, y_train_social_network
from includes.taxi_fare import x_train_taxi_fare, y_train_taxi_fare

def generate_random_datasets(num_datasets, min_samples, max_samples, min_features, max_features):
    random_data = [[], []]  # Initialize list to hold x and y datasets
    
    i = 0
    samples = [100000, 200000, 300000, 400000, 500000]

    for sample in samples:

        np.random.seed(1 + i)
        i+= 1
        #num_samples = np.random.randint(min_samples, max_samples + 1)
        num_features = np.random.randint(min_features, max_features + 1)
        
        x = np.random.rand(sample, num_features)  # Generate random x data
        y = np.random.randint(-1, 1, size=(sample, 1))  # Generate random y data with values -1 or 1
        y[y == 0] = 1  # Replace 0s with 1s in y
        
        random_data[0].append(x)
        random_data[1].append(y)
    
    return random_data

if __name__ == '__main__':

    """
    # Supposons que x, y et b soient définis
    x = np.random.rand(100000, 30)  # Exemple de données d'entrée
    y = np.random.randint(-1, 1, size=(100000, 1))  # Exemple de données de sortie
    y[y==0] = 1
    """
    
    # Génération des datasets aléatoirement
    num_datasets = 5  # Number of datasets to generate
    min_samples = 200000  # Minimum number of samples per dataset
    max_samples = 500000  # Maximum number of samples per dataset
    min_features = 5  # Minimum number of features per sample
    max_features = 30  # Maximum number of features per sample

    random_data = generate_random_datasets(num_datasets, min_samples, max_samples, min_features, max_features)

    
    real_datasets = [
        [x_train_social_network, x_train_indian_liver_patient, x_train_cancer, x_train_diabete2, x_train_mushroom, x_train_taxi_fare],
        [y_train_social_network, y_train_indian_liver_patient, y_train_cancer, y_train_diabete2, y_train_mushroom, y_train_taxi_fare]
    ]


    #X_train = x_train_taxi_fare
    #y_train = y_train_taxi_fare
    b = 0

    list_exec_time_GD = []
    list_exec_time_NAG = []

    
    liste_lignes_dataset = []
    liste_colonnes_dataset = []
    
    #x_final, b_final = nesterov_optimizer(X_train, y_train, b)

    #for X_train, y_train in zip(random_data[0], random_data[1]):
    for X_train, y_train in zip(real_datasets[0], real_datasets[1]):
        
        ligne, colonne = X_train.shape
        liste_colonnes_dataset.append(colonne)
        liste_lignes_dataset.append(ligne)
        #print(liste_lignes_dataset)
        #print(liste_colonnes_dataset)

        if ligne <= 20000:
            num_workers = 1
        elif ligne <= 50000:
            num_workers = 4
        elif ligne <= 100000:
            num_workers = 6
        elif ligne <= 200000:
            num_workers = 8
        elif ligne <= 300000:
            num_workers = 10
        elif ligne <= 400000:
            num_workers = 12
        elif ligne <= 500000:
            num_workers = 14
        else:
            num_workers = multiprocessing.cpu_count()

        print("Numbre de CPU : ", num_workers)
        # Diviser le dataset en fragments
        X_train_fragments = np.array_split(X_train, num_workers)
        y_train_fragments = np.array_split(y_train, num_workers)

        # Préparez les arguments sous forme de tuples
        tasks = [(X_train_fragments[i], y_train_fragments[i], b) for i in range(num_workers)]

        #______________________METHODE DE NESTEROV________________________

        start_time_NAG = time.time()
        pool = multiprocessing.Pool(processes=num_workers)

        # Utilisez starmap pour désemballer chaque tuple en arguments pour nesterov_optimizer
        results_NAG = pool.starmap(nesterov_optimizer, tasks)

        pool.close()  # Fermez le pool
        pool.join()   # Attendez que tous les processus se terminent

        end_time_NAG = time.time()

        # Initialize the sum of arrays and sum of scalars

        sum_array = np.zeros((colonne, 1))
        sum_scalar = 0.0

        # Iterate through the list and sum the arrays and scalars
        for array, scalar in results_NAG:
            sum_array += array
            sum_scalar += scalar

        results_NAG = [sum_array, sum_scalar]


        #______________________METHODE DE DESCENTE DE GRADIENT________________________

        start_time_GD = time.time()
        pool = multiprocessing.Pool(processes=num_workers)

        # Utilisez starmap pour désemballer chaque tuple en arguments pour nesterov_optimizer
        results_GD = pool.starmap(descente_gradient, tasks)

        pool.close()  # Fermez le pool
        pool.join()   # Attendez que tous les processus se terminent

        end_time_GD = time.time()

        # Initialize the sum of arrays and sum of scalars

        sum_array = np.zeros((colonne, 1))
        sum_scalar = 0.0

        # Iterate through the list and sum the arrays and scalars
        for array, scalar in results_GD:
            sum_array += array
            sum_scalar += scalar

        results_GD = [sum_array, sum_scalar]

        exec_time_GD = round(end_time_GD - start_time_GD, 4)
        exec_time_NAG = round(end_time_NAG - start_time_NAG, 4)

        list_exec_time_GD.append(exec_time_GD)
        list_exec_time_NAG.append(exec_time_NAG)

        #print("________METHODE DE NESTEROV________\n", results_NAG)
        #print("\nTemps d'execution: ", exec_time_NAG)
        #print("\n\n________METHODE DE DESCENTE DE GRADIENT________\n", results_GD)
        #print("\nTemps d'execution: ", exec_time_GD)


    #print("Temps execution GD: ",list_exec_time_GD)
    #print("Temps execution NAG: ",list_exec_time_NAG)
