import numpy as np

def fonction_objective_RegLog(w, x, y, b):
    
    y1 = np.where(y < 0.5, -1, 1)

    # Calcul du terme linéaire
    linear_term = np.dot(x, w) + b
    
    # Calcul du terme d'erreur
    loss_term = np.log(1 + np.exp(-y1 * linear_term))
    
    # Somme des erreurs
    objective_value = (0.5 * np.square(np.linalg.norm(w))) + np.sum(loss_term)
    
    return objective_value

def gradient_RegLog(w, x, y, b):
    
    y1 = np.where(y < 0.5, -1, 1)

    # Calcul du terme linéaire
    linear_term = np.dot(x, w) + b
    
    # Calcul de la partie exponentielle
    exponential_term = np.exp(y1 * linear_term)
    
    # Calcul du coefficient multiplicateur
    multiplier = -y1 / (1 + exponential_term) 
    
    # Calcul du gradient par rapport aux poids w
    gradient_w = w + np.dot(x.T, multiplier)
    
    # Calcul du gradient par rapport au biais b
    gradient_b = np.sum(multiplier)
    
    return gradient_w, gradient_b

def prediction(w, x, b):
    z = np.dot(x.T, w.T) + b
    pred = 1/ (1 + np.exp(-z))
    pred = np.where(pred < 0.5, -1, 1)

    return pred

#Méthode de descente de gradient pour minimiser la fonction objective
def descente_gradient(x, y, b, max_iterations=1000, tol=1e-4, beta=0.5):
    m, n = x.shape
    w = np.zeros((n, 1))
    grad_w, grad_b = gradient_RegLog(w, x, y, b)
    iteration = 0
    for i in range(max_iterations):
        alpha = 1.0  # Initial guess for step size
        j=0
        
        # Recherche linéaire (règle d'Armijo)
        while j < max_iterations:
            if np.all(fonction_objective_RegLog(w - alpha * grad_w, x, y, b - alpha * grad_b) <= fonction_objective_RegLog(w, x, y, b) - beta * alpha * (np.linalg.norm(grad_w)**2 + grad_b**2)):
                break
            alpha = alpha * beta
            j += 1

        # Mise à jour des poids
        w_new = w - alpha * grad_w
        b_new = b - alpha * grad_b
        
        #print(alpha)
        #print(np.linalg.norm(grad_w))
        #print(fonction_objective_RegLog(w, x, y, b, lambda_val))

        w = w_new
        b = b_new
        grad_w, grad_b = gradient_RegLog(w, x, y, b)
        
        obj_value_GD = fonction_objective_RegLog(w, x, y, b)

        iteration += 1
        # Vérification de la convergence
        if np.linalg.norm(grad_w) < tol:
            #print("CIAO GD !!!")
            break
    return w, b

#Méthode de Nesterov pouur minimiser la fonction objective
def nesterov_optimizer(x, y, b, max_iterations=1000, tol=1e-4, beta=0.05):
    m, n = x.shape
    w = np.zeros((n, 1))  # Initialisation des coefficients
    w_prev = np.zeros((n, 1))  # Initialisation des coefficients
    b_prev = 0  # Initialisation des coefficients
    grad_w, grad_b = gradient_RegLog(w, x, y, b)
    iteration = 0
    beta_nesterov = 1

    while iteration < max_iterations:
        # Calcul du pas d'Armijo optimal
        alpha = 1.0
        i = 0
        while i < max_iterations:
            if np.all(fonction_objective_RegLog(w - alpha * grad_w, x, y, b - alpha * grad_b) <= fonction_objective_RegLog(w, x, y, b) - beta * alpha * (np.linalg.norm(grad_w)**2 + grad_b**2)):
                break
            alpha = alpha * beta
            i += 1

        # Mise à jour de Nesterov
        w_next = w - alpha * grad_w
        b_next = b - alpha * grad_b

        #w = w_next + ((iteration - 1) / (iteration+2)) * (w_next - w) #Dependant des iteration
        #b = b_next + ((iteration - 1) / (iteration+2)) * (b_next - b) #Dependant des iteration

        # Mise à jour de Nesterov en utilisant Beta de Nesterov
        
        beta_prev = beta_nesterov
        beta_nesterov = (1 + np.sqrt(1 + 4 * beta_prev**2)) / 2
        w = w_next + ((beta_prev - 1) / beta_nesterov) * (w_next - w_prev)
        b = b_next + ((beta_prev - 1) / beta_nesterov) * (b_next - b_prev)
        
        b_prev = b_next
        w_prev = w_next
        
        # Mise à jour du gradient
        grad_w, grad_b = gradient_RegLog(w, x, y, b)

        #y_pred = prediction(w, x, b)
        #print(np.linalg.norm(grad_w))
        #print(fonction_objective_RegLog(w, x, y, b, lambda_val, class_weights))
        
        obj_value_NAG = fonction_objective_RegLog(w, x, y, b)

        iteration = iteration + 1

        # Condition d'arrêt
        if np.linalg.norm(grad_w) < tol:
            #print("CIAO NAG !!!")
            break

    return w, b
"""
# Supposons que x, y et b soient définis
x = np.random.rand(80000, 30)  # Exemple de données d'entrée
y = np.random.randint(-1, 1, size=(80000, 1))  # Exemple de données de sortie
y[y==0] = 1
b = 0

x_final, b_final = nesterov_optimizer(x, y, b)
"""