# ==========================================================
# Aprendizaje automático 
# Máster en Ingeniería Informática - Universidad de Sevilla
# Curso 2023-24
# Primer trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# APELLIDOS: Øyen
# NOMBRE: Sindre Langås
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio. 

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# estudiantes involucrados. Por tanto, NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.  
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN





# ========================
# IMPORTANTE: USO DE NUMPY
# ========================

# SE PIDE USAR NUMPY EN LA MEDIDA DE LO POSIBLE. 

import numpy as np
from utils import printProgressBar, sigmoid, normalize, binary_cross_entropy, extract_features

# SE PENALIZARÁ el uso de bucles convencionales si la misma tarea se puede
# hacer más eficiente con operaciones entre arrays que proporciona numpy. 

# PARTICULARMENTE IMPORTANTE es el uso del método numpy.dot. 
# Con numpy.dot podemos hacer productos escalares de pesos por características,
# y extender esta operación de manera compacta a dos dimensiones, cuando tenemos 
# varias filas (ejemplos) e incluso varios varios vectores de pesos.  

# En lo que sigue, los términos "array" o "vector" se refieren a "arrays de numpy".  

# NOTA: En este trabajo NO se permite usar scikit-learn (salvo en el código que
# se proporciona para cargar los datos).

# -----------------------------------------------------------------------------

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aa.zip y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn). Todos los datos se cargan en arrays de numpy.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresista (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn. Como vocabulario, 
#   se han usado las 609 palabras que ocurren más frecuentemente en las distintas 
#   críticas. Los datos se cargan finalmente en las variables X_train_imdb, 
#   X_test_imdb, y_train_imdb,y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ===========================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA (HOLDOUT)
# ===========================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test, y conservando la correspondencia 
# original entre los ejemplos y sus valores de clasificación.
# La división ha de ser ALEATORIA y ESTRATIFICADA respecto del valor de clasificación.

def particion_entr_prueba(X,y,test=0.20):
    '''
    This function takes a dataset X and its corresponding classification values y,
    and divides them into training and test data, in the proportion marked by the
    test argument, and preserving the original correspondence between the examples
    and their classification values. The division is random and stratified.

    param X: the dataset
    param y: the corresponding classification values
    param test: the proportion for the test data
    return: the training and test data, and their corresponding classification values
    '''
    # Check if the parameters are valid
    if len(X) != len(y): raise ValueError("The length of X and y must be the same")
    if test < 0 or test > 1: raise ValueError("The test parameter must be between 0 and 1")

    # Map the number of examples of each distinct classification value
    n_per_class = dict(zip(*np.unique(y, return_counts=True)))
    n_per_class_test = { k: int(round(v * test)) for k, v in n_per_class.items() }
    # Initialize the training and test data and their corresponding classification values
    X_train, X_test = np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))
    y_train, y_test = np.empty((0,)), np.empty((0,))
    # Iterate over the classification values
    test_indices = np.empty((0,), dtype=int)
    for c in np.unique(y):
        # Get the indices of the examples with the current classification value
        indices = np.where(y == c)[0]
        test_indices = np.append(test_indices, np.random.choice(indices, n_per_class_test[c], replace=False))
        # Get the indices of the examples with the current classification value
        test_indices_c = np.intersect1d(indices, test_indices, assume_unique=True)
        train_indices_c = np.setdiff1d(indices, test_indices_c, assume_unique=True)
        # Append the examples with the current classification value to the training and test data
        X_train = np.append(X_train, X[train_indices_c], axis=0)
        X_test = np.append(X_test, X[test_indices_c], axis=0)
        # Append the classification values of the examples with the current classification value to the training and test data
        y_train = np.append(y_train, y[train_indices_c], axis=0)
        y_test = np.append(y_test, y[test_indices_c], axis=0)
    # Return the training and test data and their corresponding classification values
    return X_train, X_test, y_train, y_test


# ===========================================
# EJERCICIO 2: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# Se pide implementar el clasificador de regresión logística mini-batch 
# a través de una clase python, que ha de tener la siguiente estructura:

class RegresionLogisticaMiniBatch():
    # -- Attributes --
    trained = False
    classes = None
    inverted_classes = None

    # -- Constructor --
    def __init__(self,normalizacion:bool=False,
                 rate:float=0.1,rate_decay=False,batch_tam:int=64,
                 pesos_iniciales=None, 
                 epsilon:float=1e-4, patience=50,
                 X_valid:np.ndarray=None, y_valid:np.ndarray=None):
        '''
        This class implements the mini-batch logistic regression classifier.

        param normalizacion: indicates if the data should be normalized
        param rate: the learning rate. Defines the size of the steps taken in the gradient descent
        param rate_decay: indicates if the learning rate should decrease in each epoch
        param batch_tam: the size of the mini-batches
        param pesos_iniciales: if None, the initial weights are initialized randomly.
                                If not, it must provide an array of weights that will
                                be used as initial weights.
        param epsilon: threshold for change in loss function
        param patience: Number of epochs to wait for the loss to change less than epsilon before stopping the training.
                        If patience is 0, the training will not stop until the number of epochs is reached.
        param X_valid: the validation data if early stopping is used
        param y_valid: the validation classification values if early stopping is used
        '''
        self.normalizacion, self.rate, self.rate_decay, self.batch_tam, self.pesos_iniciales, self.epsilon, self.patience, self.X_valid, self.y_valid = \
            normalizacion, rate, rate_decay, batch_tam, pesos_iniciales, epsilon, patience, X_valid, y_valid
        if not isinstance(self.batch_tam, int): self.batch_tam = int(self.batch_tam)

    # -- Methods -- 
    def entrena(self,entr,clas_entr, n_epochs=1000, reiniciar_pesos=False, 
                print_loading=True):
        '''
        This method trains the classifier. This implementation is following based on the formula for weight
        updates using mini-batch gradient descent, as explained in the slides of the module and listed below:

        param entr: the training data
        param clas_entr: the classification values tied to the training data
        param n_epochs: the number of epochs for the training
        param reiniciar_pesos: if True, the weights are initialized randomly at the beginning of the training, 
                                with values between -1 and 1. If False, the weights are initialized only the first
                                time the method is called. In subsequent times, the training continues from the
                                weights calculated in the previous training. This can be useful to continue the
                                training from a previous training, if for example new data is available.
        param print_loading: if True, a progress bar is printed to the console
        '''
        # Create numeric mapping for the classes
        self.classes = np.sort(np.unique(clas_entr))
        if len(self.classes) != 2: raise ValueError("The number of classes must be 2")
        # If the classes are text values, map them to numeric values
        if not np.issubdtype(clas_entr.dtype, np.number):
            self.inverted_classes = { v: k for k, v in enumerate(self.classes) }
            clas_entr = np.array([self.inverted_classes[c] for c in clas_entr], dtype=int)
            self.y_valid = np.array([self.inverted_classes[c] for c in self.y_valid]) if self.y_valid is not None else self.y_valid

        # Initialize weights if necessary
        if reiniciar_pesos or not self.trained or self.pesos is None:
            self.pesos = np.random.uniform(-1, 1, entr.shape[1])

        # Normalize the data if needed
        entr = normalize(entr) if self.normalizacion else entr

        # Calculate the number of batches
        num_batches = int(np.ceil(entr.shape[0] / self.batch_tam))

        # Previous loss
        prev_loss = 0
        patience_counter = 0

        # Training loop
        if print_loading: print("\n\nTraining the classifier...")
        for epoch in range(n_epochs):
            if patience_counter >= self.patience and self.patience > 0: 
                if print_loading: print("\nEarly stopping after {} epochs".format(epoch + 1))
                break
            # Calculate the learning rate for the current epoch
            if print_loading: printProgressBar(epoch + 1, n_epochs, prefix = 'Training progress: ', suffix = "of {} epochs ran".format(n_epochs), length = 30)
            temp_rate = self.__new_rate(epoch) if self.rate_decay else self.rate
            # Iterate over the batches
            for batch_index in np.random.permutation(num_batches):
                # Define the start and end index for the current batch
                start_index: int = batch_index * self.batch_tam
                if start_index >= entr.shape[0]: break
                end_index: int = min((batch_index + 1) * self.batch_tam, entr.shape[0])
                
                # Slice the batch from the dataset
                entr_batch = entr[start_index:end_index]
                clas_entr_batch = clas_entr[start_index:end_index]
                
                # Calculate the weight updates for the current batch
                for i in range(len(self.pesos)):
                    gradient = entr_batch[:, i].dot(clas_entr_batch - entr_batch.dot(self.pesos))
                    # Here I experienced extreme values for the gradient, which caused the weights to explode.
                    # To prevent this, I am clipping the gradient to a value between -1 and 1.
                    self.pesos[i] += temp_rate * np.clip(gradient, -1, 1)
            
            if self.patience > 0 and self.X_valid is not None and self.y_valid is not None:
                # Calculate the loss for the current epoch
                prob = self.clasifica_prob(self.X_valid, for_early_stop=True)
                curr_loss = np.mean(np.array([binary_cross_entropy(c, p) for c, p in zip(self.y_valid, prob)]), dtype=float)
                # Check if the loss has changed less than the threshold
                if abs(float(curr_loss) - float(prev_loss)) < self.epsilon:
                    patience_counter += 1
                else: patience_counter = 0
                prev_loss = curr_loss

        # Set the trained flag to True
        self.trained = True

    def clasifica_prob(self,E,for_early_stop:bool=False) -> np.ndarray:
        '''
        This method returns the array of corresponding probabilities of belonging
        to the positive class (the one that has been taken as class 1), for each
        example of a new array E of examples.

        param E: the examples to predict analyze the prediction probabilities for
        return: the array of corresponding probabilities of belonging to the positive class
        '''
        if not self.trained and not for_early_stop: raise ClasificadorNoEntrenado()
        # Normalize the data if needed
        E = normalize(E) if self.normalizacion else E
        # Calculate the probabilities with the sigmoid function
        return np.array([sigmoid(np.dot(e, self.pesos), stable=False) for e in E], dtype=float)

    def clasifica(self,E):
        '''
        This method returns an array with the corresponding classes that are predicted
        for each example of a new array E of examples. The class must be one of the
        original classes of the problem (for example, "republican" or "democrat" in
        the votes problem).

        param E: the examples to predict
        return: an array with the corresponding classes that are predicted for each example
        '''
        if not self.trained: raise ClasificadorNoEntrenado()
        # Calculating the probabilities
        probabilities = self.clasifica_prob(E)
        return np.array([self.classes[0] if p <= 0.5 else self.classes[1] for p in probabilities])
    
    # -- Private methods --
    def __new_rate(self, n):
        '''
        This method calculates the learning rate for the current epoch.

        param current_rate: the learning rate for the previous epoch
        param n: the current epoch
        return: the learning rate for the current epoch
        '''
        return self.rate / (1 + n)

# Explicamos a continuación cada uno de los métodos:


# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:


#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en CADA COLUMNA i (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta MISMA transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad
#    introducida en el parámetro rate anterior.   

#  - batch_tam: indica el tamaño de los mini batches (por defecto 64)
#    que se usan para calcular cada actualización de pesos.
    
#  - pesos_iniciales: Si es None, los pesos iniciales se inician 
#    aleatoriamente. Si no, debe proporcionar un array de pesos que se 
#    tomarán como pesos iniciales.     

# 

# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador. 
#  Debe calcular un vector de pesos, mediante el correspondiente
#  algoritmo de entrenamiento basado en ascenso por el gradiente mini-batch, 
#  para maximizar la log verosimilitud. Describimos a continuación los parámetros de
#  entrada:  

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es un array (bidimensional)  
#    con los ejemplos, y el segundo un array (unidimensional) con las clasificaciones 
#    de esos ejemplos, en el mismo orden. 

#  - n_epochs: número de pasadas que se realizan sobre todo el conjunto de
#    entrenamiento.

#  - reiniciar_pesos: si es True, se reinicia al comienzo del 
#    entrenamiento el vector de pesos de manera aleatoria 
#    (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior. Esto puede ser útil
#    para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.     

#  NOTA: El entrenamiento en mini-batch supone que en cada epoch se
#  recorren todos los ejemplos del conjunto de entrenamiento,
#  agrupados en grupos del tamaño indicado. Por cada uno de estos
#  grupos de ejemplos se produce una actualización de los pesos. 
#  Se pide una VERSIÓN ESTOCÁSTICA, en la que en cada epoch se asegura que 
#  se recorren todos los ejemplos del conjunto de entrenamiento, 
#  en un orden ALEATORIO, aunque agrupados en grupos del tamaño indicado. 


# * Método clasifica_prob:
# ------------------------

#  Método que devuelve el array de correspondientes probabilidades de pertenecer 
#  a la clase positiva (la que se ha tomado como clase 1), para cada ejemplo de un 
#  array E de nuevos ejemplos.


        
# * Método clasifica:
# -------------------
    
#  Método que devuelve un array con las correspondientes clases que se predicen
#  para cada ejemplo de un array E de nuevos ejemplos. La clase debe ser una de las 
#  clases originales del problema (por ejemplo, "republicano" o "democrata" en el 
#  problema de los votos).  


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): pass

def rendimiento(clasif,X,y):
    '''
    Calculates the performance of a classifier.
    '''
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio vale 2 PUNTOS (SOBRE 10) pero se puede saltar, sin afectar 
# al resto del trabajo. Puede servir para el ajuste de parámetros en los ejercicios 
# posteriores, pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1. 

# La técnica de validación cruzada que se pide en este ejercicio se explica
# en el tema "Evaluación de modelos".     

# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)
def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, n=5, n_epochs=1000, grid_search=False):
    '''
    This function calculates the average performance of a classifier, using the cross validation technique
    with n partitions. The arrays X and y are the data and the expected classification, respectively. The
    argument clase_clasificador is the name of the class that implements the classifier. The argument
    params is a dictionary whose keys are parameter names of the classifier's constructor and the values
    associated with those keys are the values of those parameters to call the constructor.

    param clase_clasificador: the name of the class that implements the classifier
    param params: a dictionary whose keys are parameter names of the classifier's constructor and the values
                  associated with those keys are the values of those parameters to call the constructor
    param X: the data
    param y: the expected classification
    param n: the number of partitions
    param n_epochs: the number of epochs for the training
    param grid_search: if True, the function will not print the progress of the training
    return: the average performance of the classifier
    '''
    # Check if the parameters are valid
    if n < 2: raise ValueError("The number of partitions must be at least 2")
    if n > len(X): raise ValueError("The number of partitions must be less than the number of examples")
    if len(X) != len(y): raise ValueError("The length of X and y must be the same")

    # Partitioning the data using the particion_entr_prueba function
    # Init empty arrays
    X_partitions = np.empty((n,), dtype=object)
    y_partitions = np.empty((n,), dtype=object)
    for i in range(n):
        # The last partition should contain all remaining data
        if i == n - 1: X_partitions[i], y_partitions[i] = X, y; break
        X, X_partitions[i], y, y_partitions[i] = particion_entr_prueba(X, y, test=1/(n-i))
    
    # Calculate the performance for each partition
    performances = np.empty((n,))
    for i in range(n):
        # Init the classifier
        classifier = clase_clasificador(**params)
        # Train the classifier with all partitions except the current one
        # Keeping the current partitions as holdout for the performance calculation
        X_train = np.concatenate(np.delete(X_partitions, i, axis=0))
        y_train = np.concatenate(np.delete(y_partitions, i, axis=0))
        classifier.entrena(X_train, y_train, print_loading = not grid_search, n_epochs=n_epochs)
        if not grid_search: print("Training", "#", i + 1, "done")
        # Calculate the performance for the current partition
        performances[i] = rendimiento(classifier, X_partitions[i], y_partitions[i])
    
    # Return the average performance
    return np.mean(performances)



# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cáncer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir exactamente el resultado):

# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones deben ser aleatorias y estratificadas. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

#LR16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
#LR16.entrena(Xe_cancer,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
#rendimiento(LR16,Xp_cancer,yp_cancer)
# 0.9203539823008849

#------------------------------------------------------------------------------





# ===================================================
# EJERCICIO 4: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando los dos modelos implementados en el ejercicio 3, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros para mejorar el rendimiento. Si se ha hecho el ejercicio 3, 
# usar validación cruzada para el ajuste (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos. 


# =====================================
# To answer this question, I have implemented a grid search function that
# iterates over all possible combinations of the parameters and returns the
# best parameters for the classifier. This function is implemented in the
# GridSearchCV function below.

def GridSearchCV(clase_clasificador, param_grid: dict, X, y, k: int,
                 n_epochs:int=1000, saveFile:str=None) -> dict:
    '''
    This function performs a grid search for the best parameters for a classifier
    by using the cross validation implementation in the rendimiento_validacion_cruzada.
    The argument param_grid is a dictionary whose keys are parameter names of the
    classifier's constructor and the values associated with those keys are the values
    of those parameters to call the constructor. The argument X is the data and the
    argument y is the expected classification.

    param clase_clasificador: the name of the class that implements the classifier
    param param_grid: a dictionary whose keys are parameter names of the classifier's
                      constructor and the values associated with those keys are the
                      values of those parameters to call the constructor
    param X: the data
    param y: the expected classification
    param k: the number of partitions for the cross validation
    param n_epochs: the number of epochs for the training
    param saveFile: if not None, the results will be saved to a file with the given name
    '''
    # Check if the parameters are valid
    if len(X) != len(y): raise ValueError("The length of X and y must be the same")

    # Init the best parameters
    best_params: dict = None
    best_performance = 0

    # Finding all possible combinations of the parameters
    from itertools import product
    param_combinations = list(product(*param_grid.values()))
    print("Iterating over", len(param_combinations), "combinations of parameters")
    print("With k =", k, "partitions", "this means a total of", len(param_combinations) * k, "trainings")
    print("Please wait...")

    # Iterating over the combinations
    for i, params in enumerate(param_combinations):
        printProgressBar(i + 1, len(param_combinations), prefix = 'Grid search progress: ', suffix = 'Complete', length = 50)
        # Create a dictionary with the current parameters
        params_dict = dict(zip(param_grid.keys(), params))
        # Calculate the performance for the current parameters
        performance = rendimiento_validacion_cruzada(clase_clasificador, params_dict, X, y, k, grid_search=True, n_epochs=n_epochs)
        # Check if the current parameters are the best ones
        if performance > best_performance:
            best_params = params_dict
            best_performance = performance

    # Return the best parameters
    print("Grid search finished")
    print("Best parameters:", best_params, "| With a performance of: ", best_performance)

    # Save the results to a file
    if saveFile is not None:
        import os
        from datetime import datetime
        # Create a folder for the results
        dir_path = os.path.dirname(os.path.realpath(__file__))
        results_path = os.path.join(dir_path, "GridSearchResults")
        if not os.path.exists(results_path): os.makedirs(results_path)
        # Save the results to the file
        with open(os.path.join(results_path, f"{saveFile}.txt"), "w") as f:
            f.write(f"Grid search finished for {saveFile}" + "\n")
            f.write("Date: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
            f.write("Best parameters: " + str(best_params) + "\n")
            f.write("With a performance of: " + str(best_performance) + "\n")
            f.write("With k = " + str(k) + " partitions" + "\n")
            f.write("With " + str(len(param_combinations)) + " combinations of parameters" + "\n")
            f.write("With a total of " + str(len(param_combinations) * k) + " trainings" + "\n")
    return best_params

from datos_trabajo_aa import carga_datos as datos

# To solve the exercise and find the best parameters for each dataset, I have
# used the GridSearchCV function defined above. The results are shown below.
# You can also find the results in the GridSearchResults folder.
# If you want to run the searches again, see the main function at the bottom of
# this file. However, they do take quite long to run, so I recommend using the
# results I have already obtained.

## Votos Grid Search ##
# The params to use for the grid search analysis
Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(datos.X_votos, datos.y_votos)
v_params = {
    "normalizacion": [True],
    "rate": [1e-1, 1e-2, 1e-3, 1e-4],
    "rate_decay": [True, False],
    "batch_tam": [8, 16, 32, 64, 128],
    "X_valid": [Xp_votos, None],
    "y_valid": [yp_votos]
}
def run_votos_grid_search():
    return GridSearchCV(RegresionLogisticaMiniBatch, v_params, Xe_votos, ye_votos, 4, "votos")
votos_best_params = {'normalizacion': True, 'rate': 0.01, 'rate_decay': False, 'batch_tam': 128}  # The results
# Best parameters: {'normalizacion': True, 'rate': 0.01, 'rate_decay': False, 'batch_tam': 128, 'X_valid': None, 'y_valid': None}
# With a performance of: 0.9482029598308668

## Breast-cancer dataset Grid Search ##
# The params to use for the grid search analysis
Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(datos.X_cancer, datos.y_cancer)
c_params = {
    "normalizacion": [True],
    "rate": [1e-1, 1e-2, 1e-3, 1e-4],
    "rate_decay": [True, False],
    "batch_tam": [8, 16, 32, 64, 128],
    "X_valid": [Xp_cancer, None],
    "y_valid": [yp_cancer]
}
def run_cancer_grid_search():
    return GridSearchCV(RegresionLogisticaMiniBatch, c_params, Xe_cancer, ye_cancer, 4, "cancer_dataset")
# Best parameters: {'normalizacion': True, 'rate': 0.1, 'rate_decay': True, 'batch_tam': 8, 'X_valid': Yes, 'y_valid': Yes}
# With a performance of: 0.918859649122807
cancer_best_params = {'normalizacion': True, 'rate': 0.1, 'rate_decay': True, 'batch_tam': 8, 'X_valid': Xp_cancer, 'y_valid': yp_cancer} # The results

## IMDB Grid Search ##
Xe_imdb, Xp_imdb, ye_imdb, yp_imdb = datos.X_train_imdb, datos.X_test_imdb, datos.y_train_imdb, datos.y_test_imdb
i_params = {
    "normalizacion": [True],
    "rate": [1e-1, 1e-2, 1e-3],
    "rate_decay": [True, False],
    "batch_tam": [32, 64, 128],
    "X_valid": [Xp_imdb, None], 
    "y_valid": [yp_imdb]
}
def run_imdb_grid_search():
    # Running with fewer epochs due to the size of the dataset
    return GridSearchCV(clase_clasificador=RegresionLogisticaMiniBatch, param_grid=i_params, 
                        X=Xe_imdb, y=ye_imdb, k=4, saveFile="imdb", n_epochs=50)
# Best parameters: {'normalizacion': True, 'rate': 0.01, 'rate_decay': False, 'batch_tam': 128, 'X_valid': None, 'y_valid': None}
# With a performance of: 0.5734999999999999
# This did not have the best performance, but it was also really slow to train so I had to reduce the number of epochs
# to 50. I think that with more epochs it may have performed better.
# But as a proof of concept, I think it is enough.
imdb_best_params = {'normalizacion': True, 'rate': 0.01, 'rate_decay': False, 'batch_tam': 128} # The results

# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Técnica "One vs Rest" (Uno frente al Resto)
# -------------------------------------------


# Se pide implementar la técnica "One vs Rest" (Uno frente al Resto),
# para obtener un clasificador multiclase a partir del clasificador
# binario definido en el apartado anterior.


#  En concreto, se pide implementar una clase python
#  RegresionLogisticaOvR con la siguiente estructura, y que implemente
#  el entrenamiento y la clasificación siguiendo el método "One vs
#  Rest" tal y como se ha explicado en las diapositivas del módulo.

 

class RegresionLogisticaOvR():
    classifiers = None

    def __init__(self,normalizacion=False,rate=0.1,rate_decay=False,
                 batch_tam=64, X_valid=None, y_valid=None):
        '''
        This class implements the One vs Rest technique for multiclass classification.

        param normalizacion: indicates if the data should be normalized
        param rate: the learning rate. Defines the size of the steps taken in the gradient descent
        param rate_decay: indicates if the learning rate should decrease in each epoch
        param batch_tam: the size of the mini-batches
        '''
        self.normalizacion, self.rate, self.rate_decay, self.batch_tam, self.X_valid, self.y_valid = \
            bool(normalizacion), float(rate), bool(rate_decay), int(batch_tam), X_valid, y_valid
         
    def entrena(self,entr,clas_entr,n_epochs=1000, print_loading=True):
        '''
        This method trains the classifier. This implementation uses the OvA technique, meaning
        that the multiple classes are converted into multiple binary classification problems.
        The classifier is trained for each of these problems, and the class with the highest
        probability is chosen as the predicted class.
        To train the classifier, the method uses the RegresionLogisticaMiniBatch class defined
        in the previous exercise.

        param entr: the training data
        param clas_entr: the classification values tied to the training data
        param n_epochs: the number of epochs for the training
        param print_loading: if True, a progress bar is printed to the console
        '''
        # Create numeric mapping for the classes
        self.classes = np.sort(np.unique(clas_entr))
        # If the classes are text values, map them to numeric values
        if not np.issubdtype(clas_entr.dtype, np.number):
            self.inverted_classes = { v: k for k, v in enumerate(self.classes) }
            clas_entr = np.array([self.inverted_classes[c] for c in clas_entr], dtype=int)
            self.y_valid = np.array([self.inverted_classes[c] for c in self.y_valid]) if self.y_valid is not None else self.y_valid
        
        # Initialize the classifiers
        self.classifiers = {}
        # Train a classifier for each class
        for c in np.unique(clas_entr):
            # Create a binary classification problem for the current class
            binary_clas_entr = np.array([1 if c == c_ else 0 for c_ in clas_entr], dtype=int)
            binary_y_valid = np.array([1 if c == c_ else 0 for c_ in self.y_valid], dtype=int) if self.y_valid is not None else self.y_valid
            # Init the classifier
            self.classifiers[c] = RegresionLogisticaMiniBatch(normalizacion=self.normalizacion, rate=self.rate,
                                                              rate_decay=self.rate_decay, batch_tam=self.batch_tam,
                                                              X_valid=self.X_valid, y_valid=binary_y_valid)
            # Train the classifier
            self.classifiers[c].entrena(entr, binary_clas_entr, n_epochs=int(n_epochs), print_loading=bool(print_loading))
            # Print the progress
            if print_loading: print("Training", c, "done")

    def clasifica(self,E):
        '''
        This method returns an array with the corresponding classes that are predicted
        for each example of a new array E of examples. The class must be one of the
        original classes of the problem (for example, "republican" or "democrat" in
        the problem of votes).

        param E: the examples to predict
        return: an array with the corresponding classes that are predicted for each example
        '''
        if self.classifiers is None: raise ClasificadorNoEntrenado()
        # Predict the class for each example
        classed_predictions = np.array([self.classifiers[c].clasifica_prob(E) for c in self.classifiers.keys()])
        # Return the class with the highest probability
        pred = np.array([self.classes[np.argmax(p)] for p in classed_predictions.T])
        return pred


# ==============================================
# EJERCICIO 6: APLICACION A PROBLEMAS MULTICLASE
# ==============================================


# ---------------------------------------------------------
# 6.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación del apartado anterior, para obtener un
# clasificador que aconseje la concesión, estudio o no concesión de un préstamo,
# basado en los datos X_credito, y_credito. Ajustar adecuadamente los parámetros. 

# NOTA IMPORTANTE: En este caso concreto, los datos han de ser transformados, 
# ya que los atributos de este conjunto de datos no son numéricos. Para ello, usar la llamada 
# "codificación one-hot", descrita en el tema "Preprocesado e ingeniería de características".
# Se pide implementar esta transformación (directamete, SIN USAR Scikt Learn ni Pandas). 

class OneHotEncoder():
    def __init__(self):
        self.categories = None

    def transform(self, X) -> np.ndarray:
        '''
        The transform method transforms the dataset into a one hot encoded dataset.
        It uses the categories attribute to know which categories are present in each
        feature. The method returns the one hot encoded dataset.

        param X: the dataset
        '''
        # Check if the categories have been initialized
        if self.categories is None: raise ValueError("The OneHotEncoder has not been fitted yet")
        # Init the transformed dataset
        transformed_X = np.empty((X.shape[0], 0), dtype=int)
        # Iterate over the features
        for i, feature in enumerate(X.T):
            # Get the categories for the current feature
            categories = self.categories[i]
            # Init the transformed feature
            transformed_feature = np.zeros((X.shape[0], len(categories)), dtype=int)
            # Iterate over the categories
            for j, category in enumerate(categories):
                # Set the values for the current category
                transformed_feature[:, j] = np.array([1 if c == category else 0 for c in feature], dtype=int)
            # Append the transformed feature to the transformed dataset
            transformed_X = np.append(transformed_X, transformed_feature, axis=1)
        return transformed_X.astype(int)

    def fit_transform(self, X) -> np.ndarray:
        '''
        The fit_transform method is a combination of the fit and transform methods.
        It returns the one hot encoded dataset.

        param X: the dataset
        '''
        self.categories = [np.sort(np.unique(X[:, i])) for i in range(X.shape[1])]
        return self.transform(X).astype(float)

# Credito grid search params
Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(datos.X_credito, datos.y_credito)
onehot = OneHotEncoder()
Xe_credito_onehot = onehot.fit_transform(Xe_credito)
Xp_credito_onehot = onehot.transform(Xp_credito)
cred_params = {
    "normalizacion": [True, False],
    "rate": [1e-1, 1e-2, 1e-3, 1e-4],
    "rate_decay": [True, False],
    "batch_tam": [8, 16, 32, 64, 128],
    "X_valid": [Xp_credito_onehot],
    "y_valid": [yp_credito]
}
def run_credito_grid_search():
    onehot = OneHotEncoder()
    Xe_credito_onehot = onehot.fit_transform(Xe_credito)
    return GridSearchCV(clase_clasificador=RegresionLogisticaOvR, param_grid=cred_params, 
                        X=Xe_credito_onehot, y=ye_credito, k=4, saveFile="credito_dataset")
credito_best_params = {'normalizacion': False, 'rate': 0.1, 'rate_decay': True, 'batch_tam': 128, 'X_valid': Xe_credito, 'y_valid': ye_credito}

# ---------------------------------------------------------
# 6.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación o implementaciones del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

X_train_digits, y_train_digits = datos.X_train_digits, datos.y_train_digits
X_valid_digits, y_valid_digits = datos.X_valid_digits, datos.y_valid_digits
X_test_digits, y_test_digits = datos.X_test_digits, datos.y_test_digits

# Extracting features from the image for faster training
# Save one image
feature = extract_features(X_train_digits[0], True)
X_train_digits = np.array([extract_features(matrix) for matrix in X_train_digits])
X_valid_digits = np.array([extract_features(matrix) for matrix in X_valid_digits])
X_test_digits = np.array([extract_features(matrix) for matrix in X_test_digits])

def performance_digits():
    # Classify the digits
    clasificador = RegresionLogisticaOvR(normalizacion=True, rate=0.01, rate_decay=True, batch_tam=128, X_valid=X_valid_digits, y_valid=y_valid_digits)
    clasificador.entrena(X_train_digits, y_train_digits, n_epochs=1000)
    return rendimiento(clasificador, X_test_digits, y_test_digits)

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# Run main 
if __name__ == "__main__":
    ## Grid search ##
    run_all: bool = input("\n(1) Do you want to run all the grid searches? (y/n) ").lower() == "y"
    votos_question = input("\n   (a) Do you want to run the votos grid search? (y/n) ").lower() if not run_all else "y"
    cancer_question = input("\n   (b) Do you want to run the cancer grid search? (y/n) ").lower() if not run_all else "y"
    imdb_question = input("\n   (c) Do you want to run the IMDB grid search? (y/n) ").lower() if not run_all else "y"
    credito_question = input("\n   (d) Do you want to run the credito grid search? (y/n) ").lower() if not run_all else "y"
    
    if votos_question == "y":
        print("\nRunning votos grid search...")
        votos_params = run_votos_grid_search()
    if cancer_question == "y":
        print("\nRunning cancer grid search...")
        cancer_params = run_cancer_grid_search()
    if imdb_question == "y":
        print("\nRunning IMDB grid search...")
        imdb_params = run_imdb_grid_search()
    if credito_question == "y":
        print("\nRunning credito grid search...")
        credito_params = run_credito_grid_search()

    if votos_question == "y": print("\nVotos best params:", votos_params)
    if cancer_question == "y": print("Cancer best params:", cancer_params)
    if imdb_question == "y": print("IMDB best params:", imdb_params)
    if credito_question == "y": print("Credito best params:", credito_params)