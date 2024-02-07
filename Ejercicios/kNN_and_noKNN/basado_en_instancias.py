from numpy import unique, sum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler

def kNNBestKScore(T_k, X_train, y_train, X_test, y_test) -> (int, float):
    '''
    - T_k: the maximum number of neighbors to consider.
    - X_train: the training data set features.
    - y_train: the classifications of the training data set.
    - X_test: the test data set features.
    - y_test: the classifications of the test data set.

    returns a tuple (maxK, maxS) where maxK is the number of neighbors between 1 and 'T_k' that produces the highest score with the kNN-classifier, and maxS is the score for maxK.
    '''
    # Setting the initial values for maxK and maxS
    maxK = 0
    maxS = 0.0
    # Looping through the number of neighbors
    for k in range(1, T_k + 1):
        # Using the standard Euclidean distance (p=2 by default)
        model = KNeighborsClassifier(n_neighbors=k,weights='distance')
        # Training and scoring the model, then checking if the current score is greater than the max score
        # Using training data to fit and test data to score
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > maxS:
            maxK = k
            maxS = score
    return (maxK, maxS)

def nokNNClassifierPredict(T_k,X_train,y_train,X) -> list:
    '''
    Implementation of a no-k-NN classifier. An implementation of kNN without the relying on k.

    - T_k: the maximum number of neighbors to consider.
    - X_train: the training data set features.
    - y_train: the classifications of the training data set.
    - X: an array of new values/features to be classified.

    Returns X: an array of new the classifications set by the model
    '''
    # In this task, I am following a 5 step process, based on the notes in the jupyter notebook. 
    # 1) Defining initial values (N, N_c, M_c), 
    # 2) Getting the neighborhood sets A_k, 
    # 3) Calculating the mass functions for each neigborhood for each classification, 
    # 4) Calculating the probability function, 
    # 5) Calculating the decisions

    regulizer = StandardScaler().fit(X_train)
    Xn_train = regulizer.transform(X_train)
    Xn = regulizer.transform(X)
    classes = unique(y_train)

    # 1 Defining initial values
    N = len(Xn_train)
    ## Getting the count of all individual classifications
    N_c = [len(y_train[y_train == c]) for c in classes]
    M_c = [(2*N-1)*(N_c[i]/N) for i in range(len(N_c))]
    
    # 2 Getting the neighborhood sets using the KNeighborsClassifier
    knarray = []
    for k in range(1, T_k + 1):
        # Using the standard Euclidean distance (p=2 by default)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(Xn_train, y_train)
        # Getting the indices of the k nearest neighbors
        kneigh = model.kneighbors(Xn, return_distance=False)
        #print("k" + str(k) + ": " + str(kneigh))
        knarray.append(kneigh)
    A_k = [y_train[j] for j in knarray]
    
    # 3 Calculating the mass functions for each neigborhood for each classification, using the formula in the book
    def m_c(x_indice, k_indice, c) -> float:
        c_count = len(A_k[k_indice - 1][x_indice][A_k[k_indice - 1][x_indice] == c])
        return float(c_count/(M_c[c]*k))
    
    def predictX(x_indice) -> int:
        # 4 Calculating the probability function
        ## Calculating probability for each class
        probabilities = []
        for c in classes:
            massFuncs = [m_c(x_indice, k, c)/k for k in range(1, T_k + 1)]
            probabilities.append(sum(massFuncs))
        # 5 Calculating the decisions
        ## Getting the index of the maximum probability
        return probabilities.index(max(probabilities))
    
    return [predictX(i) for i in range(len(X))]

def nokNNClassifierScore(T_k, X_train, y_train, X_test, y_test) -> float:
    '''
    - T_k: the maximum number of neighbors to consider.
    - X_train: the training data set features.
    - y_train: the classifications of the training data set.
    - X_test: the test data set features.
    - y_test: the classifications of the test data set.

    returns the score of the nokNNClassifierPredict function.
    '''
    results = nokNNClassifierPredict(T_k, X_train, y_train, X_test) == y_test
    return len(results[results == True])/len(results)

# Testing the nokNNClassifierPredict function
def run(dataset, T_k: int = 10, rand_state: int = 1) -> float:
    '''
    Trains and tests the nokNNClassifierPredict function 
    '''
    X_data, y_data, X_names, y_names = \
    dataset.data, dataset.target, dataset.feature_names, dataset.target_names
    
    X_train, X_test, y_train, y_test = \
    train_test_split(X_data,y_data,test_size = 0.4,
                   random_state=rand_state,stratify=y_data)
    # Running the nokNNClassifierPredict function
    score = nokNNClassifierScore(T_k, X_train, y_train, X_test, y_test)
    return score
    
def run_all() -> (float, float, float):
    '''
    Runs the nokNNClassifierPredict function on the iris, breast cancer and wine datasets
    '''
    iris = run(load_iris(), 10, 10)
    breast = run(load_breast_cancer(), 7, 1)
    wine = run(load_wine(), 1, 33)
    print("iris: " + str(iris))
    print("breast: " + str(breast))
    print("wine: " + str(wine))
    return (iris, breast, wine)

# Running the predictions on the data sets
run_all()