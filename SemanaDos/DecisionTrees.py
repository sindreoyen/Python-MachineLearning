from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from itertools import combinations

def subsetDecisionTreeClassifier(X_train, y_train, cs: list) -> DecisionTreeClassifier:
    '''
    - X_train: the training data set features.
    - y_train: the classifications of the training data set.
    - cs: A list of indices representing the features to be included in the model.
    
    returns a trained decision tree classifier that uses a subset of features for training.
    '''
    # Splitting the data 
    # Filtering the cs list to only include valid indices
    cs_train = X_train[:, cs]
    # Training the classifier
    model = DecisionTreeClassifier()
    model.fit(cs_train, y_train)
    return model

def subsetForestClassifierPredict(X_train, y_train, X) -> list:
    '''
    This subsetForestClassifier should take in a training set, split up the training set into different subsets
    and train different decision trees on each subset. To classify a new value, it should use each of the decision trees
    to classify the new value and return the most common classification.
    - X_train: the training data set features.
    - y_train: the classifications of the training data set.
    - X: an array of new values/features to be classified.

    returns an array of new the classifications set by the model
    '''
    # Getting all the possible indices
    n_indices = len(X_train[0])
    # Calculating all different permutations of an array of length indices - 1
    all_combinations = list(list(combinations(range(n_indices), n_indices - 1)))
    # Creating a list of classifiers
    classifiers = [subsetDecisionTreeClassifier(X_train, y_train, cs) for cs in all_combinations]
    # Classifying the new values for each classifier and choosing the most common classification
    all_classifications = [classifiers[i].predict(X[:, all_combinations[i]]) for i in range(len(classifiers))]
    prediction = []
    for i in range(len(X)):
        # Getting the classifications for each classifier
        classifications = [all_classifications[j][i] for j in range(len(all_classifications))]
        # Getting the most common classification
        prediction.append(max(set(classifications), key = classifications.count))
    return prediction

def subsetForestClassifierScore(X_train,y_train,X_test,y_test) -> float:
    '''
    - X_train: the training data set features.
    - y_train: the classifications of the training data set.
    - X_test: the test data set features.
    - y_test: the classifications of the test data set.
    
    returns the score of the model.
    '''
    predictions = subsetForestClassifierPredict(X_train,y_train,X_test)
    # Finding the count of true predictions and dividing by the total number of predictions
    return [predictions[i] == y_test[i] for i in range(len(predictions))].count(True) / len(predictions)

def testRun(dataset):
    # Splitting the data
    X_data, y_data = \
    dataset.data, dataset.target
    X_train, X_test, y_train, y_test = \
    train_test_split(X_data,y_data,test_size = 0.28,random_state=12)
    print("subsetForestClassifier score: " + str(subsetForestClassifierScore(X_train, y_train, X_test, y_test)))
    # Compare to the score of the RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("RandomForestClassifier score: " + str(model.score(X_test, y_test)))

print("\nIris dataset:")
testRun(load_iris())
print("\nBreast cancer dataset:")
testRun(load_breast_cancer())
print("\nWine dataset:")
testRun(load_wine())