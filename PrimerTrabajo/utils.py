# Print iterations progress
# Source: (https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters, accessed 28.11.23)
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


# ----------------------------------------------------------------
# Utils for ML
import numpy as np

def normalize(X):
    '''
    Normalizes the data in X.

    param X: the data to normalize
    return: the normalized data
    '''
    # Calculate the mean and standard deviation for each column
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Normalize the data
    return (X - mean) / std

def binary_cross_entropy(y, p):
    '''
    This method calculates the binary cross entropy for a given target value y and prediction probability p.

    param y: the target value
    param p: the prediction probability
    '''
    # Check for invalid values
    return -y * np.log(p) - (1 - y) * np.log(1 - p) if p != 0 and p != 1 else 0

def sigmoid(x, stable=False) -> float:
    '''
    This method calculates the sigmoid function for a value x.

    param x: the value to calculate the sigmoid function for
    return: the sigmoid of the value
    '''
    return float(1 / (1 + np.exp(-x))) if not stable else __stable_sigmoid(x)
def __stable_sigmoid(x) -> float:
    '''
    Numerically stable sigmoid function.
    
    This method was created to prevent overflow and underflow errors when calculating the sigmoid function.
    Thus, I researched how to calculate the sigmoid function in a numerically stable way and found the
    following article by Tim Vieira:
    - (Vieira, 2014, "A Numerically Stable Way to Compute the Sigmoid Function", https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        - Accessed on 28.11.23
    
    param x: the value to calculate the sigmoid for
    return: the sigmoid of the value
    '''
    if x >= 0: return float(1 / (1 + np.exp(-x)))
    else: float(np.exp(x) / (1 + np.exp(x)))