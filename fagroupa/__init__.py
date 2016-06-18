import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model, svm
import sklearn.cross_validation as cv
from sklearn.kernel_ridge import KernelRidge
import time

MAX_NUM_ITERATIONS = 10
NUMBER_OF_K_FOLDS = 5


def generate_selected_model(option_of_model, iteration):
    if option_of_model == "1":  # LinearRegression
        name_of_edited_var = "None, no parameters available to iterate"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return linear_model.LinearRegression(n_jobs=-1, normalize=True), value_to_try, name_of_edited_var

    elif option_of_model == "2":  # Ridge
        name_of_edited_var = "alpha"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return linear_model.Ridge(alpha=value_to_try, normalize=True), value_to_try, name_of_edited_var

    elif option_of_model == "3":  # Lasso
        name_of_edited_var = "alpha"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return linear_model.Lasso(alpha=value_to_try, normalize=True, warm_start=True), value_to_try, name_of_edited_var

    elif option_of_model == "4":  # LassoLars
        name_of_edited_var = "alpha"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return linear_model.LassoLars(alpha=value_to_try, normalize=True), value_to_try, name_of_edited_var

    elif option_of_model == "5":  # BayesianRidge
        name_of_edited_var = "n_iter"
        max_of_this_model = 1000
        value_to_try = int((max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS)
        return linear_model.BayesianRidge(n_iter=value_to_try, normalize=True), value_to_try, name_of_edited_var

    elif option_of_model == "6":  # LogisticRegression
        name_of_edited_var = "None, no parameters available to iterate"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return linear_model.LogisticRegression(n_jobs=-1, warm_start=True), value_to_try, name_of_edited_var

    elif option_of_model == "7":  # Perceptron
        name_of_edited_var = "n_iter"
        max_of_this_model = 100
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return linear_model.Perceptron(n_iter=value_to_try,
                                       warm_start=True, n_jobs=-1), value_to_try, name_of_edited_var

    elif option_of_model == "8":  # KernelRidge
        name_of_edited_var = "alpha"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return KernelRidge(alpha=value_to_try), value_to_try, name_of_edited_var

    elif option_of_model == "9":  # SVC
        name_of_edited_var = "poly"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return svm.SVC(kernel='poly'), value_to_try, name_of_edited_var

    elif option_of_model == "10":  # LinearSVC
        name_of_edited_var = "None, no parameters available to iterate"
        max_of_this_model = 1
        value_to_try = (max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS
        return svm.LinearSVC(), value_to_try, name_of_edited_var

    elif option_of_model == "11":  # SVR
        name_of_edited_var = "degree"
        max_of_this_model = 6
        value_to_try = int((max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS)
        return svm.SVR(degree=value_to_try), value_to_try, name_of_edited_var

    elif option_of_model == "12":  # SVC, rbf
        name_of_edited_var = "alpha"
        max_of_this_model = 6
        value_to_try = int((max_of_this_model/MAX_NUM_ITERATIONS)*iteration + max_of_this_model/MAX_NUM_ITERATIONS)
        return svm.SVC(degree=value_to_try, kernel='rbf'), value_to_try, name_of_edited_var

    else:
        print("ERROR: invalid model selected")
        return None


def execute_model(option_of_model):
    # Load the digits database
    digits = datasets.load_digits()

    # Lists in order to store the results in every step
    results_of_cross_validation_on_trining_data = []
    value_tried = []
    stds = []
    time_to_train = []
    graph_description = ""

    # We create a variable i that will be used to test ranges of values
    for iteration in range(MAX_NUM_ITERATIONS):
        # Start the iteration timestamp
        start_time = time.time()

        # We create an instance of our corresponding models
        clf, specific_value_tried, graph_description = generate_selected_model(option_of_model, iteration)

        # We fit the model with the training data
        clf.fit(digits.data, digits.target)

        # We calculate scores for every fold of the CV
        scores = cv.cross_val_score(clf, digits.data, digits.target, cv=NUMBER_OF_K_FOLDS)

        # Print result of current iteration
        print(str(iteration) + ": Accuracy: %0.2f (+/- %0.2f), or not..." % (scores.mean(), scores.std()*2))

        # Store results in arrays to be analyzed later on a plot
        #       Mean score of folds in training data
        results_of_cross_validation_on_trining_data.append(scores.mean())
        #       Standard deviation of scores throughout the folds od training data
        stds.append(scores.std()*2)
        #       Number of n_iter in this iteration
        value_tried.append(specific_value_tried)
        #       Time per iteration
        time_to_train.append(time.time() - start_time)

    # Create the plot figure
    fig, ax1 = plt.subplots()
    # Plot mean score of folds in training data for all iterations
    ax1.plot(value_tried, results_of_cross_validation_on_trining_data, 'b-')
    # Plot mean +/- std of training data
    ax1.plot(value_tried, [(x[0]+x[1]) for x in zip(results_of_cross_validation_on_trining_data, stds)], "r--")
    ax1.plot(value_tried, [(x[0]-x[1]) for x in zip(results_of_cross_validation_on_trining_data, stds)], "r--")
    # Set label titles (Y)
    ax1.set_ylabel('Accuracy')
    # Set label titles (X)
    ax1.set_xlabel(graph_description)
    # Set axis labels to corresponding color
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    # Plot on top of the same plot
    ax2 = ax1.twinx()
    # Plot the time in green
    ax2.plot(value_tried, time_to_train, "g-")
    # Set label titles (Y)
    ax2.set_ylabel('Time to train (seconds)')
    # Set label titles (X)
    ax2.set_xlabel(graph_description)
    # Set axis labels to corresponding color
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
    # Display the plot
    plt.show()


class Run:
    def __init__(self):
        model_selected = ""
        while model_selected != "q":
            print("Please indicate which model you would like to use:")
            print("    1) Generalized Linear Models: Ordinary Least Squares")
            print("    2) Generalized Linear Models: Ridge Regression")
            print("    3) Generalized Linear Models: Lasso")
            print("    4) Generalized Linear Models: LARS Lasso")
            print("    5) Generalized Linear Models: Bayesian Ridge Regression")
            print("    6) Generalized Linear Models: Logistic regression")
            print("    7) Generalized Linear Models: Perceptron")
            print("    8) Kernel ridge regression")
            print("    9) Support Vector Machines: Single-class classification")
            print("    10) Support Vector Machines: Multi-class classification")
            print("    11) Generalized Linear Models: Regression")
            print("    12) Generalized Linear Models: Kernel functions, rbf")

            model_selected = input("Select option(\"q\" to exit): ")

            if model_selected in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]:
                execute_model(model_selected)
            elif model_selected == "q":
                print("Hope it has been helpful, enjoy data science! bye!")
            else:
                print("Invalid option selected, please introduce the number of the option")
