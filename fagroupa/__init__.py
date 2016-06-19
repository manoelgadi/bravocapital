import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model, svm
import sklearn.cross_validation as cv
from sklearn.kernel_ridge import KernelRidge
import time
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy.stats
from sklearn import metrics
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

MAX_NUM_ITERATIONS = 10
NUMBER_OF_K_FOLDS = 5


class MachineLearningMethods:

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
                self.execute_model(model_selected)
            elif model_selected == "q":
                print("Hope it has been helpful, enjoy data science! bye!")
            else:
                print("Invalid option selected, please introduce the number of the option")

    @staticmethod
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

    def execute_model(self, option_of_model):
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
            clf, specific_value_tried, graph_description = self.generate_selected_model(option_of_model, iteration)

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


class CleaningAutoClass:

    def __init__(self, tr_filepath = "./dev.csv"):
        # Import data
        print("Reading in data...")
        df = pd.read_csv(tr_filepath, header=0, index_col=None)
        print(len(df))
        print ("IDENTIFYING TYPES...")
        list_ib = set()  #input binary
        list_icn = set() #input categorical nominal
        list_ico = set() #input categorical ordinal
        list_if = set()  #input numerical continuos (input float)
        list_inputs = set()
        output_var = 'ob_target'

        for var_name in df.columns:
            if re.search('^i',var_name):
                list_inputs.add(var_name)
            if re.search('^ib_',var_name):
                list_ib.add(var_name)
            elif re.search('^icn_',var_name):
                list_icn.add(var_name)
            elif re.search('^ico_',var_name):
                list_ico.add(var_name)
            elif re.search('^if_',var_name):
                list_if.add(var_name)
            elif re.search('^ob_',var_name):
                output_var = var_name

        # Check number of NAs in each column
        print("Checking NAs...")
        numberOfNAs = df.isnull().sum()
        ColumnsWithCountOfNAs = numberOfNAs[numberOfNAs > 0]
        print(ColumnsWithCountOfNAs)

        #Removing NAs
        df_removed_rows = df[df.isnull().any(axis=1)]
        df = df.dropna()
        print("The following rows have been removed from the dataset:")
        print(df_removed_rows)

        print("Detecting NAs section finalized")

        #Outlier detection and replacing it with maximum/minimum value
        print("Starting outlier detection...")


        for colName in list_if:
            minLimit = df[colName].mean() - df[colName].std()*3
            maxLimit = df[colName].mean() + df[colName].std()*3

            minDelete = df[df[colName] < minLimit]
            minDelete[colName] = minLimit
            df = df[df[colName] >= minLimit]
            df = df.append(minDelete)
            print("minDel:" + str(colName) + " -> Values set to -3STD: " + str(len(minDelete)))

            maxDelete = df[df[colName] > maxLimit]
            maxDelete[colName] = maxLimit
            df = df[df[colName] <= maxLimit]
            df = df.append(maxDelete)
            print("maxDel:" + str(colName) + " -> Values set to +3STD: " + str(len(maxDelete)) + "\n")

        print("Detecting outliers section finalized")
        #Saving file
        print("Saving file...")
        df.to_csv("cleanAutomaticDataframe.csv", sep=',', encoding='utf-8')
        print("Dataframe successfully cleaned and saved! :)")


class CleaningManualClass:
    def countNas(self, df):
        # Check number of NAs in each column
        numberOfNAs = df.isnull().sum()
        if sum(numberOfNAs)>0:
            ColumnsWithCountOfNAs = numberOfNAs[numberOfNAs > 0]
            return(ColumnsWithCountOfNAs)
        else:
            return "No NAs have been found in the dataset"

    def deleteNas(self, df):
        df_removed_rows = df[df.isnull().any(axis=1)]
        df = df.dropna()
        print("The following rows have been removed from the dataset:")
        print(df_removed_rows)
        return(df)

    def replaceNasMed(self, df):
        list_ib = set()  #input binary
        list_icn = set() #input categorical nominal
        list_ico = set() #input categorical ordinal
        list_if = set()  #input numerical continuos (input float)
        list_inputs = set()
        output_var = 'ob_target'

        for var_name in df.columns:
            if re.search('^i',var_name):
                list_inputs.add(var_name)
                #print (var_name,"is input")
            if re.search('^ib_',var_name):
                list_ib.add(var_name)
                #print (var_name,"is input binary")
            elif re.search('^icn_',var_name):
                list_icn.add(var_name)
                #print (var_name,"is input categorical nominal")
            elif re.search('^ico_',var_name):
                list_ico.add(var_name)
                #print (var_name,"is input categorical ordinal")
            elif re.search('^if_',var_name):
                list_if.add(var_name)
                #print (var_name,"is input numerical continuos (input float)")
            elif re.search('^ob_',var_name):
                output_var = var_name

        numberOfNAs = df.isnull().sum()
        if sum(numberOfNAs) > 0:
            for colName in list_inputs:
                df.loc[colName] = df.apply(lambda x: x.fillna(x.median()), axis=0)
                print("NAs have been successfully replaced with median")
        else:
            print("No NAs to replace")
        return df

    def replaceNasZero(self, df):
        numberOfNAs = df.isnull().sum()
        if sum(numberOfNAs) > 0:
            df = df.fillna(0)
            print("NAs have been successfully replaced with 0s")
        else:
            print("No NAs to replace")
        return df

    def countOutliers(self, df):
        list_ib = set()  #input binary
        list_icn = set() #input categorical nominal
        list_ico = set() #input categorical ordinal
        list_if = set()  #input numerical continuos (input float)
        list_inputs = set()
        output_var = 'ob_target'

        for var_name in df.columns:
            if re.search('^i',var_name):
                list_inputs.add(var_name)
                #print (var_name,"is input")
            if re.search('^ib_',var_name):
                list_ib.add(var_name)
                #print (var_name,"is input binary")
            elif re.search('^icn_',var_name):
                list_icn.add(var_name)
                #print (var_name,"is input categorical nominal")
            elif re.search('^ico_',var_name):
                list_ico.add(var_name)
                #print (var_name,"is input categorical ordinal")
            elif re.search('^if_',var_name):
                list_if.add(var_name)
                #print (var_name,"is input numerical continuos (input float)")
            elif re.search('^ob_',var_name):
                output_var = var_name

        for colName in list_if:
            minLimit = df[colName].mean() - df[colName].std()*3
            maxLimit = df[colName].mean() + df[colName].std()*3
            minDelete = df[df[colName] < minLimit]
            maxDelete = df[df[colName] > maxLimit]
            print (str(colName) + ": Outliers above maximum value: " + str(len(maxDelete)) + ". Outliers below minimum value: " + str(len(minDelete)))

    def replaceOutliers(self, df):
        list_ib = set()  #input binary
        list_icn = set() #input categorical nominal
        list_ico = set() #input categorical ordinal
        list_if = set()  #input numerical continuos (input float)
        list_inputs = set()
        output_var = 'ob_target'

        for var_name in df.columns:
            if re.search('^i',var_name):
                list_inputs.add(var_name)
                #print (var_name,"is input")
            if re.search('^ib_',var_name):
                list_ib.add(var_name)
                #print (var_name,"is input binary")
            elif re.search('^icn_',var_name):
                list_icn.add(var_name)
                #print (var_name,"is input categorical nominal")
            elif re.search('^ico_',var_name):
                list_ico.add(var_name)
                #print (var_name,"is input categorical ordinal")
            elif re.search('^if_',var_name):
                list_if.add(var_name)
                #print (var_name,"is input numerical continuos (input float)")
            elif re.search('^ob_',var_name):
                output_var = var_name

        outliers_per_col = {}
        for colName in list_if:
            minLimit = df[colName].mean() - df[colName].std()*3
            maxLimit = df[colName].mean() + df[colName].std()*3

            minDelete = df[df[colName] < minLimit]
            outliers_per_col[str(colName)+"_min"] = len(minDelete)
            minDelete[colName] = minLimit
            #rowsToDelete = rowsToDelete.append(minDelete)
            df = df[df[colName] >= minLimit]
            df = df.append(minDelete)
            print("minDel:" + str(colName) + " -> Values set to -3STD: " + str(len(minDelete)))

            print(colName)
            maxDelete = df[df[colName] > maxLimit]
            outliers_per_col[str(colName)+"_max"] = len(maxDelete)
            maxDelete[colName] = maxLimit
            #rowsToDelete = rowsToDelete.append(maxDelete)
            df = df[df[colName] <= maxLimit]
            df = df.append(maxDelete)
            print("maxDel:" + str(colName) + " -> Values set to +3STD: " + str(len(maxDelete)))
            print("End of replacing outliers section")
        return df


    def deleteOutliers(self, df):
        list_ib = set()  #input binary
        list_icn = set() #input categorical nominal
        list_ico = set() #input categorical ordinal
        list_if = set()  #input numerical continuos (input float)
        list_inputs = set()
        output_var = 'ob_target'

        for var_name in df.columns:
            if re.search('^i',var_name):
                list_inputs.add(var_name)
               # print (var_name,"is input")
            if re.search('^ib_',var_name):
                list_ib.add(var_name)
               # print (var_name,"is input binary")
            elif re.search('^icn_',var_name):
                list_icn.add(var_name)
               # print (var_name,"is input categorical nominal")
            elif re.search('^ico_',var_name):
                list_ico.add(var_name)
               # print (var_name,"is input categorical ordinal")
            elif re.search('^if_',var_name):
                list_if.add(var_name)
               # print (var_name,"is input numerical continuos (input float)")
            elif re.search('^ob_',var_name):
                output_var = var_name

        rowsToDelete = pd.DataFrame(columns=list_inputs)
        count_of_outliers = {}
        for colName in list_if:
            minLimit = df[colName].mean() - df[colName].std()*3
            maxLimit = df[colName].mean() + df[colName].std()*3

            minDelete = df[df[colName] < minLimit]
            count_of_outliers[str(colName)+"_min"] = len(minDelete)
            rowsToDelete = rowsToDelete.append(minDelete)
            df = df[df[colName] >= minLimit]
            print("minDel:" + str(colName) + "  ->" + str(len(minDelete)))

            #print(colName)
            maxDelete = df[df[colName] > maxLimit]
            count_of_outliers[str(colName)+"_max"] = len(maxDelete)
            rowsToDelete = rowsToDelete.append(maxDelete)
            #print(len(df))
            df = df[df[colName] <= maxLimit]
            #print(len(df))
            print("maxDel:" + str(colName) + "  ->" + str(len(maxDelete)))
        print(str(sum(count_of_outliers.values())) + " outliers were successfully detected and corresponding rows have been deleted")
        print("End of deleting outliers section")
        return df

    def __init__(self, filename = "./dev.csv"):
        user_input = ""
        dfOriginal = pd.read_csv(filename, header=0, index_col=None)
        cleanDataFrame = dfOriginal
        while user_input != "q":
            print("Welcome to the Super Group A Data Cleaner!")
            print("1) Please press 1 if you want to clean NAs")
            print("2) Please press 2 if you want to clean outliers")
            print("3) Please press 3 if you want to save the clean dataframe in a new file")
            print("4) Please press 'q' if you want to quit without saving")
            user_input = input("Please select option: ")
            if user_input == "1":
                second_user_input = ""
                while second_user_input != "q":
                    print("a) Please press 'a' if you want to see the count of NAs")
                    print("b) Please press 'b' if you want to delete existing NAs")
                    print("c) Please press 'c' if you want to replace them with 0s")
                    print("d) Please press 'd' if you want to replace them with the median")
                    print("e) Please press 'q' if you want to go back to the main menu")
                    second_user_input = input("Please select an option: ")

                    if (second_user_input == "a"):
                       print(self.countNas(cleanDataFrame))
                    elif(second_user_input == "b"):
                        cleanDataFrame = self.deleteNas(cleanDataFrame)
                    elif(second_user_input == "c"):
                        cleanDataFrame = self.replaceNasZero(cleanDataFrame)
                    elif(second_user_input == "d"):
                        cleanDataFrame = self.replaceNasMed(cleanDataFrame)
                    elif(second_user_input == "q"):
                        print("Detecting NAs section finalized")
            elif user_input == "2":
                third_user_input = ""
                while third_user_input != "q":
                    print("a) Please press 'a' if you want to see the count of Outliers (note: it will reevaluate outliers each time it's executed)")
                    print("b) Please press 'b' if you want to delete existing Outliers")
                    print("c) Please press 'c' if you want to replace them with Maximum/Minimum depending on the value")
                    print("e) Please press 'q' if you want to go back to the main menu")
                    third_user_input = input("Please select an option: ")

                    if (third_user_input == "a"):
                        self.countOutliers(cleanDataFrame)
                    elif(third_user_input == "b"):
                        cleanDataFrame = self.deleteOutliers(cleanDataFrame)
                    elif(third_user_input == "c"):
                        cleanDataFrame = self.replaceOutliers(cleanDataFrame)
                    elif(third_user_input == "q"):
                        print("Detecting outliers section finalized")
            elif user_input == "3":
                print("Saving dataframe...")
                cleanDataFrame.to_csv("cleanManualDataframe.csv", sep=',', encoding='utf-8')
                print("Dataframe successfully saved. No more problems!")

            elif user_input == "q":
                print("Goodbye! Thanks for an amazing course!")


class PerformRandForest:
    def __init__(self, df):

        # Check if the user followed the instructions and set the name of the target variable as TARGET
        try:
            df['TARGET']
        except:
            print("There must be a variable called TARGET in the dataframe in order to run the random forest!")

        # Reading number of variables in the dataframe. Defining X - predictors, Y - target variable.

        ncol = df.shape[1]

        X = df.drop('TARGET',1)
        y = df['TARGET']

        # Create tree object. As a rule of thumb n_estimators should be equal to the squared number of variables of the dataframe.
        # Every other variable is at it's default setting tha the user can tweek.

        model = RandomForestClassifier(n_estimators=(ncol**2), criterion='gini', max_features = 'auto', bootstrap = True, max_depth = None)

        # Train the model using the training sets and check score
        result = model.fit(X, y)

        # Predict Output
        y_pred = result.predict_proba(X)[:,1]

        # Converting predictions array to datafram. Adding prediction column to the end of the dataframe and renaming it.

        pred = pd.DataFrame(y_pred)

        frames = [df,pred]

        df = pd.concat(frames, axis=1)

        df = df.rename(columns = {0:'Prediction'})

        print("The predicted values have been added at the end of your dataframe under the name #Prediction# ")

        df.to_csv("PerformRandForest.csv", sep=',', encoding='utf-8')


class binningDummyCreattion:
    def calculateEntropy(self, prob):

            entropy = -1*prob*np.log2(prob)
            return entropy

    def performBinning(self, x):

        # Assign initial value to entropy and best number of bins
        bestEntropy = 1.0
        best = 0

        for i in bins:
            try:
                data2 = [x, self.df['TARGET']]
                data = pd.concat(data2, axis=1)
                try:
                    data['binned'] = pd.qcut(data.ix[:,0], i, labels=False)

                # In case there is no differenciation
                except:
                    data['binned'] = data.ix[:,0]


                bindf = pd.DataFrame(index=range(round(float(data.shape[0])/(i+1))), columns=range(i))
                bindf = bindf.fillna(0)


                entropyList = []

                total = data.shape[0]


                for j in range(i):

                    sumTarget = data[data['binned']==j].ix[:,1].sum()

                    prob = sumTarget /total

                    # Applying entropy function
                    entropyList.append(self.calculateEntropy(prob))

                totEntropy= 0

                # Calculating total Entropy
                for j in entropyList:
                    totEntropy = totEntropy + (j/len(entropyList))

                # Checking if new entropy is lower than the previous one
                if totEntropy < bestEntropy:
                    print(totEntropy)
                    bestEntropy = totEntropy
                    best = i

                    print(best)

                else:
                    break

            except:
                break

        global binned
        binned[list(data.columns.values)[0]] = (pd.qcut(data.ix[:, 0], best, labels=False))

    def __init__(self, df):
        try:
            df['TARGET']
        except:
            print("There must be a variable called TARGET in the dataframe in order to perform the supervised binning!")

        # Making variable size of bins global so it can be used in the performBinning function also

        global bins
        bins = [2,5,10,25,50,75,100,150,200,250]


        # Splitting dataframe into floats, integers and strings

        df_num = df.select_dtypes(include=[np.float])
        df_int = df.select_dtypes(include=[np.int])
        df_string = list(df.select_dtypes(include=[object]))

        # Making variable size of bins global so it can be used in the performBinning function also

        global binned
        binned = pd.DataFrame()

        # Applying binning function to the num dataframe
        df_num.apply(self.performBinning,axis =0)
        binned = binned.ix[:, binned.columns != 'TARGET']

        # Generating dummies
        df_string = pd.get_dummies(df_string)

        #Merging dataframes

        df2= [binned,df_string]
        frame = pd.concat(df2, axis=1)

        frame2 = [df_int,frame]
        df = pd.concat(frame2, axis=1)

        df.to_csv("bin.csv")


class performPCAAndRatios:
    def __init__(self, df):
        # Reading the number of variables of the dataframe
        nrow = df.shape[0]
        ncol = df.shape[1]


        # convert it to numpy arrays
        X=df.values

        #Scaling the values
        X = scale(X)
        pca = PCA(n_components=min(nrow,ncol))
        pca.fit(X)

        # Finding by which two variables is the variance of the dataset mostly explained

        var = pca.explained_variance_ratio_

        index1 = max(enumerate(var))
        index1 = list(index1)
        index1 = index1.pop(0)
        var2 = np.delete(var, index1)
        index2 = max(enumerate(var2))
        index2 = list(index2)
        index2 = index2.pop(0)

        dframe = pd.DataFrame(data=X)

        # Creating Ratios by executing calculations between these first two variables

        dframe['Addition'] = dframe.iloc[:,index1] + dframe.iloc[:,index2]
        dframe['Substract'] = dframe.iloc[:,index1] - dframe.iloc[:,index2]
        dframe['Multiply'] = dframe.iloc[:,index1] * dframe.iloc[:,index2]
        dframe['Divide'] = dframe.iloc[:,index1] / dframe.iloc[:,index2]
        dframe['All_1'] = dframe.iloc[:,index1] - dframe.iloc[:,index2] / dframe.iloc[:,index2]
        dframe['All_2'] = dframe.iloc[:,index1] - dframe.iloc[:,index2] / dframe.iloc[:,index1]
        dframe['All_3'] = dframe.iloc[:,index1] + dframe.iloc[:,index2] / dframe.iloc[:,index2]
        dframe['All_4'] = dframe.iloc[:,index1] + dframe.iloc[:,index2] / dframe.iloc[:,index1]
        dframe['All_5'] = dframe.iloc[:,index1] * dframe.iloc[:,index2] / dframe.iloc[:,index2]
        dframe['All_6'] = dframe.iloc[:,index1] * dframe.iloc[:,index2] / dframe.iloc[:,index1]

        # merging old dataframe and the transformed one
        frames = [df, dframe]

        df = pd.concat(frames, axis=1)

        df.to_csv("PCA.csv")