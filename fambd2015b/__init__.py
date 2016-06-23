import pandas as pd
import numpy
from sklearn.preprocessing import Imputer
import six
import ast
import time
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import SGDClassifier as SGDC
import pandas as pd
import numpy
from sklearn.preprocessing import Imputer
import six
import ast


df = pd.read_csv('https://dl.dropboxusercontent.com/u/28535341/dev.csv')  
#df = ts.sample(frac=0.3, replace=True)
#print(df.dtypes)
#print(df.head()) 

def describe_df(dtf):
    int_type, str_type, float_type, undf  = {},{},{},{}
    print('Dataframe Shape:', df.shape)
    for cm in dtf.columns:
        int_type[cm] = [1 for x in list(dtf[cm]) if isinstance(x, six.integer_types) or isinstance(x, numpy.int64)]
        str_type[cm] = [1 for x in list(dtf[cm]) if isinstance(x, six.string_types)]
        float_type[cm] = [1 for x in list(dtf[cm]) if isinstance(x, float) or isinstance(x, numpy.float64)]
        undf[cm] = [1 for x in list(dtf[cm]) if x == None]
        print(cm, '--Row Type:',dtf[cm].dtype,':->', 
              'Integers:',round((sum(int_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
              'Floats:',round((sum(float_type[cm]) / len(dtf[cm])*100), 2),'%','--',
              'Strings:',round((sum(str_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
              'Empty Rows:',round(((dtf[cm].isnull().sum()+sum(undf[cm])) / len(dtf[cm])*100), 2),'%')
              
def try_numeric(dtf):
    int_type, str_type, float_type, undf  = {},{},{},{} 
    cvt = {}
    print('Trying to convert all values that appear to be numeric to numeric:') 
    for cm in dtf.columns:
        cvt[cm] = list(dtf[cm])
        for idx, row in enumerate(cvt[cm]):
            try:
                cvt[cm][idx] = ast.literal_eval(row.replace(',','.'))
            except:
                cvt[cm][idx] = row
            dtf[cm] = cvt[cm]
    for cm in dtf.columns:
            int_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.integer_types) or isinstance(x, numpy.int64)]
            str_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.string_types)]
            float_type[cm] = [1 for x in list(df[cm]) if isinstance(x, float) or isinstance(x, numpy.float64)]
            undf[cm] = [1 for x in list(df[cm]) if x == None]
            print(cm, '--Row Type:',df[cm].dtype,':->', 
                  'Integers:',round((sum(int_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
                  'Floats:',round((sum(float_type[cm]) / len(dtf[cm])*100), 2),'%','--',
                  'Strings:',round((sum(str_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
                  'Empty Rows:',round(((dtf[cm].isnull().sum()+sum(undf[cm])) / len(dtf[cm])*100), 2),'%')
    return(dtf)

def eliminate_minority(dtf):
    print('1.Convert to most frequent Data Type')
    int_type, str_type, float_type, undf  = {},{},{},{} 
    for cm in dtf.columns:
        int_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.integer_types) or isinstance(x, numpy.int64)]
        str_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.string_types)]
        float_type[cm] = [1 for x in list(df[cm]) if isinstance(x, float) or isinstance(x, numpy.float64)]
        undf[cm] = [1 for x in list(df[cm]) if x == None]
        if(sum(int_type[cm]) > sum(str_type[cm]) and sum(int_type[cm]) > sum(float_type[cm])):
            df[cm] = [numpy.nan if isinstance(x, six.string_types) else x for x in list(df[cm])]
        elif(sum(str_type[cm]) > sum(int_type[cm]) and sum(str_type[cm]) > sum(float_type[cm])):
            df[cm] = [numpy.nan if isinstance(x, six.integer_types) or isinstance(x, numpy.int64) or
                      isinstance(x, float) or isinstance(x, numpy.float64)
                      else x for x in list(df[cm])]
        elif(sum(float_type[cm]) > sum(int_type[cm]) and sum(float_type[cm]) > sum(str_type[cm])):
            df[cm] = [numpy.nan if isinstance(x, six.string_types) else x for x in list(df[cm])]
    return(df.dtypes) 


def outlier(dtf):  # finds outliers within 3 standard deviations and converts to NaNs
    print("This code finds the outliers within 3 standard deviations and converts to NaNs") 
    b = len(dtf) 
    for var_name in dtf.columns: 
        dtf[var_name] = dtf[var_name][numpy.abs(dtf[var_name]-dtf[var_name].mean())<=(3*dtf[var_name].std())] 
        print(var_name,'->',len(list(dtf[var_name][numpy.abs(dtf[var_name]-dtf[var_name].mean())<=(3*dtf[var_name].std())])),
                         'not removed, original:', len(list(dtf[var_name]))) #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    print("Outliers removed") 
    return(dtf)
    
def impute(dtf): # imputes by mean if float, imputes by median if interger.         
    col_list = []
    nans_r = dtf.shape[0] - dtf.dropna().shape[0]
    perc_row = nans_r / len(dtf)  
    for var_name in dtf.columns:
        if dtf[var_name].isnull().sum() >= .50*len(dtf):   
            col_list.append(var_name)
    for var_name in dtf.columns: 
        if var_name in col_list:
            dtf = dtf.drop(col_list, axis = 1)
        elif perc_row < 0.02:
            dtf = dtf.dropna()
        elif dtf[var_name].dtype == "float64":
            dtf.fillna(dtf.mean()[var_name]) 
        else:
            dtf.fillna(dtf.median()[var_name]) 
    return(dtf)  
    
def a1_main(df): 
    df1 = try_numeric(df) 
    df2 = eliminate_minority(df1)
    df3 = outlier(df2) 
    df_f = impute(df3) 
    return(df_f)     

print("A.4.2) Automated WoE Calculation, binning and Transformation with Automated Supervised Binning.")
print()
print("This is a program that given a dataframe, does the following: obtains the name of the target variable, calculates the WoE for each value then the IV for each variable, bins the results into quintiles, creates a final dataframe with three columns: column name, IV, and binned value.") 

# A.4.1 will recieve input from A.1. The result from A.1 should be a clean dataframe that is ready to run these calculations on. 
import pandas as pd
import numpy as np
df = pd.read_csv('https://dl.dropboxusercontent.com/u/28535341/dev.csv') 
df_clean = df # assign a clean dataframe if we need it later

# while loop to obtain user input on name of target variable. Use if target variable unknown by program. 
while True: 
    tv = input("Enter name of target variable (type e to exit):  ")   
    if tv in df.columns:
        print("Perfect! Thanks")
        break
    elif tv == 'e': 
        break  
    
while True: 
    id_var = input("Enter name of id variable (type e if none or to exit):  ")   
    if id_var in df.columns:
        print("Perfect! Thanks")  
        break
    elif id_var == 'e': 
        break
    
td1 = [tv,id_var]
td  = [i for i in td1 if i != 'e']   
td = [x for x in td if len(x) > 0]

# Function to calculate WOE and IV given a dataframe with no binning
def auto_woe(df): 
    iv_list = [] 
    tv = 'ob_target' 
    a= 0.01
    td = ['ob_target','id']  # hard code variables 
    df_drop_tar = df.drop(td, axis=1)  
    for var_name in df_drop_tar.columns: 
        biv = pd.crosstab(df[var_name],df[tv])   
        WoE = np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)) #multiply by 1.0 to transform into float and add "a=0.01" to avoid division by zero.
        IV = sum(((1.0*biv[0]/sum(biv[0])+a) - (1.0*biv[1]/sum(biv[1])+a))*np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)))
        iv_list.append(IV)
    iv_list = iv_list + [0] + [0] 
    col_list =list(df.columns)
    test = pd.DataFrame({'Column Name' : col_list,'IV' : iv_list})
    return(test.sort_values(by = 'IV', ascending = False))  
    
# Function for determing optimal number of bins (up to 20) for each variable with over 20 distint values using IV. 
# This is for running on samples only. 
def auto_bin(df):
    col_list = []
    loop_list = [i for i in range(1,21)]   
    iv_list = []
    bins_list = [] 
    iv_max = 0 
    a= 0.01
    tv = 'ob_target'
    td = ['ob_target','id'] 
    df_drop_tar = df.drop(td, axis=1)  
    for var_name in df_drop_tar.columns:
        if(len(df[var_name].unique()) >= 24):  # more than 20 unique values to use binning   
            col_list.append(var_name)
    for var_name in col_list:  
        for i in loop_list: 
            bins = np.linspace(df[var_name].min(), df[var_name].max(), i)  
            groups = np.digitize(df[var_name], bins) 
            biv = pd.crosstab(groups,df[tv])   
            WoE = np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)) #multiply by 1.0 to transform into float and add "a=0.01" to avoid division by zero.
            IV = sum(((1.0*biv[0]/sum(biv[0])+a) - (1.0*biv[1]/sum(biv[1])+a))*np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)))
            if IV > iv_max:
                iv_max = IV
                bin_f = len(bins)    
        iv_list.append(iv_max) 
        bins_list.append(bin_f)  
    #iv_list = iv_list + [0]
    #bins_list = bins_list + [0] 
    test = pd.DataFrame({'Column Name' : col_list,'IV' : iv_list, 'Num_Bins' : bins_list})  
    return(test.sort_values(by = 'IV', ascending = False))

# Function for determining value of IV    
def iv_binning(df2):   
    df2['Usefulness'] = ['Suspicous' if x > 0.5 else 'Strong' if x <= 0.5 and x > 0.3 else 'Medium'
                         if x <= 0.3 and x > 0.1 else 'Weak' if x <= 0.1 and x > 0.02 else
                         'Not Useful' for x in df2['IV']]  # Source for 'Usefullness Values' Siddiqi (2006)  
    return(df2)
    
print("These are secondary fuctions. The first gives  the sorted IV for all variables and their usefulness. The second gives the best number of bins and the associated IV and usefulness for the variables that can be binned.")  

def woe(df):     # gives IV for each variable and its usefullness   
    df1 = auto_woe(df) 
    df_woe = iv_binning(df1)
    return(df_woe)    

def bins(df):  # gives IV for binned variables and their usefullness
    df2 = auto_bin(df)
    df_bin = iv_binning(df2)
    return(df_bin)
    
print("These are the main functions. One to compare the automatically selected binning with the scenario in which no binning was performed and the other to transform the original dataset to a binned dataset using the automatically chosen binning.") 

def compare(df): # compares IV and Usefullness for binned and IV and Usefullness for unbinned varaibles  
    df1 = auto_woe(df) 
    df_woe = iv_binning(df1)  
    df2 = auto_bin(df)
    df_bin = iv_binning(df2) 
    common = df_bin.merge(df_woe,on=['Column Name'])  
    common.columns = ['Column Name', 'IV Bins', 'Num Bins', 'Usefulness IV', 'IV No Bins', 'Usefulness IV']
    return(common)   
    
def transform(df): # Bins dataset according to the best bins on the basis of an increase in IV
    common = compare(df) 
    bin_list = common['Num Bins']
    variable_list = common['Column Name']
    for idx,var_name in enumerate(variable_list): 
        bins = np.linspace(df[var_name].min(), df[var_name].max(), bin_list[idx])
        df[var_name] = np.digitize(df[var_name], bins) 
    return(df) 
    
def a4_2_main(df): # Main fucntion: compares IV and Usefullness for binned and IV and Usefullness for unbinned varaibles  
    df1 = auto_woe(df) 
    df_woe = iv_binning(df1)   
    df2 = auto_bin(df)
    df_bin = iv_binning(df2) 
    common = df_bin.merge(df_woe,on=['Column Name'])  
    common.columns = ['Column Name', 'IV Bins', 'Num Bins', 'Usefulness IV', 'IV No Bins', 'Usefulness IV']
    return(common) 
    
a4_2_main(df)


# dev = pd.read_csv("dev.csv")
df = pd.read_csv('https://dl.dropboxusercontent.com/u/28535341/dev.csv') 
#a = dev['ob_target'] #target vble 

def get_tar():
    while True: 
        tv = input("Enter name of target variable (type e to exit):  ")   
        if tv in dev.columns:
            print("Perfect! Thanks") 
            break
        elif tv == 'e': 
            break  
    return(tv)
    
def get_id():
    while True: 
        id_var = input("Enter name of id variable (type e if none or to exit):  ")   
        if id_var in dev.columns:
            print("Perfect! Thanks")   
            break
        elif id_var == 'e':  
            break
    return
    
a = dev[tv] 
to_drop = [tv,id_var] #Drop id 
to_drop  = [i for i in td1 if i != 'e']   
to_drop = [x for x in td if len(x) > 0]
in_model = dev.drop(to_drop, axis=1)    

for var_name in in_model:
    dev[var_name] = (dev[var_name]- dev[var_name].mean()) / (dev[var_name].max() - dev[var_name].min()) #normalize vble
    
b = in_model.as_matrix().astype(np.float) #New matrix b

def cv(b, a, clf_class, **kwargs):
    kf = KFold(len(a), n_folds=10, shuffle=True)
    apr = a.copy() 
    for train_index, test_index in kf:
        b_train, b_test = b[train_index], b[test_index]
        atr = a[train_index]
        clf = clf_class(**kwargs)
        clf.fit(b_train,atr)
        apr[test_index] = clf.predict(b_test)
    return apr #Kfold
    
def pct_correct(a_true,apr):
    return np.mean(a_true == apr) #Calculating accuracy
    
def get_b(dev): 
    to_drop = ['ob_target','id']
    in_model = dev.drop(to_drop, axis=1)
    for var_name in in_model: 
        dev[var_name] = (dev[var_name]- dev[var_name].mean()) / (dev[var_name].max() - dev[var_name].min())
    b = in_model.as_matrix().astype(np.float)
    return(b) 
    
def get_a(dev):
    a = dev['ob_target']
    return(a) 

def model_select_main(df): 
    a = df['ob_target']
    to_drop = ['ob_target','id']
    in_model = df.drop(to_drop, axis=1) 
    #for var_name in in_model: 
        #df[var_name] = (df[var_name]- df[var_name].mean()) / (df[var_name].max() - df[var_name].min())
    b = in_model.as_matrix().astype(np.float)
    name = ["Support Vector Machines","Random Forest","K-Nearest-Neighbors","AdaBoost","Gradient Boosting","Stochastic Gradient Descent"]
    model = [SVC,RF,KNN,ADA,GBC,SGDC]
    num = len(model) #Numbers to iterate
    list = []
    for index in range(num):
        list.append({'Model Name': name[index], '% Correct': pct_correct(a, cv(b, a, model[index]))}) #Models accuracy
    dev2 = pd.DataFrame(list)
    sorted_models = dev2.sort_values(by='% Correct', ascending=False) #Accuracy order
    return(print(sorted_models))

model_select_main(df) 

df = pd.read_csv('https://dl.dropboxusercontent.com/u/28535341/dev.csv')  
#df = ts.sample(frac=0.3, replace=True)
#print(df.dtypes)
#print(df.head())

def describe_df(dtf): 
    int_type, str_type, float_type, undf  = {},{},{},{}
    print('Dataframe Shape:', df.shape)
    for cm in dtf.columns:
        int_type[cm] = [1 for x in list(dtf[cm]) if isinstance(x, six.integer_types) or isinstance(x, numpy.int64)]
        str_type[cm] = [1 for x in list(dtf[cm]) if isinstance(x, six.string_types)]
        float_type[cm] = [1 for x in list(dtf[cm]) if isinstance(x, float) or isinstance(x, numpy.float64)]
        undf[cm] = [1 for x in list(dtf[cm]) if x == None]
        print(cm, '--Row Type:',dtf[cm].dtype,':->', 
              'Integers:',round((sum(int_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
              'Floats:',round((sum(float_type[cm]) / len(dtf[cm])*100), 2),'%','--',
              'Strings:',round((sum(str_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
              'Empty Rows:',round(((dtf[cm].isnull().sum()+sum(undf[cm])) / len(dtf[cm])*100), 2),'%')
              
def try_numeric(dtf):
    int_type, str_type, float_type, undf  = {},{},{},{} 
    cvt = {}
    print('Trying to convert all values that appear to be numeric to numeric:') 
    for cm in dtf.columns:
        cvt[cm] = list(dtf[cm])
        for idx, row in enumerate(cvt[cm]):
            try:
                cvt[cm][idx] = ast.literal_eval(row.replace(',','.'))
            except:
                cvt[cm][idx] = row
            dtf[cm] = cvt[cm]
    for cm in dtf.columns:
            int_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.integer_types) or isinstance(x, numpy.int64)]
            str_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.string_types)]
            float_type[cm] = [1 for x in list(df[cm]) if isinstance(x, float) or isinstance(x, numpy.float64)]
            undf[cm] = [1 for x in list(df[cm]) if x == None]
            print(cm, '--Row Type:',df[cm].dtype,':->', 
                  'Integers:',round((sum(int_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
                  'Floats:',round((sum(float_type[cm]) / len(dtf[cm])*100), 2),'%','--',
                  'Strings:',round((sum(str_type[cm]) / len(dtf[cm])*100), 2),'%','--', 
                  'Empty Rows:',round(((dtf[cm].isnull().sum()+sum(undf[cm])) / len(dtf[cm])*100), 2),'%')
    return(dtf)

def eliminate_minority(dtf):
    print('1.Convert to most frequent Data Type')
    int_type, str_type, float_type, undf  = {},{},{},{}
    for cm in dtf.columns:
        int_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.integer_types) or isinstance(x, numpy.int64)]
        str_type[cm] = [1 for x in list(df[cm]) if isinstance(x, six.string_types)]
        float_type[cm] = [1 for x in list(df[cm]) if isinstance(x, float) or isinstance(x, numpy.float64)]
        undf[cm] = [1 for x in list(df[cm]) if x == None]
        if(sum(int_type[cm]) > sum(str_type[cm]) and sum(int_type[cm]) > sum(float_type[cm])):
            df[cm] = [numpy.nan if isinstance(x, six.string_types) else x for x in list(df[cm])]
        elif(sum(str_type[cm]) > sum(int_type[cm]) and sum(str_type[cm]) > sum(float_type[cm])):
            df[cm] = [numpy.nan if isinstance(x, six.integer_types) or isinstance(x, numpy.int64) or
                      isinstance(x, float) or isinstance(x, numpy.float64)
                      else x for x in list(df[cm])]
        elif(sum(float_type[cm]) > sum(int_type[cm]) and sum(float_type[cm]) > sum(str_type[cm])):
            df[cm] = [numpy.nan if isinstance(x, six.string_types) else x for x in list(df[cm])] 
    return(dtf)  
    
def outlier(dtf):  # finds outliers within a user given number of standard deviations and converts to NaNs
    print("This code finds the outliers within a user given number of standard deviations and converts to NaNs") 
    b = len(dtf) 
    while True:
        a = input("We will remove outliers now. Choose the number of standard deviations to check for outliers or 'e' to exit: ")
        a = int(a)
        if a == 'e':
            break
        else: 
            for var_name in dtf.columns: 
                dtf[var_name] = dtf[var_name][numpy.abs(dtf[var_name]-dtf[var_name].mean())<=(a*dtf[var_name].std())] 
                print(var_name,'->',len(list(dtf[var_name][numpy.abs(dtf[var_name]-dtf[var_name].mean())<=(a*dtf[var_name].std())])),
                         'not removed, original:', len(list(dtf[var_name]))) #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
                #b = len(dtf[~(np.abs(dtf.Data-dtf.Data.mean())>(3*df.Data.std()))])
            print("Outliers removed") 
            break
    return(dtf)

def user_impute(dtf):
    col_list = []
    nans_t = dtf.isnull().values.ravel().sum()
    nans_r = dtf.shape[0] - dtf.dropna().shape[0]
    perc_row = nans_r / len(dtf)  
    for var_name in dtf.columns:
        if dtf[var_name].isnull().sum() >= .50*len(dtf): 
            col_list.append(var_name) 
    print("There are",nans_t, "NaNs,",nans_r, "rows with NaNs, (",perc_row," percent of all rows) and", len(col_list),"columns that have 50% or more NaNs")
    print() 
    while True:
        a = input("Choose an option for NaNs: 'dc' to drop all columns with over 50% NaNs, 'dr' to drop all rows with NaNs, or e to exit: ")
        if a == 'dc':
            print("Invald columns dropped. ") 
            dtf = dtf.drop(col_list, axis = 1) 
            break
        elif a == 'dr':
            print("Invalid rows dropped. ")  
            dtf = dtf.dropna()
            break
        elif a == 'e': 
            break
        else:
            print("That is not an option. Try again")
    print()
    while True:
        b = input("Choose an imputation strategy: 'm' for mean, 'md' for median, or type 'e' to exit if none:  ")
        if b == 'm':
            print("Imputed by mean. ")
            dtf = dtf.fillna(dtf.mean())
            break
        elif b == 'md':
            print("Imputed by median. ")
            dtf = dtf.fillna(dtf.median())  
            break
        elif b == 'e':
            break
        else:
            print("That is not an option. Try again")
    return(dtf)

def h1_main(df): 
    df1 = try_numeric(df) 
    df2 = eliminate_minority(df1)
    df3 = outlier(df2)
    df_f = user_impute(df3)  
    return(df_f)     
    
import pandas as pd
import numpy as np
df = pd.read_csv('https://dl.dropboxusercontent.com/u/28535341/dev.csv') 
df_clean = df # assign a clean dataframe if we need it later 

# while loop to obtain user input on name of target variable
while True: 
    tv = input("Enter name of target variable (type e to exit):  ")   
    if tv in df.columns:
        print("Perfect! Thanks")
        break
    elif tv == 'e':
        break 
    
while True: 
    id_var = input("Enter name of id variable (type e if none or to exit):  ")   
    if id_var in df.columns:
        print("Perfect! Thanks")  
        break
    elif id_var == 'e':   
        break  
    

td1 = [id_var]
td  = [i for i in td1 if i != 'e']  
td = [x for x in td if len(x) > 0]  
df_drop = df.drop(td,axis=1)   

def user_woe(df):  
    bin_list = []
    td = ['ob_target','id']
    df_drop = df.drop(td,axis=1) 
    for var_name in df_drop: 
        if(len(df_drop[var_name].unique()) >= 24):  # more than 24 unique values to use binning 
            bin_list.append(var_name) 
    iv_list = [] 
    a= 0.01 
    for var_name in bin_list:  
        biv = pd.crosstab(df[var_name],df[tv])   
        WoE = np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)) #multiply by 1.0 to transform into float and add "a=0.01" to avoid division by zero.
        IV = sum(((1.0*biv[0]/sum(biv[0])+a) - (1.0*biv[1]/sum(biv[1])+a))*np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)))
        iv_list.append(IV)
    iv_list = iv_list
    test = pd.DataFrame({'Column Name' : bin_list,'IV' : iv_list}) 
    return(test.sort_values(by = 'IV', ascending = False)) 
    
def user_bin(df):
    col_list = []  
    iv_list = []
    bins_list = []  
    iv_max = 0  
    a= 0.01  
    for var_name in df_drop: 
        if(len(df_drop[var_name].unique()) >= 24):  # more than 24 unique values to use binning (user cannot change) 
            col_list.append(var_name)  
    print("There are " + str(len(col_list)) + " columns that are elegible for binning")  
    print() 
    for var_name in col_list:  
        num = input('Please enter the number of bins you would like for varaible: ' + str(var_name) + 
                    ' (Min = '+ str(min(df[var_name])) + ', Max = '+ str(max(df[var_name])) + 
                    ', Std = ' + str(round(np.std(df[var_name]),2)) + ')')  
        bins = np.linspace(df[var_name].min(), df[var_name].max(), num) 
        groups = np.digitize(df[var_name], bins) 
        biv = pd.crosstab(groups,df[tv])   
        WoE = np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)) #multiply by 1.0 to transform into float and add "a=0.01" to avoid division by zero.
        IV = sum(((1.0*biv[0]/sum(biv[0])+a) - (1.0*biv[1]/sum(biv[1])+a))*np.log((1.0*biv[0]/sum(biv[0])+a) / (1.0*biv[1]/sum(biv[1])+a)))    
        iv_list.append(IV) 
        bins_list.append(num)  
    test = pd.DataFrame({'Column Name' : col_list,'IV' : iv_list, 'Num_Bins' : bins_list})  
    return(test.sort_values(by = 'IV', ascending = False))

# Function for determining value of IV    
def iv_binning(df2):   
    df2['Usefulness'] = ['Suspicous' if x > 0.5 else 'Strong' if x <= 0.5 and x > 0.3 else 'Medium'
                         if x <= 0.3 and x > 0.1 else 'Weak' if x <= 0.1 and x > 0.02 else
                         'Not Useful' for x in df2['IV']]  # Source for 'Usefullness Values' Siddiqi (2006) 
    return(df2)

def user_compare(df): # compares IV and Usefullness for binned and IV and Usefullness for unbinned varaibles  
    df1 = user_woe(df) 
    df_woe = iv_binning(df1) 
    df2 = user_bin(df)
    df_bin = iv_binning(df2) 
    common = df_bin.merge(df_woe,on=['Column Name'])  
    common.columns = ['Column Name', 'IV Bins', 'Num Bins', 'Usefulness IV', 'IV No Bins', 'Usefulness IV']
    return(common)

def user_transform(df): 
    common = user_compare(df) 
    bin_list = common['Num Bins']
    variable_list = common['Column Name']
    for idx,var_name in enumerate(variable_list):
        bins = np.linspace(df[var_name].min(), df[var_name].max(), bin_list[idx]) 
        df[var_name] = np.digitize(df[var_name], bins)  
    
def h3_main(df): # Main function to compare IV and Usefullness for binned and IV and Usefullness for unbinned varaibles  
    df1 = user_woe(df) 
    df_woe = iv_binning(df1)  
    df2 = user_bin(df)
    df_bin = iv_binning(df2) 
    common = df_bin.merge(df_woe,on=['Column Name'])  
    common.columns = ['Column Name', 'IV Bins', 'Num Bins', 'Usefulness IV', 'IV No Bins', 'Usefulness IV']
    return(common)


