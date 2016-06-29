# -*- coding: utf-8 -*-
import string  
import re

import numpy as np
import pandas as pd
import random
from deap import creator, base, tools, algorithms
from sklearn import metrics
from sklearn.preprocessing import Imputer
import statsmodels.robust.scale as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
import re

legalchars = "abcdefghijklmnopqrstuvwxyx1234567890"
replace = (
    (("Ã","Å","Ä","À","Á","Â","å","å","ä","à","á","â"),"a"),
    (("Ç","Č","ç","č"),"c"),
    (("É","È","Ê","Ë","Ĕ","è","ê","ë","ĕ","é"),"e"),
    (("Ğ","Ģ","ģ","ğ"),"g"),
    (("Ï","Î","Í","Ì","ï","î","í","ì"),"i"),
    (("Ñ","ñ"),"n"),
    (("Ö","Ô","Ō","Ò","Ó","Ø","ö","ô","ō","ò","ó","ø"),"o"),
    (("Ŝ","Ş","ŝ","ş"),"s"),
    (("Ü","Ū","Û","Ù","Ú","ü","ū","û","ù","ú"),"u"),
    (("Ÿ","ÿ","&"),"y"),
    (("@")," ")
    )

def remove_chars(subject):
    """ Replace all chars that arent in allowed list """
    return re.sub(r'[^\x00-\x7f]',r' ',subject).replace("nan", " ")
    
def safe_chars_nombrepersona(subject):
    for r in replace:
        for c in r[0]:
            subject = subject.replace(c,r[1])
    a = remove_chars(subject)
    return a.title().replace('Do A ','').replace('Dona ','').replace('Don ','').replace('Mr ','').replace('Mrs ','').replace('Ms ','')

def safe_upperstr(subject):
    for r in replace:
        for c in r[0]:
            subject = subject.replace(c,r[1])
    s = remove_chars(normalize(subject))
    return s.upper()

def safe_upperstr_razonsocial(subject):
    for r in replace:
        for c in r[0]:
            subject = subject.replace(c,r[1])
    s = remove_chars(normalize(subject))
    return s.upper().replace('.','').replace(' SL','').replace(' SA','').replace(' SAU','').replace(' SLU','').replace(' SLP','').replace(' SLL','').replace(' SC','')



def DataCleaning(df):
    print ('DATA CLEANING. ')

    #Reemplazamos los valores infinito y menos infinito por cero
    df=df.replace([np.inf, -np.inf], 0)

    #Transformamos los Missing values en NaN y reemplazamos todos los NaN por la media de los datos de la columna en la que se encuentra.
    df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

    #Eliminamos los outliers.
    df2 = df[np.abs(df-df.mean())<=(3*df.std())]

    return df2

    
def Ratios_PCA_DT(df_inicial, output_var):
    
    df=df_inicial.drop(output_var, 1)

    # CREACION AUTOMATICA DE RATIOS: 

    # Creamos una lista con los nombres de las variables del dataframe que nos servira para asignar nombres
    # automaticamente a las nuevas variables que vamos a crear:
    var_names = list(df.columns.values)

    # APLICACION DE TRANSFORMACIONES
    longitud = len(df.columns)

    for i in range(0, longitud):
        for j in range(0, longitud):
            df['('+var_names[i]+'-'+var_names[j]+')/'+var_names[j]] = (df[df.columns[i]] - df[df.columns[j]]) / df[df.columns[j]] # TRANSFORMACION 1: (X-Y) / Y
            df[var_names[i]+'+'+var_names[j]] = df[df.columns[i]] + df[df.columns[j]] # TRANSFORMACION 2: X+Y
            df[var_names[i]+'*'+var_names[j]] = df[df.columns[i]] * df[df.columns[j]] # TRANSFORMACION 3: X*Y
            df[var_names[i]+'/'+var_names[j]] = df[df.columns[i]] / df[df.columns[j]] # TRANSFORMACION 4: X/Y
            df[var_names[i]+'-'+var_names[j]] = df[df.columns[i]] - df[df.columns[j]] # TRANSFORMACION 5: X-Y


    for i in range(0, longitud):
        df[var_names[i]+'^2'] = df[df.columns[i]] * df[df.columns[i]] # TRANSFORMACION 6: X^2

    print ('NUEVAS VARIABLES CREADAS:', len(df.columns)-longitud)

    # Creamos un df que unicamente contenga los ratios creados. Eliminamos las variables originales:
    df_ratios = df.drop(var_names, axis=1)
    df2 = df_ratios.fillna(0)
    df2 = df2.replace([np.inf,-np.inf],0)    

    # APLICACION PCA:
    # ESCALAMOS LOS DATOS:

    # El calculo de los componentes principales de una serie de variables depende de las unidades de medida empleadas. 
    # Si transformamos las unidades de medida, lo mas probable es que cambien a su vez los componentes obtenidos.
    # Para evitar este problema escalamos los datos. Con ello, se eliminan las diferentes unidades de medida 
    # y se consideran todas las variables implicitamente equivalentes en cuanto a la informacion recogida.
    df2=df2.values
    df2=scale(df2)

    # Aplicamos el modelo PCA sobre el dataframe:
    pca = PCA(n_components = len(df_ratios.columns))

    # Aplicamos el modelo PCA sobre el dataframe y calculamos la varianza explicada acumulada de cada componente:
    pca.fit(df2)
    var = pca.explained_variance_ratio_.cumsum()

    # Calculamos el numero de componente que explican un % de varianza determinado:
    for i in range(0, len(df_ratios.columns)):
        var_explained = var[i:i+1]
        if var_explained >= 0.75:
            print ('NUMERO DE COMPONENTES QUE EXPLICAN EL 75% DE LA VARIANZA...', i)
            # Aplicamos el PCA para el numero de componentes que explican el nivel de varianza deseado:
            pca = PCA(n_components = i)
            df2_pca = pca.fit_transform(df2)
            break  

    # Transformamos la matrix que devuelve el PCA en un nuevo dataframe: 
    df3= pd.DataFrame(data = df2_pca[::])    

        
    # APLICACION DEL DECISION TREE
    # Eliminamos NAN y numeros infinitos:
    df_ratios = df_ratios.fillna(0)
    df_ratios = df_ratios.replace([np.inf,-np.inf],0)

    # Calculamos el numero de componentes:
    num_pca = len(df3.columns)

    # Aplicamos el arbol de decision donde X = Dataframe con los ratios creados e Y = Componentes Principales:
    # Generamos un arbol por cada uno de los componentes principales. Para cada uno de ellos obtenemos las principales
    # ratios

    list_features = []

    for i in range(0, num_pca):
        dt = DecisionTreeRegressor(max_depth=3)
        dt_fit = dt.fit(df_ratios, df3[i])
        feature = dt_fit.tree_.feature
        feature = feature.tolist()
        list_features.append(feature)

    #print dt.feature_importances_
    #feature = dt.feature_importances_

    # Transformamos la lista de listas en una unica lista que contiene los numeros de los RATIOS importantes.
    # Eliminamos los duplicados y los valores negativos. Finalmente ordenamos la lista de menor a mayor:
    import itertools
    list_features = list(itertools.chain.from_iterable(list_features))
    list_unicos = list(set(list_features))
    list_pos = [x for x in list_unicos if x >= 0 ]
    list_pos.sort()

    print ('SELECCIONADOS...', len(list_pos), 'RATIOS') 
    
    # Creamos el df con los ratios buenos. A partir del numero de ratios lo buscamos en el df de ratios.
    # Para ello creamos una lista donde almancenamos los nombres de los ratios que queremos.
    # A partir de esa lista generamos un df con los ratios buenos.
    # Finalmente extraemos la variable target y la incluimos en el df junto con los ratios:

    list_names = []

    for i in range(0, len(list_pos)):
        list_names.append(df_ratios.columns[list_pos[i]])

    df_ratios_sel = df_ratios[list_names]
    df_target = df_inicial[output_var]

    df_final = pd.concat([df_ratios_sel,df_target], axis=1)

    print ('CREANDO DATAFRAME CON LOS RATIOS SELECCIONADOS Y LA VARIABLE TARGET...')
    
    return df_final
    
# GeneticLogisticRegression  
def GeneticLogisticRegression(df, output_var):

    df2=df.drop(output_var, 1)
    dfSize = len(df2.columns) # Numero de columnas menos la target.
    Y_train=df[output_var]
    
    print "GENETIC ALGORITHM FOR FEATURE SELECTION:"

    #SETING UP THE GENETIC ALGORITHM and CALCULATING STARTING POOL (STARTING CANDIDATE POPULATION)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    #GENEREATE BETAS LOGISTIC REGRESSION 
    def randomBetas():
        return round(random.uniform(-1,1),4)
        
    toolbox.register("attr_float", randomBetas)  
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dfSize)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        return sum(individual),

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    NPOPSIZE = 50 #RANDOM STARTING POOL SIZE
    population = toolbox.population(NPOPSIZE)  

    
    # GENETIC ALGORITHM (2 GENERATIONS)
    sum_current_gini=0.0
    sum_current_gini_1=0.0
    sum_current_gini_2=0.0
    
    OK = 1
    a=0
    while OK:  #REPEAT UNTIL IT DO NOT IMPROVE, AT LEAST A LITLE, THE GINI IN 2 GENERATIONS
        a=a+1
        print 'loop ', a
        OK=0

        # GENERATING OFFSPRING
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1) #CROSS-X PROBABILITY = 50%, MUTATION PROBABILITY=10%
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        populationCX =toolbox.select(offspring, k=len(population))
        mergedPopulation = population + populationCX
    
        sum_current_gini_2=sum_current_gini_1
        sum_current_gini_1=sum_current_gini
        sum_current_gini=0.0

        dic_gini={}
        for i in range(np.shape(mergedPopulation)[0]):
               
             Y_lreg = (df2 * list(mergedPopulation[i])).sum(axis=1)
             dfY_lreg = pd.DataFrame({'y_output':Y_lreg})
             Y_blreg = np.where(dfY_lreg['y_output']>0, 1, 0)

             fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_blreg)
             auc = metrics.auc(fpr, tpr)
             gini_power = abs(2*auc-1)
            
             gini=float(gini_power)
             dic_gini[gini]=mergedPopulation[i]   


        #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - START
        list_gini=sorted(dic_gini.keys(),reverse=True)
        population=[]
        for i in list_gini[:NPOPSIZE]:
            population.append(dic_gini[i])
            gini=float(i)
            sum_current_gini+=gini
          
        #HAS IT IMPROVED AT LEAST A LITLE THE GINI IN THE LAST 2 GENERATIONS
        print 'sum_current_gini=', sum_current_gini, 'sum_current_gini_1=', sum_current_gini_1, 'sum_current_gini_2=', sum_current_gini_2
        if(sum_current_gini>sum_current_gini_1+0.0001 or sum_current_gini>sum_current_gini_2+0.0001):
            OK=1

    gini=list_gini[0] 
    parameters=dic_gini[list_gini[0]]
    print parameters
    
    # PRINTING OUT THE LIST OF FEATURES
    for i in range(len(parameters)):
        print 'Parameter', list(df2.columns)[i], ':', parameters[i] 
    #print 'gini: ', gini

    return parameters

def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
