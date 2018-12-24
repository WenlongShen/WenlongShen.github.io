---
layout:     post
title:      "Kaggle之路：Titanic"
subtitle:   "Titanic: Machine Learning from Disaster"
date:       2018-12-21
author:     "Wenlong Shen"
header-img: "img/bg/2018_9.jpg"
tags: 机器学习 Kaggle 2018
---

#### Overview

Titanic可谓是Kaggler的必经之路。我们以其为例，走一个完整的机器学习分析流程。

#### Step 1: 问题分析

关于Titanic的相关描述可参考官网，这是一个二分类的基本问题。

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

#### Step 2: 数据整理

首先加载必要的库。数据这块儿用pandas，模型用scikit-learn和xgboost，计算的库numpy和scipy，作图用matplotlib和seaborn等。


```python
#-*- coding: UTF-8 -*- 
#!/usr/bin/env python

# system parameters
import sys
print("Python version: {}". format(sys.version))

# functions for data processing and analysis
import pandas as pd
print("pandas version: {}". format(pd.__version__))

# machine learning algorithms
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neural_network
from sklearn import feature_selection, model_selection, metrics
import xgboost
from xgboost import XGBClassifier
print("scikit-learn version: {}". format(sklearn.__version__))
print("xgboost version: {}". format(xgboost.__version__))

# scientific computing
import numpy as np
import scipy as sp
print("NumPy version: {}". format(np.__version__))
print("SciPy version: {}". format(sp.__version__)) 

# data visualization
from pandas.tools.plotting import scatter_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import IPython
from IPython import display
print("matplotlib version: {}". format(mpl.__version__))
print("seaborn version: {}". format(sns.__version__))
print("IPython version: {}". format(IPython.__version__))
# Configure Visualization Defaults
# %matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use("ggplot")
sns.set_style("white")
pylab.rcParams["figure.figsize"] = 12,8

# misc libraries
import random
import time

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('-'*25)
# Input data files are available in the "datasets/titanic/" directory.
# Listing all files
from subprocess import check_output
data_dir = "datasets/titanic/"
print(check_output(["ls", data_dir]).decode("utf8"))

```

接下来我们看看数据的基本情况，可以使用info、describe等函数。


```python
train = pd.read_csv(data_dir+"train.csv")
test = pd.read_csv(data_dir+"test.csv")
train.info()
#test.info()
#train.describe(include = "all")
#data_train.sample(10)
```

个人习惯将train和test合并做数据整理和特征工程，另外PassengerId仅仅用于数据标识，Ticket意义不太明确，我们将这两项去掉。


```python
# saving passenger id in advance in order to submit later.
passengerId = test.PassengerId

train_len = len(train)
dataset =  pd.concat([train, test], axis=0).reset_index(drop=True)
#dataset.info()
#dataset.head()
drop_column = ["PassengerId", "Ticket"]
dataset.drop(drop_column, axis=1, inplace = True)
dataset.info()
```

我们统计一下原数据中的缺失值。


```python
print("Data columns with null values:\n", dataset.isnull().sum())
```

处理缺失值是数据预处理重要的一环，往往要综合考虑，多次尝试。常见的方法除均值、中值、众值外，还包括关联性考察、重新编码、去除缺失值过多的特征项、利用机器学习赋值等。

这里我们先看看Embarked特征项的两个缺失值，考察下Embarked特征项的整体情况。


```python
dataset[dataset.Embarked.isnull()]
pd.DataFrame(dataset.Embarked.value_counts(dropna=False))
```

可以看到，出现频率最高的为S，所以一种处理方式就是将缺失值设为众数S。不过，仔细看下Embarked和Pclass/Fare之间的关系会发现，Pclass=1，Fare=80时，Embarked最有可能是C，所以这里我们也可将缺失值设为C。


```python
h = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=dataset)
```

后续数据中，我们暂时用类似处理Embarked的方法填补Fare，用中值填补Age，用N填补Cabin。


```python
dataset.Embarked.fillna("C", inplace=True)
missing_value = dataset[(dataset.Pclass == 3) & (dataset.Embarked == "S") & (dataset.Sex == "male")].Fare.mean()
dataset.Fare.fillna(missing_value, inplace=True)
dataset.Age.fillna(dataset.Age.median(), inplace=True)
dataset.Cabin.fillna("N", inplace=True)
dataset.info()
#dataset.describe(include = "all")
```

接下来，我们简单看看train中各特征值之间的关系，包括与目标值Survived之间的关系等。


```python
data_plot = dataset[:train_len]
#histogram comparison of sex, class, and age by survival
h = sns.FacetGrid(data_plot, row = "Sex", col = "Pclass", hue = "Survived")
h.map(plt.hist, "Age", alpha = .75)
h.add_legend()
#pair plots of entire dataset
pp = sns.pairplot(data_plot, hue = "Survived", palette = "deep", size=1.2, diag_kind = "kde", diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={"shrink":.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor="white",
        annot_kws={"fontsize":12 }
    )
    
    plt.title("Pearson Correlation of Features", y=1.05, size=15)
correlation_heatmap(data_plot)
```

#### Step 3: 特征工程

特征工程可谓是个见仁见智见能力的事情。

我们逐个特征项来做吧，首先是Age。


```python
g = sns.kdeplot(data_plot.Age[data_plot.Survived == 0], color="Red", shade = True)
g = sns.kdeplot(data_plot.Age[data_plot.Survived == 1], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
```

可以看到整体差异不大，但儿童的生存率出现峰值，是人性的光辉吧。我们不妨抽提儿童作为新的特征，整体还可以按年龄分组。


```python
dataset["AgeBin"] = pd.cut(dataset.Age.astype(int), 5)
dataset["IsChild"] = [1 if i<16 else 0 for i in dataset.Age]
#dataset.info()
```

Cabin项描述了客舱编号，其开头字母可能具有信息，我们不妨抽提。


```python
#dataset.Cabin.head()
dataset.Cabin = [i[0] for i in dataset.Cabin]
g = sns.factorplot(y="Survived", x="Cabin", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
#dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix="Cabin")
#dataset.sample(10)
```

对于Fare，我们查看下分布，并使用不同于Age的分组方式。


```python
g = sns.kdeplot(data_plot.Fare[data_plot.Survived == 0], color="Red", shade = True)
g = sns.kdeplot(data_plot.Fare[data_plot.Survived == 1], ax =g, color="Blue", shade= True)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
dataset["FareBin"] = pd.qcut(dataset.Fare, 4)
```

对于Name，我们参考论坛的一些帖子，抽提Title作为特征项。


```python
dataset["Title"] = dataset.Name.str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (dataset.Title.value_counts() < 5)
dataset.Title = dataset.Title.apply(lambda x: "Misc" if title_names.loc[x] == True else x)
#dataset.Title.value_counts()
```

其它的特征项，我们也相应处理。


```python
dataset["FamilySize"] = dataset.SibSp + dataset.Parch + 1
dataset["IsAlone"] = 1
dataset["IsAlone"].loc[dataset.FamilySize > 1] = 0
dataset["IsMother"] = np.where((dataset.Age > 18) & 
                               (dataset.Title != 'Miss') & 
                               (dataset.Parch > 0) & 
                               (dataset.Sex == 'female'), 
                               1, 0)
#dataset.info()
```

为方便处理，我们需要进行一些编码、转换等工作。


```python
label = LabelEncoder()
dataset["Cabin_Code"] = label.fit_transform(dataset.Cabin)
dataset["Sex_Code"] = label.fit_transform(dataset.Sex)
dataset["Embarked_Code"] = label.fit_transform(dataset.Embarked)
dataset["Title_Code"] = label.fit_transform(dataset.Title)
dataset["AgeBin_Code"] = label.fit_transform(dataset.AgeBin)
dataset["FareBin_Code"] = label.fit_transform(dataset.FareBin)
drop_column = ["Cabin", "Sex", "Embarked", "Title", "AgeBin", "FareBin", "Name"]
dataset.drop(drop_column, axis=1, inplace = True)
dataset.info()
```

#### Step 4: 模型建立

先将train和test分开，并设好cv。


```python
train_x = dataset[:train_len]
test_x = dataset[train_len:]
train_y = pd.DataFrame(train_x.Survived)
train_x.drop("Survived", axis=1, inplace = True)
test_x.drop("Survived", axis=1, inplace = True)

ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
#train_x.info()
```

我们一次性尝试多个模型的效果。


```python
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.SGDClassifier(),
    #linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    #naive_bayes.MultinomialNB(),
    #naive_bayes.ComplementNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    #neighbors.RadiusNeighborsClassifier(radius=50),
    
    #NN
    neural_network.MLPClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    #svm.LinearSVC(),
    
    #Trees    
    #tree.DecisionTreeClassifier(),
    #tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    #discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]
```

下面评价一下各个模型的优劣。


```python
MLA_columns = ["MLA Name","MLA Parameters","MLA Train Accuracy Mean","MLA Test Accuracy Mean","MLA Test Accuracy 3*STD","MLA Time"]
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = train_y.copy()
#index through MLA and save performance to table
row_index = 0
for alg in MLA:
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, "MLA Name"] = MLA_name
    MLA_compare.loc[row_index, "MLA Parameters"] = str(alg.get_params())
    
    cv_results = model_selection.cross_validate(alg, train_x, train_y, cv  = cv_split)

    MLA_compare.loc[row_index, "MLA Time"] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, "MLA Train Accuracy Mean"] = cv_results["train_score"].mean()
    MLA_compare.loc[row_index, "MLA Test Accuracy Mean"] = cv_results["test_score"].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, "MLA Test Accuracy 3*STD"] = cv_results["test_score"].std()*3   #let's know the worst that can happen!
    
    #save MLA predictions
    alg.fit(train_x, train_y)
    MLA_predict[MLA_name] = alg.predict(train_x)
    
    row_index+=1
    
MLA_compare.sort_values(by = ["MLA Test Accuracy Mean"], ascending = False, inplace = True)
MLA_compare
```

我们还可以看看各模型预测结果间的相关性。


```python
correlation_heatmap(MLA_predict)
```

#### Step 5: 模型优化

常见的优化方法主要有调参、特征选择等。这里我们尝试使用GridSearchCV进行调参。


```python
vote_est = [
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc',  ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    ('lr',  linear_model.LogisticRegressionCV()),
    ('sgd', linear_model.SGDClassifier()),
    #('lrp', linear_model.Perceptron()),
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    ('knn', neighbors.KNeighborsClassifier()),
    ('mlp', neural_network.MLPClassifier()),
    ('svc', svm.SVC(probability=True)),
    ('nu',  svm.NuSVC(probability=True)),
    #('lsv', svm.LinearSVC()),
    ('qd', discriminant_analysis.QuadraticDiscriminantAnalysis()),
    ('xgb', XGBClassifier())
]

grid_seed = [5362]

grid_param = [
            # AdaBoostClassifier
            [{
            'n_estimators': [50, 100, 200],
            'learning_rate': [.05, .1, .2],
            'random_state': grid_seed
            }],
       
            # BaggingClassifier
            [{
            'n_estimators': [200, 300, 400],
            'max_samples': [.2, .3, .5],
            'random_state': grid_seed
            }],
            
            # ExtraTreesClassifier
            [{
            'n_estimators': [50, 100, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [6, 8, None],
            'random_state': grid_seed
            }],

            # GradientBoostingClassifier
            [{
            'learning_rate': [.1, .3], 
            'n_estimators': [100, 300, 500], 
            'max_depth': [2, 4, 6],
            'random_state': grid_seed
            }],

            # RandomForestClassifier
            [{
            'n_estimators': [100, 300, 500],
            'criterion': ['gini', 'entropy'],
            'max_depth': [6, 8, None],
            'oob_score': [True, False],
            'random_state': grid_seed
            }],
    
            # GaussianProcessClassifier
            [{    
            'max_iter_predict': [50, 100, 300],
            'random_state': grid_seed
            }],
    
            # LogisticRegressionCV
            [{
            'fit_intercept': [True, False],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'random_state': grid_seed
            }],
    
            # SGDClassifier
            [{
            'fit_intercept': [True, False],
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
            'random_state': grid_seed
            }],
    
            # Perceptron
            #[{
            #'fit_intercept': [True, False],
            #'random_state': grid_seed
            #}],
            
            # BernoulliNB
            [{
            'alpha': [.5, .75, 1.0],
            }],
    
            # GaussianNB
            [{}],
    
            # KNeighborsClassifier
            [{
            'n_neighbors': [5,6,7,8],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['minkowski'],
            'leaf_size': [16, 22, 26, 30]
            }],
    
            # MLPClassifier
            [{
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'random_state': grid_seed
            }],
            
            # SVC
            [{
            'C': [3,4,5,6],
            'gamma': [.1, .25, .5],
            'decision_function_shape': ['ovo', 'ovr'],
            'probability': [True],
            'random_state': grid_seed
            }],
    
            # NuSVC
            [{
            'gamma': [.1, .25, .5],
            'decision_function_shape': ['ovo', 'ovr'],
            'probability': [True],
            'random_state': grid_seed
            }],
    
            # LinearSVC
            #[{
            #'C': [1, 2],
            #'random_state': grid_seed
            #}],
    
            # QuadraticDiscriminantAnalysis
            [{}],
    
            #XGBClassifier
            [{
            'learning_rate': [.02, .03, .05], #default: .3
            'max_depth': [8, 10, 12], #default 2
            'n_estimators': [300, 400], 
            'colsample_bytree': [0.6, 0.8],
            'colsample_bylevel': [0.6, 0.8],
            'seed': grid_seed  
            }]   
        ]

start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip (vote_est, grid_param):
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = "roc_auc")
    best_search.fit(train_x, train_y)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print("The best parameter for {} is {} with a runtime of {:.2f} seconds.".format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print("Total optimization time was {:.2f} minutes.".format(run_total/60))

print("-"*10)
```

面对一众模型，我们在提交最终预测结果的时候还可以选择使用ensemble，这里以voting为例。


```python
ensembleVote = ensemble.VotingClassifier(estimators = vote_est , voting = "hard")
ensembleVote.fit(train_x, train_y)
prediction = ensembleVote.predict(test_x).astype(int)
submission = pd.DataFrame(data={
    "PassengerId": passengerId,
    "Survived": prediction
})
#submission.info()
submission.to_csv(data_dir+"submission.csv", index=False)
```

提交到Kaggle，最终得分0.78947。
