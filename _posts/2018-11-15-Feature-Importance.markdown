---
layout:     post
title:      "Feature Importance"
subtitle:   "特征重要性"
date:       2018-11-15
author:     "Wenlong Shen"
header-img: "img/bg/2018_6.jpg"
tags: 机器学习 2018
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

当我们训练完一个模型，得到理想的预测结果之后，或许我们还应该问问：哪个特征最为重要，它对模型有什么样的贡献？

#### Permutation Importance

Permutation的策略是考虑在模型训练完之后，将单个特征的数据值随机洗牌，破坏原有的对应关系后，再考察模型预测效果的变化情况。

	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	data = pd.read_csv('...')
	y = data['y']
	feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
	X = data[feature_names]
	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
	my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
	# calculate and show importances with the eli5 library
	import eli5
	from eli5.sklearn import PermutationImportance
	perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
	eli5.show_weights(perm, feature_names = val_X.columns.tolist())

#### Partial Dependence Plots

有时候我们希望考察单个特征是如何影响模型预测结果的，这就用到部分依赖图。下面是一个画决策树的例子：

	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	data = pd.read_csv('...')
	y = data['y']
	feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
	X = data[feature_names]
	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
	tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
	# draw the decision tree
	from sklearn import tree
	import graphviz
	tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
	graphviz.Source(tree_graph)

另外pdpbox库也可以方便画出单个特征或特征对对模型的影响：

	from matplotlib import pyplot as plt
	from pdpbox import pdp, get_dataset, info_plots
	# Create the data that we will plot
	pdp_data = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature='A')
	# plot it
	# The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.
	pdp.pdp_plot(pdp_data, 'A')
	plt.show()
	# 2D partial dependence plots show interactions between features
	features_to_plot = ['A', 'B']
	inter = pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)
	pdp.pdp_interact_plot(pdp_interact_out=inter1 feature_names=features_to_plot, plot_type='contour')
	plt.show()

#### SHAP Values

SHAP Values (an acronym from SHapley Additive exPlanations) 描述的是对于任意一个预测结果，其各个特征值的贡献情况：

	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	data = pd.read_csv('...')
	y = data['y']
	feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
	X = data[feature_names]
	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
	my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
	# arbitrarily chose row 5
	row_to_show = 5
	data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
	data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
	my_model.predict_proba(data_for_prediction_array)
	import shap  # package used to calculate Shap values
	# Create object that can calculate shap values
	explainer = shap.TreeExplainer(my_model)
	# Calculate Shap values
	shap_values = explainer.shap_values(data_for_prediction)
	shap.initjs()
	shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
	# SHAP has explainers for every type of model
	# shap.DeepExplainer works with Deep Learning models
	# shap.KernelExplainer works with all models
	# use Kernel SHAP to explain test set predictions
	k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
	k_shap_values = k_explainer.shap_values(data_for_prediction)
	shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)

Permutation importance的结果简单但不明了，缺少细节，SHAP可以提供更多地细节：

	import shap  # package used to calculate Shap values
	# Create object that can calculate shap values
	explainer = shap.TreeExplainer(my_model)
	# calculate shap values. This is what we will plot.
	# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
	shap_values = explainer.shap_values(val_X)
	# Make plot. We call shap_values[1] here to get the SHAP values for the prediction of "True".
	shap.summary_plot(shap_values[1], val_X)

同样地，SHAP对依赖图也能提供更多信息：

	import shap  # package used to calculate Shap values
	# Create object that can calculate shap values
	explainer = shap.TreeExplainer(my_model)
	# calculate shap values. This is what we will plot.
	shap_values = explainer.shap_values(X)
	# make plot. interaction_index is the one that may be interesting
	shap.dependence_plot(shap_values[1], X, interaction_index="A")