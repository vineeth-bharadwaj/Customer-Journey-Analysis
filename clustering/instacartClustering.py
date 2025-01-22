import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
from tabulate import tabulate
from sklearn.cluster         import KMeans, \
                                    AgglomerativeClustering, \
                                    Birch, \
                                    MiniBatchKMeans, \
                                    SpectralClustering, \
                                    AffinityPropagation, \
                                    MeanShift, \
                                    OPTICS, \
                                    DBSCAN
from sklearn.metrics         import silhouette_score, \
                                    davies_bouldin_score, \
                                    calinski_harabasz_score
import warnings
warnings.filterwarnings(action="ignore")


### Distribution Plots ###
'''
# my_histplot - plot histogram of a column against another category at x -axis
# my_distplot - plot distribution chart of a column against another category at x -axis
# my_kdeplot - plot kde graph of a column against another category at x -axis
# plot_bar_chart - plot a bar chart with the passed labels and values with some preset visual parameters (if not passed)
'''
def my_histplot(df, col, ax):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Histogram Plot of {col}')
def my_distplot(df, col, ax):
    sns.distplot(df[col], ax=ax)
    ax.set_title(f'Distribution Plot of {col}')
def my_kdeplot(df, col, ax):
    sns.kdeplot(df[col], ax=ax, fill=True)
    ax.set_title(f'KDE Plot of {col}')
def plot_bar_chart(df, x, y, xlabel, ylabel, title, xmin=None, xmax=None, palette='deep'):
    if df.shape[0] == 0:
        return
    size = (12, df.shape[0] / 4 + 1)
    plt.figure(figsize=size)
    sns.barplot(y=df[y], x=df[x], palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not xmin and not xmax:
        xmin, xmax = df[x].min(), df[x].max()
        xrange = (xmax - xmin) * .1
        xmin, xmax = xmin-xrange, xmax+xrange
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()

### Categorical Plots ###
'''
# my_pie_chart - plot pie chart of a column against another category at x -axis
# my_countplot - plot countplot of a column against another category at x -axis
# my_boxplot - plot boxplot of a column against another category at x -axis
# my_violinplot - plot violinplot of a column against another category at x -axis
'''
def my_pie_chart(df, col, ax):
    labels = df[col].value_counts()
    ax.pie(labels, labels=labels.index, autopct='%1.1f%%')
    ax.set_title(f'Pie Chart of {col}')
def my_countplot(df, col, ax):
    sns.countplot(x=df[col], ax=ax)
    ax.set_title(f'Count Plot of {col}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
def my_boxplot(df, col, ax):
    sns.boxplot(y=df[col], ax=ax)
def my_violinplot(df, col, ax):
    sns.violinplot(y=df[col], ax=ax)

### Matrix Plots ###
'''
# my_heatmap - plot a correlation heatmap
# plot_features - plot multiple features by calling the plot_feature function
# plot_feature - plot a single feature with the passed visual parameters
# plot_numerical_features - plot numerical features using distribution plots
# plot_categorical_features - plot categorical features using categorical plots
# plot_heatmap - use my_heatmap fucntion to plot the heatmap
'''
def my_heatmap(df, size, cmap, cbar_kws, font_size):
    plt.figure(figsize=size)
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap=cmap, center=0, cbar_kws=cbar_kws, annot_kws={"size": font_size})
    plt.title('Correlation Heatmap')
    plt.show()
def plot_features(df, plot_funcs, width_ratios, height_ratios, n_col=1):
    def plot_feature(cols):
        n_plot_funcs = len(plot_funcs)
        fig = plt.figure(figsize=(sum(width_ratios), max(height_ratios)))
        gs = fig.add_gridspec(1, n_plot_funcs * n_col, width_ratios=width_ratios, height_ratios=height_ratios)
        axes = [0] * (n_plot_funcs * n_col)

        for i in range(n_col):
            for j in range(n_plot_funcs):
                k = i * n_plot_funcs + j
                axes[k] = fig.add_subplot(gs[0, k])
                plot_funcs[j](df, cols[i], axes[k])
                axes[k].set_xlabel(cols[i])

        plt.tight_layout()
        plt.show()

    for i in range(0, len(df.columns), n_col):
        plot_feature(df.columns[i:i + n_col])
def plot_numerical_features(df, plot_funcs=[my_boxplot, my_violinplot, my_distplot], width_ratios=[2, 2, 12], height_ratios=[4], n_col=1):
    plot_features(df, plot_funcs, width_ratios, height_ratios)
def plot_categorical_features(df, plot_funcs=[my_pie_chart, my_countplot], width_ratios=[5, 11], height_ratios=[5], n_col=1):
    plot_features(df, plot_funcs, width_ratios, height_ratios)
def plot_heatmap(df, size_factor=1/2):
    df = df.select_dtypes(include=[np.number])
    height = int(len(df.columns) * size_factor)
    font_size = max(min(12, 119 // height), 8)
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        ['blue', 'lightblue', 'white', 'lightcoral', 'red'],
        N=5
    )
    cbar_kws = {'ticks': [-1, -.5, 0, .5, 1]}
    my_heatmap(df, size=(height+1, height+1), cmap=cmap, cbar_kws=cbar_kws, font_size=font_size)

### Processing Functions ###
'''
# df_dtypes - returns an organized information on the datatypes of the columns of the dataframe
# compute_dtype - returns a list of datatypes for each column in the dataframe
# compute_count - returns a list of counts for each column in the dataframe
# compute_mean - returns a list of mean of each column in the dataframe
# compute_std - returns a list of standard deviation of each column in the dataframe
# compute_min - returns a list of smallest value of each column in the dataframe
# compute_max - returns a list of largest value of each column in the dataframe
# compute_qunatile - returns a list of percentage quantile for each column in the dataframe
# compute_IQR - returns a list of interquartile range of each column in the dataframe
# compute_nunique - returns a list of count of unique values in each column in the dataframe
# compute_unique - returns a list of unique values in each column in the dataframe
# compute_mode - returns a list of modes of each column in the dataframe
# compute_mode - returns a list of count of most occuring value in each column in the dataframe
# compute_mode_percentage - returns a list of percentage count of most occuring value in each column in the dataframe
# compute_null_count - returns a list of count of null values in each column in the dataframe
# compute_null_percentage - returns a list of percentage count of null values in each column in the dataframe
# build_my_info_table: Creates a detailed summary table of the dataset's columns and their statistics.
# replace_inf: Replaces infinite values in the dataset with NaN and then fills them with the median.
# fillna: Fills missing values in numerical columns with the median and in categorical columns with the mode.
'''
def df_dtypes(df):
    pd.set_option('display.max_colwidth', None)

    df_dtypes = df.columns.groupby(df.dtypes)
    df_dtypes = pd.DataFrame({
        'dtype':     list(df_dtypes.keys()),
        '# columns': [len(df_dtypes[key])  for key in df_dtypes.keys()],
        'columns':   [list(df_dtypes[key]) for key in df_dtypes.keys()],
    })
    df_dtypes = df_dtypes.style.applymap(lambda x:'text-align: left', subset=['columns'])
    return df_dtypes
def compute_dtype(df):
    return [df[col].dtype for col in df.columns]
def compute_count(df):
    return [df[col].count() for col in df.columns]
def compute_mean(df, features):
    return [round(df[col].mean(), 1) if col in features else '' for col in df.columns]
def compute_std(df, features):
    return [round(df[col].std(), 1)         if col in features else '' for col in df.columns]
def compute_min(df, features):
    return [round(df[col].min(), 1) if col in features else '' for col in df.columns]
def compute_max(df, features):
    return [round(df[col].max(), 1) if col in features else '' for col in df.columns]
def compute_quantile(df, features, percentage):
    return [round(df[col].quantile(percentage), 1) \
                                            if col in features else '' for col in df.columns]
def compute_IQR(df, features):
    return [round(df[col].max()-df[col].min(), 1) \
                                            if col in features else '' for col in df.columns]
def compute_nunique(df):
    return [df[col].nunique() for col in df.columns]
def compute_unique(df, threshold):
    return [df[col].unique()  if df[col].nunique() < threshold else '' for col in df.columns]
def compute_mode(df):
    return [df[col].mode()[0] if len(df[col].mode()) else '' for col in df.columns]
def compute_mode_count(df):
    return [df[col].value_counts().max() for col in df.columns]
def compute_mode_percentage(df):
    return [round(df[col].value_counts().max() * 100 / df.shape[0], 1) for col in df.columns]
def compute_null_count(df):
    return [df[col].isnull().sum()                                     for col in df.columns]
def compute_null_percentage(df):
    return [round(df[col].isnull().mean() * 100, 1)                    for col in df.columns]
def build_my_info_table(df, nunique_threshold=30):
    numerical_features = df.select_dtypes(include=[np.number])
    df_info = pd.DataFrame({
        '#':        np.arange(len(df.columns)),
        'column':   df.columns,
        'dtype':    compute_dtype(df),
        'count':    compute_count(df),
        'mean':     compute_mean(df, numerical_features),
        'std':      compute_std(df, numerical_features),
        'min':      compute_min(df, numerical_features),
        '25%':      compute_quantile(df, numerical_features, .25),
        '50%':      compute_quantile(df, numerical_features, .5),
        '75%':      compute_quantile(df, numerical_features, .75),
        'max':      compute_max(df, numerical_features),
        'IQR':      compute_IQR(df, numerical_features),
        'nunique':  compute_nunique(df),
        'unique':   compute_unique(df, nunique_threshold),
        'mode':     compute_mode(df),
        'mode #':   compute_mode_count(df),
        'mode %':   compute_mode_percentage(df),
        'null #':   compute_null_count(df),
        'null %':   compute_null_percentage(df),
    })
    df_info = df_info.sort_values(by='dtype')
    return df_info
def fillna(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])
def replace_inf(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = df[numerical_features].replace([np.inf, -np.inf], np.nan)
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

### Outlier Management Functions ###
'''
# plot_feature_with_outlier: Visualizes a feature with and without detected outliers using specified plotting functions.
# outliers_iqr: Detects outliers based on the Interquartile Range (IQR) method.
# outliers_zscore: Identifies outliers using the Z-score method with a threshold.
# detect_outliers: Detects outliers in a feature using a specified method.
# plot_feature_with_outlier_methods: Compares feature distributions before and after outlier removal using multiple outlier detection methods.
# outlier_summary: Summarizes outlier detection results for numerical features using specified methods.
# get_outlier_indices: Retrieves and summarizes all outlier indices detected across numerical features using different methods.
'''
def plot_feature_with_outlier(df, plot_func, outliers_indices, outliers_methods, x_col, y_col):
    for i in range(len(outliers_methods)):
        fig, ax = plt.subplots(1, 2, figsize=(12 * 2, 4))
        ax[0].set_title('Original Data')
        ax[1].set_title(f'Data without Outliers of {y_col} using {outliers_methods[i].__name__}')
        plot_func(x=x_col, y=y_col, ax=ax[0], data=df)
        plot_func(x=x_col, y=y_col, ax=ax[1], data=df.drop(outliers_indices[i]))
        plt.tight_layout()
        plt.show()
def outliers_iqr(df, col):
    outliers = pd.DataFrame()
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    outliers = df.index[outliers]
    return outliers
def outliers_zscore(df, col, threshold=3):
    outliers = pd.DataFrame()
    z_scores = np.abs((df[col] - np.mean(df[col])) / np.std(df[col]))
    outliers = z_scores > threshold
    outliers = df.index[outliers]
    return outliers
def detect_outliers(df, target_feature, outliers_method):
    outliers_indices = [item for item in outliers_method(df, target_feature)]
    return outliers_indices
def plot_feature_with_outlier_methods(df, target_feature, outliers_methods, plot_func):
    outliers = [0] * len(outliers_methods)
    for i in range(len(outliers_methods)):
        outliers[i] = detect_outliers(df, target_feature, outliers_methods[i])
        print(f'length of {target_feature} outliers_indices: {len(outliers[i])} using {outliers_methods[i].__name__}')
    if sum(len(curr_outliers) for curr_outliers in outliers) == 0:
        return
    for curr_col in df.columns:
        if curr_col == target_feature:
            continue
        plot_feature_with_outlier(df, plot_func, outliers, outliers_methods, x_col=curr_col, y_col=target_feature)
def outlier_summary(df, method1, method2):
    numerical_features = get_numerical_features(df)
    df = df[numerical_features]
    outlier_info = pd.DataFrame({
        '#': np.arange(len(df.columns)),
        'column': df.columns,
        'dtype': compute_dtype(df),
        'count': compute_count(df),
        'mean': compute_mean(df, numerical_features),
        'std': compute_std(df, numerical_features),
        'min': compute_min(df, numerical_features),
        '25%': compute_quantile(df, numerical_features, .25),
        '50%': compute_quantile(df, numerical_features, .5),
        '75%': compute_quantile(df, numerical_features, .75),
        'max': compute_max(df, numerical_features),
        'IQR': compute_IQR(df, numerical_features),

        f'{method1.__name__}': [list(method1(df, col)) for col in numerical_features],
        f'{method2.__name__}': [list(method2(df, col)) for col in numerical_features],
    })

    outlier_info[f'{method1.__name__} length'] = outlier_info[f'{method1.__name__}'].apply(len)
    outlier_info[f'{method2.__name__} length'] = outlier_info[f'{method2.__name__}'].apply(len)

    if outlier_info[f'{method1.__name__} length'].shape[0] == 0 or \
            outlier_info[f'{method2.__name__} length'].shape[0] == 0:
        outlier_info['all_indices'] = pd.Series([set() for _ in range(len(outlier_info))],
                                                index=outlier_info.index, dtype='object')
    else:
        outlier_info['all_indices'] = outlier_info.apply(lambda x: set(x[f'{method1.__name__}']) & \
                                                                   set(x[f'{method2.__name__}']), axis=1)
    outlier_info['all_indices length'] = outlier_info['all_indices'].apply(len)

    outlier_info = outlier_info.sort_values(by='all_indices length', ascending=False)
    outlier_info = outlier_info[outlier_info['all_indices length'] > 0]
    return outlier_info
def get_outlier_indices(outlier_info, method1, method2):
    col = 'all_indices'
    all_indices = set()
    for curr_indices in outlier_info[col]:
        all_indices |= curr_indices
    outlier_percentage = round(len(all_indices) / outlier_info['count'].max() * 100, 1)
    print(f'Total indices among all numerical features are: {len(all_indices)} ({outlier_percentage} %) of the dataset')
    return list(all_indices)

### Imbalanced Feature Management Functions ###
'''
# oversampling_imbalanced_data: Balances an imbalanced dataset using the SMOTE oversampling technique.
# plot_imbalanced_feature: Visualizes class distributions before and after applying oversampling to a single feature.
# plot_imbalanced_features: Visualizes class distributions for multiple features, showing the effect of oversampling.
# oversampling_data: Applies oversampling to balance multiple imbalanced features in a dataset.
'''
def oversampling_imbalanced_data(df, target_feature, k_neighbors=5):
    if k_neighbors >= min(Counter(df[target_feature]).values()):
        return df
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]
    oversampler = SMOTE(k_neighbors=k_neighbors)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.DataFrame(y_resampled, columns=[target_feature])], axis=1)
    return df_resampled
def plot_imbalanced_feature(df, df_resampled, target_feature):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(x=df[target_feature])
    plt.title('Class Distribution Before Oversampling')
    plt.xlabel(target_feature)
    plt.subplot(1, 2, 2)
    sns.countplot(x=df_resampled[target_feature])
    plt.title('Class Distribution After Oversampling')
    plt.xlabel(target_feature)

    plt.tight_layout()
    plt.show()
def plot_imbalanced_features(df, features):
    for col in features:
        if df[col].nunique() == 1:
            continue
        df_resampled = oversampling_imbalanced_data(df, col)
        if df.shape != df_resampled.shape:
            print(f'col: {col}, df.shape: {df.shape}, df_resampled.shape: {df_resampled.shape}')
            plot_imbalanced_feature(df, df_resampled, col)
def oversampling_data(df, features):
    for col in features:
        if df[col].nunique() == 1:
            continue
        df_resampled = oversampling_imbalanced_data(df, col)
        df = df_resampled.copy()
    return df

### Feature Extraction functions ###
'''
# get_categorical_features - Extract categorical features
# get_numerical_features - Extract numerical features
# drop_id_feature - Drop features that correspond unique identifiers
# encode_str_features - encode string features into numerical features
# one_hot_encoding - implement One Hot Encoding
# get_skewed_features - Extract skewed features
# transform_skewed_features: Applies a logarithmic transformation to reduce skewness in highly skewed features.
# transform_non_skewed_features: Standardizes non-skewed features using a standard scaler to normalize their values.
'''
def get_categorical_features(df, nunique_threshold=40):
    all_df_columns = df.columns
    categorical_features = [col for col in all_df_columns if df[col].nunique() < nunique_threshold]
    return categorical_features
def get_numerical_features(df, nunique_threshold=40):
    numerical_df_columns = df.select_dtypes(include=[np.number]).columns
    numerical_features   = [col for col in numerical_df_columns if df[col].nunique() >= nunique_threshold]
    return numerical_features
def drop_id_feature(df, id_col='ID'):
    df_id = df[id_col]
    df = df.drop(columns=[id_col])
    return df, df_id
def encode_str_features(df):
    categorical_features = get_categorical_features(df.select_dtypes(exclude=[np.number]))
    label_encoder = LabelEncoder()
    for col in categorical_features:
        df[col] = label_encoder.fit_transform(df[col])
def one_hot_encoding(df):
    categorical_features = get_categorical_features(df.select_dtypes(include=[np.number]))
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features), index=df.index)
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    return df
def get_skewed_features(df, threshold=0.25):
    numerical_features = get_numerical_features(df)
    skew_df = df[numerical_features].apply(lambda x: x.skew())
    skew_df = skew_df.sort_values(ascending=False)
    skew_df = skew_df.reset_index()
    skew_df.columns = ['Feature', 'SkewFactor']
    skewed_features = list(skew_df[abs(skew_df['SkewFactor']) > threshold]['Feature'])
    non_skewed_features = list(set(numerical_features)-set(skewed_features))
    return skewed_features, non_skewed_features, skew_df
def transform_skewed_features(df, skewed_features):
    for col in skewed_features:
        df[col] = np.log1p(df[col])
def transform_non_skewed_features(df, non_skewed_features):
    for col in non_skewed_features:
        df[col] = standardScaler.fit_transform(df[[col]])

### Feature Selection Functions ###
def feature_selection(df, mutual_info_method, top_n=10):
    X = df.select_dtypes(include=[np.number])
    importance_df = []
    for feature in X.columns:
        mi_scores = []
        for other_feature in X.columns:
            if feature != other_feature:
                mi = mutual_info_method(X[[feature]], X[other_feature])
                mi_scores.append(mi[0])
        importance_df.append((feature, np.mean(mi_scores)))

    importance_df = pd.DataFrame(importance_df, columns=['Feature', 'ImportanceFactor'])
    importance_df = importance_df.sort_values(by='ImportanceFactor', ascending=False)
    importance_df = importance_df.head(top_n)
    return importance_df

### Metrics Functions ###
'''
# wcss_score: Calculates the Within-Cluster Sum of Squares (WCSS) to measure cluster compactness.
# cohesion_separation: Computes the cohesion and separation metrics for clustering.
# dunn_index: Evaluates cluster quality by calculating the Dunn Index, which is the ratio of the smallest inter-cluster distance to the largest intra-cluster distance.
'''
def wcss_score(X, labels, cluster_centers):
    wcss = 0
    for i, center in enumerate(cluster_centers):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] > 0:
            distances = np.sum((cluster_points.values - center) ** 2, axis=1)
            wcss += np.sum(distances)
    return wcss
def cohesion_separation(X, labels, cluster_centers):
    cohesion, separation = 0, 0
    for i, center in enumerate(cluster_centers):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] > 0:
            cohesion += np.sum((cluster_points.values - center) ** 2)
        for j, other_center in enumerate(cluster_centers):
            if i != j:
                separation += np.sum((center - other_center) ** 2)
    return cohesion, separation
def dunn_index(X, labels, cluster_centers):
    min_intercluster_distance = np.inf
    max_intracluster_distance = 0
    for i, center in enumerate(cluster_centers):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] > 0:
            intracluster_distances = np.linalg.norm(cluster_points.values - center, axis=1)
            max_intracluster_distance = max(max_intracluster_distance, np.max(intracluster_distances))
        for j, other_center in enumerate(cluster_centers):
            if i != j:
                intercluster_distance = np.linalg.norm(center - other_center)
                min_intercluster_distance = min(min_intercluster_distance, intercluster_distance)
    return min_intercluster_distance / max_intracluster_distance if max_intracluster_distance != 0 else 0

#ML Model Functions
'''
# evaluate_model - evaluate the trained clustering models
# get_cluster_centres - determine the centroids of clusters
# run_models - run the the clustering models
# get_best_models - determine the model with the best evaluation results
'''
def evaluate_model(X, labels, cluster_centers):
    shi  = silhouette_score(X, labels)
    dbi  = davies_bouldin_score(X, labels)
    chi  = calinski_harabasz_score(X, labels)
    wcss = wcss_score(X, labels, cluster_centers)
    coh, sep = cohesion_separation(X, labels, cluster_centers)
    dunn = dunn_index(X, labels, cluster_centers)

    result = {
        'Silhouette': shi,
        'Davies-Bouldin Index': dbi,
        'Calinski-Harabasz Index': chi,
        'WCSS': wcss,
        'Cohesion': coh,
        'Separation': sep,
        'Dunn Index': dunn,
    }
    return result
def get_cluster_centers(model, X, labels):
    if hasattr(model, 'cluster_centers_'):
        return model.cluster_centers_
    else:
        unique_labels = np.unique(labels)
        cluster_centers = np.zeros((len(unique_labels), X.shape[1]))
        for label in unique_labels:
            cluster_centers[label] = X[labels == label].mean(axis=0)
        return cluster_centers
def run_models(models, X):
    results = {}
    for name, model in models.items():
        print(f'Model {name} begining now ...')
        begin_time = time.time()
        labels = model.fit_predict(X)
        cluster_centers = get_cluster_centers(model, X, labels)
        results[name] = evaluate_model(X, labels, cluster_centers)
        end_time = time.time()
        duration = round((end_time - begin_time) / 60, 2)
        print(f'Model {name} run in'.ljust(50), f'{duration} minutes')
    results = pd.DataFrame(results).T
    results = results.reset_index()
    results = results.rename(columns={'index': 'Model'})
    return results
def get_best_model(results_df, models, metric):
    best_model_name = results_df.sort_values(by=[metric]).head(1)['Model'].iloc[0]
    best_model = models[best_model_name]
    return best_model


# Load the dataseta (aisles, departments, products, orders, order_products__prior, order_products__train)
dir_path = "instacartmarketbasket/"
order_products_train = pd.read_csv(f'{dir_path}/order_products__train.csv')
order_products_prior = pd.read_csv(f'{dir_path}/order_products__prior.csv')
orders      = pd.read_csv(f'{dir_path}/orders.csv')
products    = pd.read_csv(f'{dir_path}/products.csv')
aisles      = pd.read_csv(f'{dir_path}/aisles.csv')
departments = pd.read_csv(f'{dir_path}/departments.csv')

# Merge different dataframes
full_products = products.copy()
full_products = pd.merge(full_products, aisles, on='aisle_id', how='left')
full_products = pd.merge(full_products, departments, on='department_id', how='left')
order_products_train = pd.merge(order_products_train, orders, on='order_id', how='left')
order_products_prior = pd.merge(order_products_prior, orders, on='order_id', how='left')

# Obtain the train and test dataframes
train = pd.merge(order_products_train, full_products, on='product_id', how='left')
test = pd.merge(order_products_prior, full_products, on='product_id', how='left')

### Preprocessing ###
# Preprocessing test dataframe
train_info = build_my_info_table(train)

# Finding columns with a high percentage of missing values
nan_df = train_info[train_info['null %'] >= 10][['column', 'null %']].sort_values(by='null %')

# Plotting the missing values for visualization
plot_bar_chart(nan_df, x='null %', y='column', xlabel='Null Percentage %', ylabel='Feature', title='Null Percentage in each Feature', xmin=0, xmax=100, palette='coolwarm')

# Dropping columns with more than 50% missing values and noting how many were dropped
columns_to_drop = list(nan_df[nan_df['null %'] > 50]['column'])
print(f"Dropping {len(columns_to_drop)} columns from train dataframe due to high percentage of missing values: {columns_to_drop}")
train = train.drop(columns=columns_to_drop)

# Preprocessing test dataframe
test = test.drop(columns=columns_to_drop)
test_info = build_my_info_table(test)

# Finding columns with a high percentage of missing values in the test set
nan_df = test_info[test_info['null %'] >= 10][['column', 'null %']].sort_values(by='null %')

# Plotting the missing values for visualization
plot_bar_chart(nan_df, x='null %', y='column', xlabel='Null Percentage %', ylabel='Feature', title='Null Percentage in each Feature', xmin=0, xmax=100, palette='coolwarm')

# Dropping columns with more than 50% missing values and noting how many were dropped
columns_to_drop = list(nan_df[nan_df['null %'] > 50]['column'])
print(f"Dropping {len(columns_to_drop)} columns from test dataframe due to high percentage of missing values: {columns_to_drop}")
train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)

# Replacing infinite values with NaN and filling missing values
replace_inf(train)
replace_inf(test)

# Filling missing values with median for numerical and mode for categorical
fillna(train)
fillna(test)

print(f"Train dataframe shape after preprocessing: {train.shape}")
print(f"Test dataframe shape after preprocessing: {test.shape}")

### Extrating features from train and test dataframes ###
# Identifying categorical and numerical features
categorical_features = get_categorical_features(train)
numerical_features = get_numerical_features(train)

# Printing the count and names of categorical and numerical features
print(f'Number of categorical features: {len(categorical_features)}')
print(f'Categorical features: {categorical_features}\n')
print(f'Number of numerical features: {len(numerical_features)}')
print(f'Numerical features: {numerical_features}\n')

# Printing the shape of the train and test dataframes
print(f'Train dataframe shape before sampling: {train.shape}')
print(f'Test dataframe shape before sampling: {test.shape}')

# Shuffling the data and resetting the index
train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

# Sampling 5000 records from each dataframe
train = train.sample(5000)
test = test.sample(5000)

print(f'Train dataframe shape after sampling: {train.shape}')
print(f'Test dataframe shape after sampling: {test.shape}')

print("First 10 rows of numerical features from the train dataframe:")
#print(train.select_dtypes(include=[np.number]).head(10))
print(tabulate(train.select_dtypes(include=[np.number]).head(10), headers='keys', tablefmt='psql'))

### Initial Model Training and Evaluation ###

train_model2 = train.copy()
test_model2  = test.copy()

# Drop ID columns
train_model2, _ = drop_id_feature(train_model2, 'product_id')
train_model2, _ = drop_id_feature(train_model2, 'user_id')
train_model2, _ = drop_id_feature(train_model2, 'aisle_id')
train_model2, _ = drop_id_feature(train_model2, 'order_id')
test_model2, _ = drop_id_feature(test_model2, 'product_id')
test_model2, _ = drop_id_feature(test_model2, 'user_id')
test_model2, _ = drop_id_feature(test_model2, 'aisle_id')
test_model2, _ = drop_id_feature(test_model2, 'order_id')

# Encode String features
encode_str_features(train_model2)
encode_str_features(test_model2)
print(df_dtypes(train_model2))
# Combine train and test dataframes and perform One Hot Encoding
train_test_model2 = pd.concat([train_model2, test_model2])
train_test_model2 = one_hot_encoding(train_test_model2)

# Split the one hot encoded train-test dataframe
train_model2 = train_test_model2[:len(train_model2)]
test_model2  = train_test_model2[-len(test_model2):]
#print(df_dtypes(train_model2))
print(df_dtypes(train_model2))

# Drop non numeric columns
train_model2 = train_model2.drop(columns=train_model2.select_dtypes(exclude=[np.number]).columns)
test_model2  = test_model2.drop(columns=test_model2.select_dtypes(exclude=[np.number]).columns)
#print(df_dtypes(train_model2))
print(df_dtypes(train_model2))

### Standardize feature values ###
standardScaler = StandardScaler()

# Standardizing train dataframe
skewed_features, non_skewed_features, skew_df = get_skewed_features(train_model2) # Extract skewed and non-skewed features
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}\n')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}\n')
plot_bar_chart(skew_df, x='SkewFactor', y='Feature', xlabel='Skew Factor', ylabel='Feature', title='Skew Factor in each Feature', palette='coolwarm')

# Standardize skewed and non-skewed features
transform_skewed_features(train_model2, skewed_features)
transform_non_skewed_features(train_model2, non_skewed_features)

# Standardizing test dataframe
skewed_features, non_skewed_features, skew_df = get_skewed_features(test_model2)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}\n')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}\n')
transform_skewed_features(test_model2, skewed_features)
transform_non_skewed_features(test_model2, non_skewed_features)
print(df_dtypes(train_model2))
replace_inf(train_model2)
replace_inf(test_model2)
fillna(train_model2)
fillna(test_model2)

# Feature Importance and Selection
importance_df = feature_selection(train_model2, mutual_info_regression)
plot_bar_chart(importance_df, x='ImportanceFactor', y='Feature', xlabel='Importance Factor', ylabel='Feature', title='Importance Factor in each Feature', palette='coolwarm')
selected_features = importance_df['Feature']
train_model2 = train_model2[selected_features]
test_model2  = test_model2[selected_features]
df_dtypes(train_model2)
plot_heatmap(train_model2)

# Baseline models considered here
baseline_models = {
    'KMeans':                     KMeans(n_clusters=3),
    'AgglomerativeClustering':    AgglomerativeClustering(n_clusters=3),
    'Birch':                      Birch(n_clusters=3),
    'MiniBatchKMeans':            MiniBatchKMeans(n_clusters=3),
    'SpectralClustering':         SpectralClustering(n_clusters=3,
                                                     affinity='nearest_neighbors',
                                                     n_neighbors=5),
}

models_result2 = run_models(baseline_models, train_model2)
print(models_result2)
print(tabulate(models_result2, headers='keys', tablefmt='psql'))
metrics = ['Silhouette', 'Davies-Bouldin Index', 'Calinski-Harabasz Index', 'WCSS', 'Cohesion', 'Separation', 'Dunn Index']
for metric in metrics:
    plot_bar_chart(models_result2, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")
best_models2 = get_best_model(models_result2, baseline_models, 'WCSS')
print('Best Model of Enhanced Features Models is:', best_models2.__class__.__name__)
del train_model2, test_model2


### Apply enhancements by taking Outliers and Imbalanced Features into consideration and then apply clustering ###
train_model4 = train.copy()
test_model4  = test.copy()
encode_str_features(train_model4)
encode_str_features(test_model4)
print(df_dtypes(train_model4))

# Outlier Management
# Extract the outliers in the data
outlier_info = outlier_summary(train_model4, outliers_iqr, outliers_zscore)
outlier_indices = get_outlier_indices(outlier_info, outliers_iqr, outliers_zscore)
print('Dataset shape before drop the outliers', train_model4.shape)

# Drop the outliers
train_model4 = train_model4.drop(index=outlier_indices)
print('Dataset shape after drop the outliers ', train_model4.shape)

# Drop ID columns
train_model4, _ = drop_id_feature(train_model4, 'product_id')
train_model4, _ = drop_id_feature(train_model4, 'user_id')
train_model4, _ = drop_id_feature(train_model4, 'aisle_id')
train_model4, _ = drop_id_feature(train_model4, 'order_id')
test_ids = test_model4[['order_id', 'product_id']]
test_model4, _ = drop_id_feature(test_model4, 'product_id')
test_model4, _ = drop_id_feature(test_model4, 'user_id')
test_model4, _ = drop_id_feature(test_model4, 'aisle_id')
test_model4, _ = drop_id_feature(test_model4, 'order_id')

# Combine train and test data for One Hot Encoding and split them
train_test_model4 = pd.concat([train_model4, test_model4])
train_test_model4 = one_hot_encoding(train_test_model4) # One Hot Encoding
train_model4 = train_test_model4[:len(train_model4)]
test_model4  = train_test_model4[-len(test_model4):]
print(df_dtypes(train_model4))

# Drop non-numeric columns and retain numeric columns
train_model4 = train_model4.drop(columns=train_model4.select_dtypes(exclude=[np.number]).columns)
test_model4  = test_model4.drop(columns=test_model4.select_dtypes(exclude=[np.number]).columns)
print(df_dtypes(train_model4))

# Imbalanced Feature Management
oversamplying_features = get_categorical_features(train_model4.select_dtypes(exclude=['float']), 4)
print(tabulate(build_my_info_table(train_model4[oversamplying_features]), headers='keys', tablefmt='psql'))
plot_imbalanced_features(train_model4, oversamplying_features)
print('Dataset shape before apply the oversampling', train_model4.shape)
train_model4 = oversampling_data(train_model4, oversamplying_features)
print('Dataset shape after apply the oversampling ', train_model4.shape)
replace_inf(train_model4)
replace_inf(test_model4)
fillna(train_model4)
fillna(test_model4)

# Feature Selection
importance_df = feature_selection(train_model4, mutual_info_regression, 20)
selected_features = importance_df['Feature']
train_model4 = train_model4[selected_features]
test_model4  = test_model4[selected_features]
#print(df_dtypes(train_model4))

# Model training and Evaluation
models_result4 = run_models(baseline_models, train_pca)
print(tabulate(models_result4, headers='keys', tablefmt='psql'))

for metric in metrics:
    plot_bar_chart(models_result4, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")
best_models4 = get_best_model(models_result4, baseline_models, 'WCSS')
print('Best Model of Combine All Enhancements Models is:', best_models4.__class__.__name__)
