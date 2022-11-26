import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split



def get_data(filename) -> pd.DataFrame:
    with open(f'data/{filename}.csv', 'r', encoding='utf-8') as file:
        data_frame = pd.read_csv(file, index_col=0, low_memory=False) * 1
        return data_frame


def normalize_person_data(df) -> pd.DataFrame:
    normal_data = df[['person_id']]
    to_normalize = df.drop(['date', 'person_id'], axis=1).apply(lambda x: pd.factorize(x)[0])
    to_normalize = (to_normalize - to_normalize.min()) / (to_normalize.max() - to_normalize.min())
    df = pd.concat([normal_data, to_normalize], axis=1)
    return df


def normalize_action_data(df) -> pd.DataFrame:
    normal_data = df[['person_id', 'action_type', 'action_id']]
    to_normalize = df.drop(['person_id', 'date', 'action_type', 'action_id'], axis=1).apply(
        lambda x: pd.factorize(x)[0])
    to_normalize = (to_normalize - to_normalize.min()) / (to_normalize.max() - to_normalize.min())
    df = pd.concat([normal_data, to_normalize], axis=1)
    df = pd.get_dummies(df, columns=["action_type"])
    return df


def get_redundant_pairs(df: pd.DataFrame) -> set:
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr().abs()


def get_top_abs_correlations(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def draw_importances(train_df: pd.DataFrame, rfc: RandomForestClassifier) -> None:
    feats = {}
    for feature, importance in zip(train_df.columns, rfc.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale=5)
    sns.set(style="whitegrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight='bold')
    plt.ylabel('Features', fontsize=25, weight='bold')
    plt.title('Feature Importance', fontsize=25, weight='bold')
    plt.show()


# test_df = normalize_action_data(get_data('action_test'))
train_df = normalize_action_data(get_data('action_train'))
person_df = normalize_person_data(get_data('person'))

train_df = pd.merge(train_df, person_df, on='person_id', suffixes=('_action', '_person')).set_index(
    ['action_id', 'person_id'])

X = train_df.drop(['result'], axis=1)
y = train_df.result

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
y_train = np.array(y_train)

clf = RandomForestClassifier()
n_estimators = [300, 500, 700]
max_features = ['sqrt']
max_depth = [2, 3, 7, 11, 15]
min_samples_split = [2, 3, 4, 22, 23, 24]
min_samples_leaf = [2, 3, 4, 5, 6, 7]
bootstrap = [False]
grid_search_cv_parameters = {'n_estimators': n_estimators,
                             'max_features': max_features,
                             'max_depth': max_depth,
                             'min_samples_split': min_samples_split,
                             'min_samples_leaf': min_samples_leaf,
                             'bootstrap': bootstrap}
grid_search_cv_clf = GridSearchCV(clf, grid_search_cv_parameters, cv=5)

n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}
random_search_clf = RandomizedSearchCV(clf,
                                       param_dist,
                                       n_iter=100,
                                       cv=3,
                                       verbose=1,
                                       n_jobs=-1,
                                       random_state=0)
grid_search_cv_clf.fit(X_train_scaled, y_train)
random_search_clf.fit(X_train_scaled, y_train)

best_grid_clf = grid_search_cv_clf.best_estimator_
best_random_clf = random_search_clf.best_estimator_

y_pred_best_grid_clf = best_grid_clf.predict(X_test_scaled)
y_pred_best_random_clf = clf.predict(X_test_scaled)

conf_matrix_baseline_best_clf = pd.DataFrame(confusion_matrix(y_test, y_pred_best_grid_clf),
                                             index=['actual 0', 'actual 1'],
                                             columns=['predicted 0', 'predicted 1'])
print(conf_matrix_baseline_best_clf)
print('Baseline Random Forest recall score Best classifier', recall_score(y_test, y_pred_best_grid_clf))

conf_matrix_baseline_best_random_search_clf = pd.DataFrame(confusion_matrix(y_test, y_pred_best_random_clf),
                                                           index=['actual 0', 'actual 1'],
                                                           columns=['predicted 0', 'predicted 1'])
print(conf_matrix_baseline_best_random_search_clf)
print('Baseline Random Forest recall score regular classifier', recall_score(y_test, y_pred_best_random_clf))
