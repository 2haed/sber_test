import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from joblib import parallel_backend
from ray.util.joblib import register_ray


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


test_df = normalize_action_data(get_data('action_test'))
train_df = normalize_action_data(get_data('action_train'))
person_df = normalize_person_data(get_data('person'))

test_df = pd.merge(test_df, person_df, on='person_id', suffixes=('_action', '_person')).set_index(
    ['action_id', 'person_id'])
train_df = pd.merge(train_df, person_df, on='person_id', suffixes=('_action', '_person')).set_index(
    ['action_id', 'person_id'])

X_train = train_df.drop(['result'], axis=1)
y_train = train_df.result
X_test = test_df

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

y_train = np.array(y_train)

clf = RandomForestClassifier()
parameters = {'n_estimators': [5, 10, 15], 'max_depth': [2, 5, 7, 10]}
grid_search_cv_clf = GridSearchCV(clf, parameters, cv=5)
grid_search_cv_clf.fit(X_train_scaled, y_train)
best_clf = grid_search_cv_clf.best_estimator_
y_pred_best_clf = best_clf.predict(X_train_scaled)
y_pred_clf = clf.predict(X_train_scaled)

conf_matrix_baseline = pd.DataFrame(confusion_matrix(y_train, y_pred_best_clf), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
print(conf_matrix_baseline)
print('Baseline Random Forest recall score', recall_score(y_train, y_pred_best_clf))







