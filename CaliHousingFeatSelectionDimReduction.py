import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector_f = SelectKBest(score_func=f_regression, k=5)
X_train_f = selector_f.fit_transform(X_train_scaled, y_train)
X_test_f = selector_f.transform(X_test_scaled)

selector_m = SelectKBest(score_func=mutual_info_regression, k=5)
X_train_m = selector_m.fit_transform(X_train_scaled, y_train)
X_test_m = selector_m.transform(X_test_scaled)

pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

models = {
    "Original": LinearRegression(),
    "SelectKBest_F": LinearRegression(),
    "SelectKBest_MI": LinearRegression(),
    "PCA": LinearRegression()
}

for name, model in models.items():
    if name == "SelectKBest_F":
        X_t, X_te = X_train_f, X_test_f
    elif name == "SelectKBest_MI":
        X_t, X_te = X_train_m, X_test_m
    elif name == "PCA":
        X_t, X_te = X_train_pca, X_test_pca
    else:
        X_t, X_te = X_train_scaled, X_test_scaled

    model.fit(X_t, y_train)
    y_pred = model.predict(X_te)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse}, R2: {r2}")
