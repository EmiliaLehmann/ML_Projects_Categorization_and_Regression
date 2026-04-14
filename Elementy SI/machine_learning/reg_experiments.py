import os
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_regression_data, inverse_scale_temperature


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # cofnięcie skalowania do stopni Celsjusza
    y_train_true_original = inverse_scale_temperature(y_train, data_min, data_max)
    y_test_true_original = inverse_scale_temperature(y_test, data_min, data_max)

    y_train_pred_original = inverse_scale_temperature(y_train_pred, data_min, data_max)
    y_test_pred_original = inverse_scale_temperature(y_test_pred, data_min, data_max)

    results = {
        "train_mae": mean_absolute_error(y_train_true_original, y_train_pred_original),
        "test_mae": mean_absolute_error(y_test_true_original, y_test_pred_original),

        "train_mse": mean_squared_error(y_train_true_original, y_train_pred_original),
        "test_mse": mean_squared_error(y_test_true_original, y_test_pred_original),

        "train_rmse": np.sqrt(mean_squared_error(y_train_true_original, y_train_pred_original)),
        "test_rmse": np.sqrt(mean_squared_error(y_test_true_original, y_test_pred_original)),

        "train_r2": r2_score(y_train_true_original, y_train_pred_original),
        "test_r2": r2_score(y_test_true_original, y_test_pred_original),
    }

    return results


def save_results(results_list, output_path="results/regression_results.csv"):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)
    print(f"Wyniki zapisane do: {output_path}")


def add_result(results, model_name, parameter_name, parameter_value, metrics):
    results.append({
        "model": model_name,
        "parameter_name": parameter_name,
        "parameter_value": str(parameter_value),
        **metrics
    })

def save_result_to_txt(model_name, param_name, param_value, metrics):
    with open("results/log_regression.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "="*40 + "\n")
        f.write("--- NOWY TEST MODELU ---\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Parametr: {param_name}\n")
        f.write(f"Wartość: {param_value}\n")

        f.write(f"MAE: {metrics['test_mae']:.4f}\n")
        f.write(f"MSE: {metrics['test_mse']:.4f}\n")
        f.write(f"RMSE: {metrics['test_rmse']:.4f}\n")
        f.write(f"R2: {metrics['test_r2']:.4f}\n")

        f.write("-" * 40 + "\n")


def run_regression_experiments():
    open("results/log_regression.txt", "w").close()

    data = load_regression_data(window_size=12, test_size=0.2)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    data_min = data["data_min"]
    data_max = data["data_max"]

    results = []

    print("Start eksperymentów regresyjnych")

    # =========================
    # KNN Regressor
    # =========================
    knn_neighbors = [3, 5, 7, 9]
    for n in knn_neighbors:
        print(f"KNN Regressor | n_neighbors={n}")
        model = KNeighborsRegressor(
            n_neighbors=n,
            metric="minkowski",
            p=2
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "KNNRegressor", "n_neighbors", n, metrics)
        save_result_to_txt("KNNRegressor", "n_neighbors", n, metrics)

    knn_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    for metric_name in knn_metrics:
        print(f"KNN Regressor | metric={metric_name}")
        model = KNeighborsRegressor(
            n_neighbors=5,
            metric=metric_name,
            p=2
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "KNNRegressor", "metric", metric_name, metrics)
        save_result_to_txt("KNNRegressor", "metric", metric_name, metrics)

    knn_p_values = [1, 2, 3, 4]
    for p_value in knn_p_values:
        print(f"KNN Regressor | p={p_value}")
        model = KNeighborsRegressor(
            n_neighbors=5,
            metric="minkowski",
            p=p_value
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "KNNRegressor", "p", p_value, metrics)
        save_result_to_txt("KNNRegressor", "p", p_value, metrics)

    # =========================
    # Decision Tree Regressor
    # =========================
    tree_depths = [3, 5, 10, 15]
    for depth in tree_depths:
        print(f"Decision Tree Regressor | max_depth={depth}")
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "DecisionTreeRegressor", "max_depth", depth, metrics)
        save_result_to_txt("DecisionTreeRegressor", "max_depth", depth, metrics)

    tree_min_samples_split = [2, 5, 10, 20]
    for min_split in tree_min_samples_split:
        print(f"Decision Tree Regressor | min_samples_split={min_split}")
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=min_split,
            min_samples_leaf=1,
            random_state=42
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "DecisionTreeRegressor", "min_samples_split", min_split, metrics)
        save_result_to_txt("DecisionTreeRegressor", "min_samples_split", min_split, metrics)

    tree_min_samples_leaf = [1, 2, 4, 8]
    for leaf in tree_min_samples_leaf:
        print(f"Decision Tree Regressor | min_samples_leaf={leaf}")
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=leaf,
            random_state=42
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "DecisionTreeRegressor", "min_samples_leaf", leaf, metrics)
        save_result_to_txt("DecisionTreeRegressor", "min_samples_leaf", leaf, metrics)

    # =========================
    # Random Forest Regressor
    # =========================
    forest_estimators = [50, 100, 200, 300]
    for n_estimators in forest_estimators:
        print(f"Random Forest Regressor | n_estimators={n_estimators}")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "RandomForestRegressor", "n_estimators", n_estimators, metrics)
        save_result_to_txt("RandomForestRegressor", "n_estimators", n_estimators, metrics)

    forest_depths = [5, 10, 15, None]
    for depth in forest_depths:
        print(f"Random Forest Regressor | max_depth={depth}")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "RandomForestRegressor", "max_depth", depth, metrics)
        save_result_to_txt("RandomForestRegressor", "max_depth", depth, metrics)

    forest_min_leaf = [1, 2, 4, 8]
    for min_leaf in forest_min_leaf:
        print(f"Random Forest Regressor | min_samples_leaf={min_leaf}")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "RandomForestRegressor", "min_samples_leaf", min_leaf, metrics)
        save_result_to_txt("RandomForestRegressor", "min_samples_leaf", min_leaf, metrics)

    # =========================
    # SVR
    # =========================
    svr_c_values = [0.1, 1, 10, 100]
    for c_value in svr_c_values:
        print(f"SVR | C={c_value}")
        model = SVR(
            C=c_value,
            kernel="rbf",
            gamma="scale"
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "SVR", "C", c_value, metrics)
        save_result_to_txt("SVR", "C", c_value, metrics)

    svr_kernels = ["linear", "poly", "rbf", "sigmoid"]
    for kernel_name in svr_kernels:
        print(f"SVR | kernel={kernel_name}")
        model = SVR(
            C=1.0,
            kernel=kernel_name,
            gamma="scale"
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "SVR", "kernel", kernel_name, metrics)
        save_result_to_txt("SVR", "kernel", kernel_name, metrics)

    svr_gamma_values = ["scale", "auto", 0.01, 0.1]
    for gamma_value in svr_gamma_values:
        print(f"SVR | gamma={gamma_value}")
        model = SVR(
            C=1.0,
            kernel="rbf",
            gamma=gamma_value
        )
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, data_min, data_max)
        add_result(results, "SVR", "gamma", gamma_value, metrics)
        save_result_to_txt("SVR", "gamma", gamma_value, metrics)

    # zapis wyników
    save_results(results)

    df_results = pd.DataFrame(results)
    best_result = df_results.loc[df_results["test_rmse"].idxmin()]

    print("\nNajlepszy wynik regresji:")
    print(best_result)


if __name__ == "__main__":
    run_regression_experiments()