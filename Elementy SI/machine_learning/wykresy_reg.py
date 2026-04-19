import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data(csv_path="results/regression_results.csv"):
    df = pd.read_csv(csv_path)
    return df

# =========================
# KNN - n_neighbors
# =========================
def plot_knn_n_neighbors(df):
    data = df[
        (df["model"] == "KNNRegressor") &
        (df["parameter_name"] == "n_neighbors")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("KNN Regressor - wpływ n_neighbors")
    plt.xlabel("n_neighbors")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# KNN - metric
# =========================
def plot_knn_metric(df):
    data = df[
        (df["model"] == "KNNRegressor") &
        (df["parameter_name"] == "metric")
    ].copy()

    x_labels = data["parameter_value"].astype(str)
    train = data["train_rmse"]
    test = data["test_rmse"]

    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, train, width=width, label="Train RMSE")
    plt.bar(x + width/2, test, width=width, label="Test RMSE")

    plt.title("KNN Regressor - wpływ metric")
    plt.xlabel("metric")
    plt.ylabel("RMSE")
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()

# =========================
# KNN - p
# =========================
def plot_knn_p(df):
    data = df[
        (df["model"] == "KNNRegressor") &
        (df["parameter_name"] == "p")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("KNN Regressor - wpływ p")
    plt.xlabel("p")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# Decision Tree - max_depth
# =========================
def plot_tree_max_depth(df):
    data = df[
        (df["model"] == "DecisionTreeRegressor") &
        (df["parameter_name"] == "max_depth")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("Decision Tree Regressor - wpływ max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# Decision Tree - min_samples_split
# =========================
def plot_tree_min_samples_split(df):
    data = df[
        (df["model"] == "DecisionTreeRegressor") &
        (df["parameter_name"] == "min_samples_split")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("Decision Tree Regressor - wpływ min_samples_split")
    plt.xlabel("min_samples_split")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# Decision Tree - min_samples_leaf
# =========================
def plot_tree_min_samples_leaf(df):
    data = df[
        (df["model"] == "DecisionTreeRegressor") &
        (df["parameter_name"] == "min_samples_leaf")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("Decision Tree Regressor - wpływ min_samples_leaf")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# Random Forest - n_estimators
# =========================
def plot_forest_n_estimators(df):
    data = df[
        (df["model"] == "RandomForestRegressor") &
        (df["parameter_name"] == "n_estimators")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("Random Forest Regressor - wpływ n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# Random Forest - max_depth
# =========================
def plot_forest_max_depth(df):
    data = df[
        (df["model"] == "RandomForestRegressor") &
        (df["parameter_name"] == "max_depth")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(str)

    depth_order = ["5", "10", "15", "None"]
    data["parameter_value"] = pd.Categorical(
        data["parameter_value"],
        categories=depth_order,
        ordered=True
    )
    data = data.sort_values("parameter_value")

    x_labels = data["parameter_value"].astype(str)
    train = data["train_rmse"]
    test = data["test_rmse"]

    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, train, width=width, label="Train RMSE")
    plt.bar(x + width / 2, test, width=width, label="Test RMSE")

    plt.title("Random Forest Regressor - wpływ max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("RMSE")
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()


# =========================
# Random Forest - min_samples_leaf
# =========================
def plot_forest_min_samples_leaf(df):
    data = df[
        (df["model"] == "RandomForestRegressor") &
        (df["parameter_name"] == "min_samples_leaf")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(int)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("Random Forest Regressor - wpływ min_samples_leaf")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("RMSE")
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# SVR - C
# =========================
def plot_svr_C(df):
    data = df[
        (df["model"] == "SVR") &
        (df["parameter_name"] == "C")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(float)
    data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_rmse"]
    test = data["test_rmse"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train RMSE")
    plt.plot(x, test, marker="o", label="Test RMSE")

    plt.title("SVR - wpływ parametru C")
    plt.xlabel("C")
    plt.ylabel("RMSE")
    plt.xscale("log")
    plt.xticks(x, labels=[str(v) for v in x])
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# SVR - kernel
# =========================
def plot_svr_kernel(df):
    data = df[
        (df["model"] == "SVR") &
        (df["parameter_name"] == "kernel")
    ].copy()

    x_labels = data["parameter_value"].astype(str)
    train = data["train_rmse"]
    test = data["test_rmse"]

    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, train, width=width, label="Train RMSE")
    plt.bar(x + width / 2, test, width=width, label="Test RMSE")

    plt.title("SVR - wpływ kernel")
    plt.xlabel("kernel")
    plt.ylabel("RMSE")
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()


# =========================
# SVR - gamma
# =========================
def plot_svr_gamma(df):
    data = df[
        (df["model"] == "SVR") &
        (df["parameter_name"] == "gamma")
    ].copy()

    data["parameter_value"] = data["parameter_value"].astype(str)

    gamma_order = ["scale", "auto", "0.01", "0.1"]
    data["parameter_value"] = pd.Categorical(
        data["parameter_value"],
        categories=gamma_order,
        ordered=True
    )
    data = data.sort_values("parameter_value")

    x_labels = data["parameter_value"].astype(str)
    train = data["train_rmse"]
    test = data["test_rmse"]

    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, train, width=width, label="Train RMSE")
    plt.bar(x + width / 2, test, width=width, label="Test RMSE")

    plt.title("SVR - wpływ gamma")
    plt.xlabel("gamma")
    plt.ylabel("RMSE")
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = load_data()

    plot_knn_n_neighbors(df)
    plot_knn_metric(df)
    plot_knn_p(df)

    plot_tree_max_depth(df)
    plot_tree_min_samples_split(df)
    plot_tree_min_samples_leaf(df)

    plot_forest_n_estimators(df)
    plot_forest_max_depth(df)
    plot_forest_min_samples_leaf(df)

    plot_svr_C(df)
    plot_svr_kernel(df)
    plot_svr_gamma(df)
