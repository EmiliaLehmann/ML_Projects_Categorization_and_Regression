import os
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_classification_data


def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
    }

    return results


def save_results(results_list, output_path="results/classification_results2.csv"):
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


def run_classification_experiments():
    data = load_classification_data()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    results = []

    print("Start eksperymentów klasyfikacyjnych")

#KNN
    knn_neighbors = [3, 5, 7, 9]
    for n in knn_neighbors:
        print(f"KNN | n_neighbors={n}")
        model = KNeighborsClassifier(
            n_neighbors=n,
            metric="minkowski",
            p=2
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "n_neighbors", n, metrics)

    knn_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    for metric_name in knn_metrics:
        print(f"KNN | metric={metric_name}")
        model = KNeighborsClassifier(
            n_neighbors=5,
            metric=metric_name,
            p=2
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "metric", metric_name, metrics)

    knn_p_values = [1, 2, 3, 4]
    for p_value in knn_p_values:
        print(f"KNN | p={p_value}")
        model = KNeighborsClassifier(
            n_neighbors=5,
            metric="minkowski",
            p=p_value
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "p", p_value, metrics)

#Decision Tree

    tree_depths = [3, 5, 10, 15]
    for depth in tree_depths:
        print(f"Decision Tree | max_depth={depth}")
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=2,
            criterion="gini",
            random_state=42
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "DecisionTree", "max_depth", depth, metrics)

    tree_min_samples_split = [2, 5, 10, 20]
    for min_split in tree_min_samples_split:
        print(f"Decision Tree | min_samples_split={min_split}")
        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=min_split,
            criterion="gini",
            random_state=42
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "DecisionTree", "min_samples_split", min_split, metrics)

    tree_min_samples_leaf = [1, 2, 4, 8]
    for leaf in tree_min_samples_leaf:
        print(f"Decision Tree | min_samples_leaf={leaf}")
        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=leaf,
            random_state=42
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "DecisionTree", "min_samples_leaf", leaf, metrics)

#Random Forest

    forest_estimators = [50, 100, 200, 300]
    for n_estimators in forest_estimators:
        print(f"Random Forest | n_estimators={n_estimators}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "RandomForest", "n_estimators", n_estimators, metrics)

    forest_depths = [5, 10, 15, None]
    for depth in forest_depths:
        print(f"Random Forest | max_depth={depth}")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "RandomForest", "max_depth", depth, metrics)

    forest_min_leaf = [1, 2, 4, 8]
    for min_leaf in forest_min_leaf:
        print(f"Random Forest | min_samples_leaf={min_leaf}")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "RandomForest", "min_samples_leaf", min_leaf, metrics)

#SVM

    svm_c_values = [0.1, 1, 10, 100]
    for c_value in svm_c_values:
        print(f"SVM | C={c_value}")
        model = SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
            random_state=42
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "C", c_value, metrics)

    svm_kernels = ["linear", "poly", "rbf", "sigmoid"]
    for kernel_name in svm_kernels:
        print(f"SVM | kernel={kernel_name}")
        model = SVC(
            C=1.0,
            kernel=kernel_name,
            gamma="scale",
            random_state=42
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "kernel", kernel_name, metrics)

    svm_gamma_values = ["scale", "auto", 0.01, 0.1]
    for gamma_value in svm_gamma_values:
        print(f"SVM | gamma={gamma_value}")
        model = SVC(
            C=1.0,
            kernel="rbf",
            gamma=gamma_value,
            random_state=42
        )
        metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "gamma", gamma_value, metrics)

#
    #zapis wyników
    save_results(results)

    df_results = pd.DataFrame(results)
    best_result = df_results.loc[df_results["test_accuracy"].idxmax()]

    print("\nNajlepszy wynik klasyfikacji:")
    print(best_result)


if __name__ == "__main__":
    run_classification_experiments()