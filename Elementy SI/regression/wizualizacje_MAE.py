import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

data = {
    "Neurons": pd.DataFrame({
        "Val": [16, 32, 64, 128],
        "Train": [1.0692, 0.9578, 0.8146, 0.5548],
        "Test": [0.7919, 0.8344, 0.7181, 0.9436]
    }),
    "Activation": pd.DataFrame({
        "Val": ["tanh", "relu", "sigmoid", "elu"],
        "Train": [0.9733, 0.9486, 2.7274, 0.9315],
        "Test": [0.8721, 0.9989, 1.8361, 0.5862]
    }),
    "Layers": pd.DataFrame({
        "Val": [1, 2, 3, 4],
        "Train": [0.9688, 0.9438, 0.9682, 0.9707],
        "Test": [0.6899, 0.5490, 0.6981, 0.9346]
    }),
    "Window": pd.DataFrame({
        "Val": [6, 12, 18, 24],
        "Train": [0.9679, 0.9811, 0.9038, 0.9444],
        "Test": [0.6920, 0.7605, 0.5705, 0.7719]
    }),
    "LR": pd.DataFrame({
        "Val": [0.001, 0.005, 0.01, 0.05],
        "Train": [1.1291, 0.9640, 0.8949, 1.8340],
        "Test": [0.5009, 0.8054, 0.5501, 5.2722]
    }),
    "TrainingSize": pd.DataFrame({
        "Val": [400, 800, 1200, 1600],
        "Train": [1.4886, 1.1751, 0.9759, 0.8247],
        "Test": [1.1162, 1.0687, 0.7804, 0.5647]
    }),
    "Epochs": pd.DataFrame({
        "Val": [25, 50, 75, 100],
        "Train": [0.9918, 0.9664, 0.9298, 0.8985],
        "Test": [0.5980, 0.6663, 0.5230, 0.6144]
    })
}

titles = {
    "Neurons": "3.1 Liczba neuronów",
    "Activation": "3.2 Funkcja aktywacji",
    "Layers": "3.3 Liczba warstw",
    "Window": "3.4 Wielkość okna",
    "LR": "3.5 Learning Rate",
    "TrainingSize": "3.6 Rozmiar zbioru",
    "Epochs": "3.7 Liczba epok" # Dodany brakujący tytuł
}

for key, df in data.items():
    plt.figure(figsize=(10, 6))
    if key == "Activation":
        df_melt = df.melt(id_vars="Val", var_name="Zbiór", value_name="MAE")
        sns.barplot(data=df_melt, x="Val", y="MAE", hue="Zbiór", palette="muted")
        plt.xlabel("Funkcja aktywacji")
    else:
        plt.plot(df["Val"], df["Train"], marker='o', label='Uczący', linewidth=2)
        plt.plot(df["Val"], df["Test"], marker='s', label='Testowy', linewidth=2, linestyle='--')
        plt.xlabel(titles[key].split(" ")[-1])

        if key == "LR":
            plt.xscale('log')
            plt.xticks(df["Val"], df["Val"])

    plt.title(titles[key], fontweight='bold')
    plt.ylabel("Błąd MAE")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()