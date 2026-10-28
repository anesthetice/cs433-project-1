from implementations import Array2D
from implementations import Array1D
import numpy as np
import matplotlib.pyplot as plt
import extra as ex

def plot_training_and_stats(
    train_losses: list[float],
    valid_losses: list[float],
    y_valid: Array1D,
    x_valid: Array2D,
    w: Array1D,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    ax: plt.Axes = ax1
    ax.plot(
        np.arange(len(train_losses)),
        train_losses,
        ls="--",
        color="teal",
        label="training loss",
    )
    ax.plot(
        np.arange(len(valid_losses)),
        valid_losses,
        ls="--",
        color="indianred",
        label="validation loss",
    )
    ax.set_xlabel("iteration number")
    ax.set_ylabel("loss")
    ax.legend()


    sensitivities = []
    specificities = []
    f1s = []

    thresholds = np.linspace(0, 1, 201)
    for tr in thresholds:
        stats = ex.compute_statistics(
            y_true=y_valid, y_pred=ex.logreg_predict(x_valid, w, tr)
        )
        sensitivities.append(stats["sensitivity"])
        specificities.append(stats["specificity"])
        f1s.append(stats["F1"])

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1s = np.array(f1s)

    tr_best_f1 = thresholds[np.argmax(f1s)]
    stats_best_f1 = ex.compute_statistics(
        y_true=y_valid, y_pred=ex.logreg_predict(x_valid, w, tr_best_f1)
    )

    #print(f"Threshold of {tr_best_f1:.4f} (best F1 score)")
    print(f"├─┄ TP = {stats_best_f1['TP']}, TN = {stats_best_f1['TN']}")
    print(f"├─┄ FP = {stats_best_f1['FP']}, FN = {stats_best_f1['FN']}")
    print(f"├─┄ sensitivity = {stats_best_f1['sensitivity']:.2f}")
    print(f"├─┄ specificity = {stats_best_f1['specificity']:.2f}")
    print(f"├─┄ precision = {stats_best_f1['precision']:.2f}")
    print(f"└─┄ F1-score = {stats_best_f1['F1']:.2f}\n")

    ax: plt.Axes = ax2
    ax.plot(thresholds, sensitivities, label="sensitivity")
    ax.plot(thresholds, specificities, label="specificity")
    ax.plot(thresholds, f1s, label="F1-score")
    ax.vlines(
        x=tr_best_f1,
        ymin=0,
        ymax=1,
        ls="--",
        color="gray",
        label="F1 threshold",
    )
    ax.set_xlabel("threshold")
    ax.legend()

    fig.tight_layout()

    return fig
