import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="ticks", palette="muted")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})


df = pd.read_csv("Final_Output_Batch.csv")
df["equivalent_diameter_nm"] = 2 * np.sqrt(df["area_bio_nm2"] / np.pi)


fig, axes = plt.subplots(1, 3, figsize=(16, 5))


sns.histplot(
    df["equivalent_diameter_nm"],
    kde=True,
    ax=axes[0],
    color="#2b5c8f",
    bins=15,
)
median_val = df["equivalent_diameter_nm"].median()
axes[0].axvline(
    median_val,
    color="#d9534f",
    linestyle="--",
    linewidth=2,
    label=f"Median: {median_val:.1f} nm",
)
axes[0].set_title("A) Object Size Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Equivalent Circular Diameter (nm)")
axes[0].set_ylabel("Count")
axes[0].legend()


sns.scatterplot(
    data=df,
    x="area_bio_nm2",
    y="integrated_density",
    ax=axes[1],
    color="#2e7d32",
    s=50,
    alpha=0.8,
)
sns.regplot(
    data=df,
    x="area_bio_nm2",
    y="integrated_density",
    ax=axes[1],
    scatter=False,
    color="#1b5e20",
)
axes[1].set_title(
    "B) Biological Area vs. Total Intensity", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Biological Area (nm²)")
axes[1].set_ylabel("Integrated Density (a.u.)")


sns.boxplot(
    y=df["mean_intensity"],
    ax=axes[2],
    color="#f57c00",
    width=0.3,
    boxprops=dict(alpha=0.7),
)
sns.stripplot(
    y=df["mean_intensity"],
    ax=axes[2],
    color="black",
    alpha=0.5,
    jitter=0.2,
    size=5,
)
axes[2].set_title("C) Signal Concentration", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Mean Intensity (a.u.)")

plt.tight_layout()


plt.savefig("ExM_Analysis_Plots.png", dpi=300)
print("Graphs were successfully saved as 'ExM_Analysis_Plots.png'")
plt.show()