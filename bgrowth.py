# growth_curves.py
#
# Script that generates sample growth curves using the Zwietering growth
# curve logistic model.
#
# Author: Daniel A. Cuevas (dcuevas08.at.gmail.com)
# Date:   January 22, 2018


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Zwietering logistic growth model
def logistic(time, y0, lag, mu, A):
    denom = (1 + np.exp(((4 * mu / A) * (lag - time)) + 2))
    return y0 + ((A - y0) / denom)


N = 10000  # Number of curves
t = 50  # Number of hours
np.random.seed(9)  # Set random number seed

# Generate growth curve parameters for all growth curves
# Starting OD = [0.05, 0.10]
y0 = np.random.uniform(0.05, 0.10, N)

# Biomass yield = [0.1, 1.2]
A = np.random.uniform(0.1, 1.2, N)


# Maximum growth rate = [A/t, 1.1A]
mu = [np.random.uniform(a / t, a * 1.1, 1)[0] for a in A]

# Lag time = [0.0, 20.0]
lag = np.random.uniform(0, 20, N)

# Generate sample IDs for each growth curve
sample = ["sample{}".format(x) for x in range(N)]

# Generate the time vector of 30 minute intervals
time = np.arange(start=0, stop=t, step=0.5)

# Create a pandas DataFrame containing all parameter data
pheno = pd.DataFrame({"sample": sample, "y0": y0, "A": A, "mu": mu,
                      "lag": lag})

#logistic definition of growth curve 
def logistic(time, y0, lag, mu, A):
	logcurve = []
	logcurve = y0 + ((A - y0) / (1 + np.exp(((4*mu / A) * (lag-time)) + 2)))
	return logcurve 
	
# Use parameters to generate growth curves
# Curves will be stored in a long format pandas DataFrame
curve_data = {"sample": [], "time": [], "od": []}
for idx, x in pheno.iterrows():
    sample = x["sample"]
    y0 = x["y0"]
    lag = x["lag"]
    mu = x["mu"]
    A = x["A"]
    log_curve = logistic(time, y0, lag, mu, A)
    for i, t in enumerate(time):
        curve_data["sample"].append(sample)
        curve_data["time"].append(t)
        curve_data["od"].append(log_curve[i])
curves = pd.DataFrame(curve_data)
#calculate GS
pheno["growthscore"] = (pheno["A"] - pheno["y0"]) + pheno["mu"] * 0.25
#calculate GL
def GL(logcurve, y0, A):
	return len(logcurve)/np.sum((1 / ((A-y0)+(logcurve-y0))))

growthlevels = {"sample" :[], "growthlevel": []}
for name, group in curves.groupby("sample"):
	logcurve = np.array(group["od"].values)
	y0 = pheno.loc[pheno["sample"] == name, "y0"].values[0]
	A = pheno.loc[pheno["sample"] == name, "A"].values[0]
	growthlevels["sample"].append(name)
	growthlevels["growthlevel"].append(GL(logcurve, y0, A))
tmp_df = pd.DataFrame(growthlevels)
tmp_df.set_index("sample", inplace=True)
data = pheno.join(tmp_df, on="sample")

data["GL class"] = "--"
data.loc[data["growthlevel"] > 0.25, "GL class"] = "-"
data.loc[data["growthlevel"] > 0.35, "GL class"] = "+"
data.loc[data["growthlevel"] > 0.50, "GL class"] = "++"
data.loc[data["growthlevel"] > 0.75, "GL class"] = "+++"
data.loc[data["growthlevel"] > 1.00, "GL class"] = "++++"
data["GS class"] = "--"
data.loc[data["growthscore"] > 0.15, "GS class"] = "-"
data.loc[data["growthscore"] > 0.25, "GS class"] = "+"
data.loc[data["growthscore"] > 0.35, "GS class"] = "++"
data.loc[data["growthscore"] > 0.50, "GS class"] = "+++"
data.loc[data["growthscore"] > 0.65, "GS class"] = "++++"

sns.set(style='white', context='talk')
g = sns.jointplot("growthlevel", "growthscore", data=data, kind="reg", height=8)
g.ax_joint.collections[0].set_visible(False)
colors = sns.color_palette()
for idx, row in data.iterrows():
	if row["GL class"] == "--":
		g.ax_joint.plot(row["growthlevel"], row["growthscore"], color=colors[3], marker="o", markersize=5)
	elif row["GL class"] == "-":
		g.ax_joint.plot(row["growthlevel"], row["growthscore"], color=colors[5], marker="o", markersize=5)
	elif row["GL class"] == "+":
		g.ax_joint.plot(row["growthlevel"], row["growthscore"], color=colors[4], marker="o", markersize=5)
	elif row["GL class"] == "++":
		g.ax_joint.plot(row["growthlevel"], row["growthscore"], color=colors[2], marker="o", markersize=5)
	elif row["GL class"] == "+++":
		g.ax_joint.plot(row["growthlevel"], row["growthscore"], color=colors[1], marker="o", markersize=5)
	else:
		g.ax_joint.plot(row["growthlevel"], row["growthscore"], color=colors[0], marker="o", markersize=5)
# Draw lines
xlim = plt.xlim(-0.1, None)
ylim = plt.ylim(-0.1, None)
g.ax_joint.plot(xlim, [0.15, 0.15], "k--", linewidth=1)
g.ax_joint.plot(xlim, [0.25, 0.25], "k--", linewidth=1)
g.ax_joint.plot(xlim, [0.35, 0.35], "k--", linewidth=1)
g.ax_joint.plot(xlim, [0.50, 0.50], "k--", linewidth=1)
g.ax_joint.plot(xlim, [0.65, 0.65], "k--", linewidth=1)
txt = plt.xlabel("Growth Level")
txt = plt.ylabel("Growth Score")

g.savefig("figure1.pdf", dpi=400, bbox_inches="tight")

#I honestly have no idea what's going on here, but I don't think I truly need it.
'''curves_plot = curves.join(data[["sample", "GL class", "GS class"]].set_index("sample"), on="sample")

curves_plot["GL class"] = curves_plot["GL class"].astype("category")
curves_plot["GL class"].cat.reorder_categories(["++++", "+++", "++", "+", "-", "--"], inplace=True)
curves_plot["GS class"] = curves_plot["GS class"].astype("category")
curves_plot["GS class"].cat.reorder_categories(["++++", "+++", "++", "+", "-", "--"], inplace=True)

ax = sns.lineplot(data=curves_plot, time="time", unit="sample", condition="GL class", value="od", err_style="boot_traces", n_boot=100)
sns.despine()

ax.figure.savefig("figure2A.pdf", dpi=400, bbox_inches="tight")
 
ax = sns.lineplot(data=curves_plot, time="time", unit="sample", condition="GL class", value="od", err_style="boot_traces", n_boot=100)
sns.despine()

ax.figure.savefig("figure2B.pdf", dpi=400, bbox_inches="tight")'''
