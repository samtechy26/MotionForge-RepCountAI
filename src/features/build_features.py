import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

#--------------------------------
# Load Data
#--------------------------------

df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenets.pkl')

predictor_columns = df.columns[:6]

# Plot Settings

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Dealing with missing values
for col in predictor_columns:
    df[col] = df[col].interpolate()

# Claculating the duration
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()

#-----------------------------------------------
# Buttlerworth lowpass filter
# Applying the lowpass filter
#------------------------------------------------

df_lowpass = df.copy()

Lowpass = LowPassFilter()

fs = 5 # this is calculated by (1/(200/1000)) where 200 is our frequency in milliseconds

cutoff = 1.3

for col in predictor_columns:
    df_lowpass = Lowpass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

#----------------------------------------------    
# Principle Component Analysis (PCA)
#----------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal Component Number")
plt.ylabel("explained variance")
plt.show()

# Using 3 as the optimal value based on the elbow rule
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


#-----------------------------------------------
# Sum of Squares attributes
#------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)



#------------------------------------------------
# Temporal Abstraction
#------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = list(predictor_columns)
predictor_columns.extend(["acc_r", "gyr_r"])
predictor_columns = pd.Index(predictor_columns)

ws = int(1000 / 200)

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)


#-------------------------------------------------
# Freuency Features
# Discrete Fourier Transformation
#-------------------------------------------------

df_freq = df_temporal.copy().reset_index()
freqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2000 / 200)

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = freqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# Dealing with overlapping windows

df_freq = df_freq.dropna()
df_freq.iloc[::2]

#-------------------------------------------
# Clustering
# Using KMeans
#-------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distance")
plt.show()

# Using 5 as the optimal k value

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]

df_cluster["cluster"] = kmeans.fit_predict(subset)

#plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

plt.legend()
plt.show()

# Export the data

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
