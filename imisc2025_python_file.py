# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18, 2025

@author: Kaan, Elif :)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    v_measure_score,
    silhouette_score
)
# import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.cm as cm


# Veri Okuma

df = pd.read_excel("imisc.xlsx")

"""
le = LabelEncoder()


df["Economic Category"] = df["Economic Category"].map({
    "Emerging and Developing Economies": 0,
    "Advanced": 1
})

df["Economic Category Name"] = le.fit_transform(df["Economic Category Name"])

countries = df["Country"].copy()
df_num = df.drop(["Country", "Ranking"], axis=1)

# Korelasyon
my_cors = df_num.corr().round(2)
# Korelasyon Isı Haritası
sns.heatmap(
 my_cors,
 
 annot = True,
 square=True,
 cmap=sns.color_palette("flare", as_cmap=True))

#Veri setinin normalizasyonu
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_num)
df_normalized = pd.DataFrame(df_scaled, columns=df_num.columns)
"""



# Kullanılacak değişkenler
cols = ["Government", "Technology Sector", "Data and Infrastructure"]

# Alt veri seti + sonsuz ve eksik değer temizliği
X = df[cols].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=0, how="any")


# Ölçekleme (k-ortalamalar ve silhouette için iyi pratik)
# Xs = StandardScaler().fit_transform(X)
# Xs = MinMaxScaler().fit_transform(X)

# 4) k=2..8 için k-ortalamalar ve silhouette
scores = []
for k in range(2, 9):
    if len(X) <= k:
        # Silhouette için her kümeye en az 1 nokta düşmeli; çok az gözlem varsa atla
        continue
    km = KMeans(n_clusters=k, random_state=1903)
    labels = km.fit_predict(X)
    s = silhouette_score(X, labels, metric="euclidean")
    scores.append({"k": k, "silhouette": s})

result = pd.DataFrame(scores)
print(result)

# En iyi k
best_k = result.loc[result["silhouette"].idxmax(), "k"]
best_s = result["silhouette"].max()
print(f"\nEn iyi k: {best_k} (Silhouette = {best_s:.4f})")

plt.figure()
plt.plot(result["k"], result["silhouette"], marker="o")
plt.xticks(result["k"])
plt.xlabel("k")
plt.ylabel("Silhouette skoru")
plt.title("K-ortalamalar için Ortalama Silhouette Endeks Değerleri")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()



# Küme sayısının belirlenmesi ve K-Ortalamalar Algoritmasının uygulanması (Küme sayısı 2-8 arasında değer olmalıdır)
cluster_num = int(input("Lütfen Küme Sayısını Belirleyiniz "))
kMeans_model = KMeans(n_clusters=cluster_num, random_state=1903)
kMeans_model.fit_predict(X)
kMeans_model.labels_

#Kümelerin verisetine ayrı bir sütun olarak eklenmesi
df["clusters"] = kMeans_model.labels_
df["clusters"] = df["clusters"].astype("category")
df.clusters.value_counts()

df["Economic Category"].value_counts()




# Kümeleme sonucunda elde edilen kümelerin adlandırılması
# k=2 için, Advanced ekonomik kategorisinde olanlar 1, Emerging olanlar 0 olarak kümelenmiş
# Aferin Kaancığım:) Çok doğru bir tespit!

clstr_centers = kMeans_model.cluster_centers_
# K-Ortalamalar Algoritamsından elde edilen kkümelerin görselleştirilmesi
colors = cm.tab10.colors
for i in range(cluster_num):
    plt.scatter(
        X.values[kMeans_model.labels_ == i, 0],
        X.values[kMeans_model.labels_ == i, 1],
        s=50, c=[colors[i]], label=f"Cluster {i}"
    )
plt.scatter(
    kMeans_model.cluster_centers_[:, 0],
    kMeans_model.cluster_centers_[:, 1],
    s=100, c="black", label="Centers"
)
plt.legend()
plt.show()

# Kümeleme Performansının Ölçülmesi
kMeans_model.inertia_
print("WCSS: %.3f" % kMeans_model.inertia_)

# Silhoutte Index
silh_val = silhouette_score(X, kMeans_model.labels_, metric="euclidean")
print("Silhouette Skor Değeri: %.3f" % silh_val)

graph = SilhouetteVisualizer(kMeans_model, colors="yellowbrick")
graph.fit(X)

# En uygun k küme sayısının belirlenmesi için Dirsek Yöntemi
# WCSS ve Silhouette Skorlarının karşılaştırılması

wcss = []
k = range(2,9)
for i in k:
 kMeans_m = KMeans(n_clusters = i, random_state = 0)
 kMeans_m.fit(X)
 wcss.append(kMeans_m.inertia_)
plt.plot(k, wcss, 'bx-')
plt.xticks(k)
plt.title("En İyi Küme Sayısı (WCSS)")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("WCSS")
plt.show()

silhouette = []
k = range(2,9)
for i in k:
 kMeans_m = KMeans(n_clusters = i, random_state = 0)
 kMeans_m.fit(X)
 silhouette.append(silhouette_score(X, kMeans_m.labels_, metric = "euclidean"))
plt.plot(k, silhouette, 'bx-')
plt.xticks(k)
plt.title("En İyi Küme Sayısı (Silhouette Skoru)")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Silhouette Skoru")
plt.show()








# Gerçek ülke ekonomi kategorileri ile k-Ortalamalardan elde edilen küme etiketlerini kıyaslama
# Tıpkı bir sınıflandırma algoritmasından elde edilen sonuçların değerlendirilmesi gibi

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df["Economic Category Num"] = (
    df["Economic Category"].astype(str).str.strip()
    .map({"Advanced": 1, "Emerging and Developing Economies": 0})
    .astype("Int64")
)


# Tahmin edilen ikili etiketler (1=Advanced, 0=EDE)
y_pred = df.clusters
y_true = df["Economic Category Num"]


# 6) Confusion matrix ve classification report
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # sırayla: EDE(0), Advanced(1)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0 (EDE)", "Actual 1 (Advanced)"],
    columns=["Pred 0 (EDE)", "Pred 1 (Advanced)"]
)

print("Confusion Matrix:\n", cm_df, "\n")

"""
print("Accuracy:", f"{accuracy_score(y_true, y_pred):.4f}\n")
print("Classification Report:\n",
      classification_report(y_true, y_pred,
                            target_names=["EDE (0)", "Advanced (1)"],
                            digits=3))



# Uyum Analizleri
# 1) Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y_true, y_pred)

# 2) Normalized Mutual Information (NMI) - 'average_method' varsayılan 'arithmetic'
nmi = normalized_mutual_info_score(y_true, y_pred)

# 3) Homogeneity, Completeness, V-Measure (üçü bir arada)
hom, comp, v = homogeneity_completeness_v_measure(y_true, y_pred)

# (İsterseniz V-Measure'ı ayrıca da alabilirsiniz)
v2 = v_measure_score(y_true, y_pred)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Homogeneity: {hom:.4f}")
print(f"Completeness: {comp:.4f}")
print(f"V-Measure: {v:.4f}  (ayrıca v_measure_score ile: {v2:.4f})")

# 4) (Opsiyonel) Silhouette Score
# Not: Silhouette, GERÇEK etiketleri değil, küme etiketlerini (yp) ve özellik matrisini (X) ister.
# Ayrıca en az 2 farklı küme bulunmalı.
mask = y_true.notna() & y_pred.notna()
try:
    if 'X' in globals() and len(np.unique(y_pred)) >= 2:
        sil = silhouette_score(X[mask], y_pred)
        print(f"Silhouette Score: {sil:.4f}")
    else:
        print("Silhouette hesaplanmadı: X tanımlı değil veya tek küme var.")
except Exception as e:
    print(f"Silhouette hesaplanırken hata: {e}")


"""


# PCA ile 3-boyutlu verinin 2 temel bileşene göre Scatterplot üzerinde gösterimi
# Scatterplot hem ülkelerin gerçek ekonomi kategorilerini (Renk) hem de küme etiketlerini (Şekil) gösteriyor

from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---- Ayarlar ----
feat_cols = ["Government", "Technology Sector", "Data and Infrastructure"]
true_col = "Economic Category Num"   # 1=Advanced, 0=EDE (önceki adımda siz oluşturdunuz)
pred_col = "clusters"                # k=2 sonucu: 1=Advanced, 0=EDE olacak şekilde

# Alt veri ve temizlik
need_cols = feat_cols + [true_col, pred_col]
Z = df[need_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

# Tür güvenliği
Z[true_col] = Z[true_col].astype(int)
Z[pred_col] = Z[pred_col].astype(int)

# Ölçekleme + PCA(2D)
# Xs = StandardScaler().fit_transform(Z[feat_cols].to_numpy())
pc = PCA(n_components=2, random_state=1903).fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": pc[:, 0],
    "PC2": pc[:, 1],
    "y_true": Z[true_col].to_numpy(),   # 0=EDE, 1=Advanced
    "y_pred": Z[pred_col].to_numpy(),   # 0 veya 1
}, index=Z.index)

# Renk ve işaret eşlemesi
color_map  = {0: "blue", 1: "red"}   # 0=EDE mavi, 1=Advanced kırmızı
marker_map = {0: "o",    1: "^"}     # 0=daire, 1=üçgen

# Çizim
fig, ax = plt.subplots(figsize=(7, 5))
for pv, marker in marker_map.items():
    sub = plot_df[plot_df["y_pred"] == pv]
    ax.scatter(sub["PC1"], sub["PC2"],
               c=sub["y_true"].map(color_map),
               marker=marker,
               alpha=0.9,
               edgecolor="white",
               linewidths=0.5,
               label=f"Pred {pv}")

# Lejandlar (işaret ve renk ayrı)
# Pred
shape_legend = [
    Line2D([0],[0], marker='o', linestyle='', label='Cluster 0 (EDE)', markerfacecolor='gray', markeredgecolor='black', markersize=9),
    Line2D([0],[0], marker='^', linestyle='', label='Cluster 1 (Advanced)', markerfacecolor='gray', markeredgecolor='black', markersize=9),
]
color_legend = [
    Patch(facecolor='blue', label='EDE'),
    Patch(facecolor='red',  label='Advanced'),
]
first_legend = ax.legend(handles=shape_legend, title="Cluster Labels", loc="upper right")
ax.add_artist(first_legend)
ax.legend(handles=color_legend, title="Economy Categories", loc="lower right")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Economy Categories vs. Cluster Labels of Countries")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()






# Hangi ülkeler hatalı olarak sıralandı?
# Advanced olup EDE kümesindekiler vs. EDE olup Advanced kümesindekiler


valid = y_true.isin([0, 1]) & y_pred.isin([0, 1])

adv_to_ede = (
    df.loc[valid & (y_true == 1) & (y_pred == 0), "Country"]
      .dropna().astype(str).drop_duplicates().sort_values()
)
ede_to_adv = (
    df.loc[valid & (y_true == 0) & (y_pred == 1), "Country"]
      .dropna().astype(str).drop_duplicates().sort_values()
)

print("Advanced olup EDE kümelenenler:")
for c in adv_to_ede:
    print(c)

print("\nEDE olup Advanced kümelenenler:")
for c in ede_to_adv:
    print(c)










# Ülkelerin olması gerektikleri küme merkezine göre küçükten büyüğe sıralamalaru


# Kümelere (y_pred) göre merkezler
centroid = {}
for c in [0, 1]:
    sel = (y_pred == c)
    if sel.sum() == 0:
        raise ValueError(f"Küme {c} boş; merkez hesaplanamadı.")
    centroid[c] = X[sel].mean(axis=0)




bad = (y_true != y_pred)
if not bad.any():
    print("Hatalı kümelenen ülke yok.")
else:
    X_bad = X[bad]
    ytrue_b = y_true[bad]
    ypred_b = y_pred[bad]
    countries_b = df.loc[bad, "Country"].astype(str).to_numpy()

    # Her gözlem için y_true merkezine uzaklık
    target_centers = np.vstack([centroid[int(t)] for t in ytrue_b])
    d = np.linalg.norm(X_bad - target_centers, axis=1)

    # 5) İki yön için indeksler ve küçük→büyük sıralama
    adv_to_ede_idx = np.where((ytrue_b == 1) & (ypred_b == 0))[0]
    ede_to_adv_idx = np.where((ytrue_b == 0) & (ypred_b == 1))[0]

    adv_order = adv_to_ede_idx[np.argsort(d[adv_to_ede_idx])]   # küçük→büyük
    ede_order = ede_to_adv_idx[np.argsort(d[ede_to_adv_idx])]   # küçük→büyük

    print("Advanced olup EDE kümelenenler (küçük→büyük):")
    for rank, i in enumerate(adv_order, start=1):
        print(f"{rank}) {countries_b[i]}")

    print("\nEDE olup Advanced kümelenenler (küçük→büyük):")
    for rank, i in enumerate(ede_order, start=1):
        print(f"{rank}) {countries_b[i]}")





import geopandas as gpd
import matplotlib.pyplot as plt

# 1) https://geojson-maps.kyd.au/  bu siteden dünya haritasını geoJSON formatında indirdim.
#Dosya yolunu kendi bilgisayarınıza göre ayarlarsınız. :)

world = gpd.read_file("custom.geo.json")


#Verisetimizde yazan ülke isimleriyle json dosyasında yazan ülke isimleri uyuşmuyor.
#Mapping yöntemiyle bu sorunu düzelttik.

country_mapping = {
    "Czech Republic": "Czechia",
    "Guinea Bissau": "Guinea-Bissau",
    "Central African Republic": "Central African Rep.",
    "Republic of Korea": "South Korea",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Democratic Republic of the Congo": "Dem. Rep. Congo",
    "Saint Vincent and the Grenadines": "St. Vin. and Gren.",
    "Gambia (Republic of The)": "Gambia",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Eswatini": "eSwatini",
    "Viet Nam": "Vietnam",
    "Republic of Moldova": "Moldova",
    "Syrian Arab Republic": "Syria",
    "Equatorial Guinea": "Eq. Guinea",
    "Russian Federation": "Russia",
    "Solomon Islands": "Solomon Is.",
    "Brunei Darussalam": "Brunei",
    "State of Palestine": "Palestine",
    "Antigua and Barbuda": "Antigua and Barb.",
    "Marshall Islands": "Marshall Is.",
    "Iran (Islamic Republic of)": "Iran",
    "Sao Tome and Principe": "São Tomé and Principe",
    "United Republic of Tanzania": "Tanzania",
    "Dominican Republic": "Dominican Rep.",
    "Bosnia and Herzegovina": "Bosnia and Herz.",
    "South Sudan": "S. Sudan",
    "Lao People's Democratic Republic": "Laos"
}


# Countries adında yeni bir sütun ekledik verisetimize.
#Halihazırda olan Country ile de değiştirilebilir ama ben yapmadım çünkü neden olmasın :)
df["Countries"] = df["Country"].replace(country_mapping)



# JSON dosyasındaki ülkeleri merge ettik.
merged = world.merge(df, how="left", left_on="name", right_on="Countries")

# Harita çizimi
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
merged.plot(column="clusters", cmap="tab10", linewidth=0.8, ax=ax, edgecolor="0.8", legend=True, missing_kwds={"color": "lightgrey"})
plt.title("Clusters obtained with k-Means Algorithm", fontsize=14)
plt.show()




#Bu sefer gerçek kategorilere göre dünya haritası resmettik.
world_merged = world.merge(df[["Countries", "Economic Category"]], left_on="name", right_on="Countries", how="left")
"""
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
merged.plot(column="Economic Category", cmap="tab10", linewidth=0.8, ax=ax, edgecolor="0.8", legend=True, missing_kwds={"color": "lightgrey"})
plt.title("Economy Categories of Countries", fontsize=14)
plt.show()
"""

fig, ax = plt.subplots(1, 1, figsize=(15, 8))
merged.plot(
    column="Economic Category",
    cmap="tab10_r",  # tab10'un ters çevrilmiş versiyonu
    linewidth=0.8,
    ax=ax,
    edgecolor="0.8",
    legend=True,
    missing_kwds={"color": "lightgrey"}
)
plt.title("Economy Categories of Countries", fontsize=14)
plt.show()


# Economic Category sütunundaki 1 ↔ 0 değişimi

"""
# Harita çizimi
plt.figure(figsize=(16,10))
world_merged.plot(column="Economic Category", 
                  legend=False, 
                  cmap="coolwarm", 
                  missing_kwds={"color": "lightgrey"},
                  edgecolor="black",
                  linewidth=0.8)

plt.title("Economy Categories of Countries", fontsize=8)
plt.show()

df["Economic Category"].value_counts()


"""




