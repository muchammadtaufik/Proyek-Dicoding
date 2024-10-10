import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Judul untuk aplikasi Streamlit
st.title('Analisis Data Order Item dengan K-Means Clustering')

with st.sidebar:
    st.image("D:/Order items/gudang.jpg")

# Membaca dataset dan menampilkannya
file_path = "D:/Order items/olist_order_items_dataset.csv"

# Check jika file tersedia
try:
    df = pd.read_csv(file_path)
    st.write("Dataset berhasil dimuat:")
    st.write(df.head())  # Menampilkan 5 data teratas
except FileNotFoundError:
    st.error(f"Tidak dapat menemukan file di {file_path}")
    st.stop()

# Memilih fitur 'price' dan 'freight_value'
x_train = df[['price', 'freight_value']]

# Menampilkan korelasi fitur
st.write("Korelasi antara 'price' dan 'freight_value':")
st.write(x_train.corr())

# Melakukan feature scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Membuat clustering K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
y_cluster = kmeans.fit_predict(x_train_scaled)

# Visualisasi data hasil clustering dengan centroid
fig, ax = plt.subplots(figsize=(10, 5))

# Scatter plot untuk data yang dikelompokkan
scatter = ax.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=kmeans.labels_, cmap='viridis', s=50)

# Scatter plot untuk centroids dengan warna yang berbeda
ax.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='red', marker='*', s=200, label='Centroid 0')
ax.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='blue', marker='*', s=200, label='Centroid 1')

# Menambahkan judul dan label sumbu
ax.set_title('K-Means Clustering Visualization')
ax.set_xlabel('Price (scaled)')
ax.set_ylabel('Freight Value (scaled)')

# Menampilkan legenda
ax.legend()

# Menampilkan visualisasi di Streamlit
st.pyplot(fig)

# Menambahkan penjelasan clustering
st.write("""
Visualisasi di atas menunjukkan hasil clustering K-Means dari dataset berdasarkan dua kolom: **Price** dan **Freight Value**.
Titik merah mewakili **Centroid 0**, sedangkan titik biru mewakili **Centroid 1**.
""")
