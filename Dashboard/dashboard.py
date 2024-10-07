# Import library yang diperlukan
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv

# Membuat judul aplikasi
st.title("Analisis data order item menggunakan K-Means Clustering")

#Membaca dataset dan menampilkannya
df = read_csv("D:\Order items\olist_order_items_dataset.csv")

#Membuat data training
x_train = df[['price', 'freight_value']]

# Menampilkan data dalam bentuk tabel
st.subheader("Nilai Korelasi")
st.write(pd.DataFrame({
    " " : ["Price", "Freight_Value"],
    "Price" : [1.000000, 0.414204],
    "Freight Value" : [0.414204	, 1.000000]
}))

# Membuat visualisasi menggunakan Matplotlib
st.header("Visualisasi Clustering")
st.image("D:\Order items\Clustering order item.png")