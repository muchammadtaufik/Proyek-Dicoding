import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="dark")

def create_weather_weekday_df(df, isWeekdaybyInt):
    weather_weekday_df = df[df["weekday"] == isWeekdaybyInt].groupby(["weathersit"])["cnt"].sum().sort_values(ascending=False).reset_index()
    if not (weather_weekday_df["weathersit"] == 4).any():
        new_row = pd.DataFrame({"weathersit": [4], "cnt": [0]})
        weather_weekday_df = pd.concat([weather_weekday_df, new_row], ignore_index=True)

    weather_weekday_df.rename(columns={"weathersit": "index_cuaca", "cnt": "jumlah_pengguna"}, inplace=True)
    return weather_weekday_df

def create_byHourGroup_df(df):
    df["hr_group"] = df.hr.apply(lambda x: "Pagi Hari" if x >= 4 and x < 11
                                 else ("Siang Hari" if x >= 11 and x < 16
                                       else ("Sore Hari" if x >= 16 and x < 21
                                             else "Malam Hari")))

    byHourGroup_df = df.groupby(by="hr_group")["cnt"].sum().reset_index()
    byHourGroup_df.rename(columns={"hr_group": "categories", "cnt": "jumlah_pengguna"}, inplace=True)

    return byHourGroup_df

# Load datasets
day_df = pd.read_csv(r"C:\KULIAH\SEMESTER 5\MATERI\Dicoding\Bangkit\Materi\Analisis Data dengan Python\DATA PROJEK\bike sharing\Data\day.csv")
hour_df = pd.read_csv(r"C:\KULIAH\SEMESTER 5\MATERI\Dicoding\Bangkit\Materi\Analisis Data dengan Python\DATA PROJEK\bike sharing\Data\hour.csv")


# Sort values and convert date column
column = "dteday"
day_df[column] = pd.to_datetime(day_df[column])
hour_df[column] = pd.to_datetime(hour_df[column])

day_df.sort_values(by=column, inplace=True)
hour_df.sort_values(by=column, inplace=True)

min_date = day_df[column].min()
max_date = day_df[column].max()

with st.sidebar:
    # Replace with relative path or online URL if necessary
    st.image("C:/KULIAH/SEMESTER 5/MATERI/Dicoding/Bangkit/Materi/Analisis Data dengan Python/DATA PROJEK/bike sharing/Tampilan Streamlit/Sepeda.jpg")

    # Date input for selecting time range
    start_date, end_date = st.date_input(label="Time", min_value=min_date, max_value=max_date, value=[min_date, max_date])

# Convert start_date and end_date to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter dataframes by selected date range
main_df = day_df[(day_df[column] >= start_date) & (day_df[column] <= end_date)]
second_df = hour_df[(hour_df[column] >= start_date) & (hour_df[column] <= end_date)]

# Create dataframes for visualizations
weather_weekday_df = create_weather_weekday_df(main_df, 0)
byHourGroup_df = create_byHourGroup_df(second_df)

# Dashboard content
st.header("Bike Sharing Dashboard :sparkles:")

# Weekday pie charts
st.subheader("Weekday Weather Impact")

col1, col2 = st.columns(2)

labels_detail = ['Cerah', 'Mendung', 'Hujan Ringan', 'Hujan Lebat']

# Create pie chart for weekday
fig1, ax1 = plt.subplots()
size = weather_weekday_df["jumlah_pengguna"]
pie1 = ax1.pie(size, startangle=0)
ax1.set_title("Jumlah Pengguna Sepeda Pada Hari Biasa (Weekday)\n di Tiap Kondisi Cuaca", ha="center")
ax1.legend(pie1[0], labels_detail, bbox_to_anchor=(0.65, -0.05), loc="lower right", bbox_transform=plt.gcf().transFigure)
col1.pyplot(fig1)

# Barplot for time of day analysis
st.subheader("Time and Bike Bounce Correlation")

fig3 = plt.figure(figsize=(10, 5))
sns.barplot(y="jumlah_pengguna", x="categories", data=byHourGroup_df.sort_values(by="jumlah_pengguna", ascending=False), dodge=False)

plt.title("Jumlah Total Pengguna di Tiap Kelompok Jam", loc="center", fontsize=17)
plt.ylabel("Jumlah Pengguna")
plt.xlabel(None)
plt.tick_params(axis="x", labelsize=12)
plt.legend(title="Kategori Waktu", labels=byHourGroup_df["categories"].unique())
st.pyplot(fig3)
