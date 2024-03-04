import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("C:\kuliah\Semester 6\Submission\Data\day.csv")

# Display the dataframe using Streamlit
st.write("Dataframe from CSV:")
st.write(df)
# Missing value
st.write('Missing value:')
st.write(df.isna().sum())
# Cleaning data
df_cleaned = df.dropna()
st.write('DataFrame Cleaned:')
st.write(df_cleaned)

cleaned_file_path = 'cleaned_data.csv'
st.write('Cleaned data has been saved to:', cleaned_file_path)
df_cleaned.to_csv(cleaned_file_path, index=False)

#1. Calculate the total weekday for each month in the DataFrame
total_weekday = df.groupby('mnth')['weekday'].sum().reset_index()
st.write('Total bike sharing on weekdays based on month:')
st.write(total_weekday)
#Visualization Pie Chart
fig, ax = plt.subplots()
ax.pie(total_weekday['weekday'], labels=total_weekday['mnth'], autopct='%1.1f%%')
ax.set_title('Total bike sharing on weekdays based on month')
st.pyplot(fig)

#2. This code essentially visualizes the total bike sharing values on working days based on each month using a bar plot.
fig, ax = plt.subplots()
ax.bar(df['mnth'], df['workingday'])
ax.set_title('Total bike sharing values ​​on working days based on Month')
ax.set_xlabel('Month')
ax.set_ylabel('Total bike sharing on workingday')
st.pyplot(fig)

#3. This code essentially visualizes the Total bike sharing values ​​for holiday, weekday, and workingday based on Season using area chart
total_season = df.groupby('season')[['holiday', 'weekday', 'workingday']].sum().reset_index()
st.write('Total bike sharing values ​for holiday, weekday, and workingday based on Season:')
st.write(total_season)
#Visualization Area Chart
fig, ax = plt.subplots()
ax.stackplot(total_season['season'], total_season['holiday'], total_season['weekday'], total_season['workingday'],
    labels=['holiday Score', 'weekday Score', 'workingday Score'], alpha=0.5)
ax.set_title('Total bike sharing values ​​for holiday, weekday, and workingday based on Season')
ax.set_xlabel('Season')
ax.set_ylabel('Total bike sharing Season')
ax.legend(loc='upper left')
st.pyplot(fig)
#Visualization clustering
kmeans = KMeans(n_clusters=3)
features = df_cleaned[['weekday', 'workingday', 'holiday']]
kmeans.fit(features)
df_cleaned['cluster'] = kmeans.labels_

st.write('Clustering Results:')
st.write(df_cleaned)

fig, ax = plt.subplots()
for cluster_label, color in zip(range(3), ['red', 'blue', 'green']):
            cluster_points = df_cleaned[df_cleaned['cluster'] == cluster_label]
            ax.scatter(cluster_points['weekday'], cluster_points['workingday'], color=color, label=f'Cluster {cluster_label}')
ax.set_xlabel('Weekday')
ax.set_ylabel('Workingday')
ax.set_title('Bike Sharing Clustering')
ax.legend()
st.pyplot(fig)

