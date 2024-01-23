import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
import streamlit as st
import pickle

@st.cache_data
def load_data():
  df = pd.read_csv('new_dataset.csv')
  df.drop(['Person ID', 'Sick'], axis=1, inplace=True)
  numerical_features = df.select_dtypes(include=np.number).columns.tolist()
  numerical_features.remove('Stress Level')
  return df

df = load_data()

def show_explore_page():
  st.title("Sleep Health Life Style Data Exploration")
  st.write(" ### Sleep Disorder and Stress Level Prediction")
  
  fig1, ax1 = plt.subplots(figsize = (12, 7))
  data = df['Gender'].value_counts(normalize=True).sort_values(ascending=False)
  st.write(" ##### Gender Percentage in the Data ")
  ax1.pie(data, explode = [0.05, 0.02],labels = ['Male', 'Female'],colors = ['#64FE2E', '#E2A9F3'],                                           autopct = '%1.2f%%',
                                                                                   shadow = True)
  
  ax1.axis("Equal")
  st.pyplot(fig1)

  print()

  fig1, ax1 = plt.subplots(figsize = (12, 7))
  data = sns.histplot(data = df, x='Sleep Duration', bins = 30, kde = True, hue = 'Gender')
  plt.xlabel("Sleep Duration")
  plt.ylabel("Count")
  st.write(" ##### Sleep Duration as per Gender Wise")
  st.pyplot(fig1)
  
  print()

  fig1, ax1 = plt.subplots(figsize = (12, 7))
  data = df.groupby('Occupation')['Quality of Sleep'].mean().sort_values(ascending = False)
  ax1.bar(data.index, height = data.values, color = ['#00FFBF', '#819FF7', '#E2A9F3', '#64FE2E'])
  plt.xlabel("Occupation")
  plt.ylabel("Average Sleep Time")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  st.write(" ##### Average Sleep Time of Different Occupation")
  st.pyplot(fig1)

  print()

  fig1, ax1 = plt.subplots(figsize = (12, 7))
  data = df['Sleep Disorder'].value_counts(normalize=True).sort_values(ascending=False)
  ax1.pie(data, explode = [0.05, 0.06, 0.07],labels = ['None', 'Sleep Apnea', 'Insomnia'],
          colors = ['#00FFBF', '#819FF7', '#E2A9F3'], autopct = '%1.2f%%', shadow = True)
  ax1.axis("Equal")
  st.write(" ##### Sleep Disorder Percentage in the Data ")
  st.pyplot(fig1)

  print()
 
  
  fig1, ax1 = plt.subplots(figsize = (12, 7))
  data = df.groupby('BMI Category')['Stress Level'].mean().sort_values(ascending = False)
  ax1.bar(data.index, height = data.values,color = ['#64FE2E', '#819FF7', '#E2A9F3'])

  plt.xlabel("BMI Category")
  plt.ylabel("Average Stress Level")
  plt.tight_layout()
  st.write(" ##### Average Stress Level for the BMI Category")
  st.pyplot(fig1)

  print()

  fig = px.area(df, x='Quality of Sleep', y='Stress Level', color='Gender')
  fig.update_layout(
    xaxis_title='Quality of Sleep',
    yaxis_title='Stress Level',
    font = dict(
        size=12
    )
)
  st.write(" ##### Quality of Sleep vs Stress Level (Interactive Plot)")
  st.plotly_chart(fig, use_container_width=True)

  print()

  fig = px.area(df, x='Sleep Duration', y='Stress Level', color='Gender')
  fig.update_layout(
    xaxis_title='Sleep Duration',
    yaxis_title='Stress Level',
    font = dict(
        size=12
    )
)
  st.write(" ##### Sleep Duration vs Stress Level (Interactive Plot)")
  st.plotly_chart(fig, use_container_width=True)


  
