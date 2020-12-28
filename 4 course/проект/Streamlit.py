import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
# from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
# from plotly import tools
# init_notebook_mode(connected = True)
# import plotly.figure_factory as ff
import streamlit as st
import warnings
import random

from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('ggg1.csv', delimiter=';')
data1 = pd.read_csv('table51.csv', delimiter=';')

todo_selectbox = st.sidebar.selectbox("",("Данные и теория", "Кластеризация"))

if todo_selectbox == "Данные и теория":
    visualize_selectbox = st.sidebar.selectbox(
        "", ("Основная информация", "Распределение по зарплате и сумме кредита", "Сумма кредита и количество детей (младше 18)", 
             "Распределение по гендеру", "Распределение по возрасту", "График кореляции"))
    if visualize_selectbox =="Основная информация":
        st.write("Данные проекта")
        st.write(data)
        st.markdown('где:')
        st.markdown('CreditSum_ - сумма кредита, взятая клиентом в банке')
        st.markdown('age - возраст клиента')
        st.markdown('kolichestvo_detej_mladshe_18 - количество детей клиента (младше 18)')
        st.markdown('ConfirmedMonthlyIncome (Target) - зарплата клиента банка')
        st.markdown('')
        st.markdown('Кластеризация')
        st.markdown('«Разделяет обьекты по неизвестному признаку. Машина сама решает как лучше»')
        st.markdown('Сегодня используют для:')
        st.markdown('*Сегментация рынка (типов покупателей, лояльности)')
        st.markdown('*Объединение близких точек на карте')
        st.markdown('*Сжатие изображений')
        st.markdown('*Анализ и разметки новых данных')
        st.markdown('*Детекторы аномального поведения') 
        st.markdown('Кластеризация - это классификация, но без заранее известных классов. Она сама ищет похожие объекты и объединяет их в кластеры. Количество кластеров можно задать заранее или доверить это машине. Похожесть объектов машина определяет по тем признакам, которые ей разметили, объекты со схожими характеристиками определяются в один класс.')
        st.markdown('Кластеризация (cluster analysis) — задача группировки множества объектов на подмножества (кластеры) таким образом, чтобы объекты из одного кластера были более похожи друг на друга, чем на объекты из других кластеров по какому-либо критерию.')
        st.markdown('Задача кластеризации относится к классу задач обучения без учителя.')
        st.markdown('Цели кластеризации могут быть различными в зависимости от особенностей конкретной прикладной задачи: ')
        st.markdown('*Упростить дальнейшую обработку данных, разбить множество X^n на группы схожих объектов чтобы работать с каждой группой в отдельности (задачи классификации, регрессии, прогнозирования).')
        st.markdown('*Сократить объём хранимых данных, оставив по одному представителю от каждого кластера (задачи сжатия данных).')
        st.markdown('*Выделить нетипичные объекты, которые не подходят ни к одному из кластеров (задачи одноклассовой классификации).')
        st.markdown('*Построить иерархию множества объектов (задачи таксономии).') 
        st.markdown('Каждый метод кластеризации имеет свои ограничения и выделяет кластеры лишь некоторых типов. ')
        st.markdown('Понятие «тип кластерной структуры» зависит от метода и также не имеет формального определения.')
        
    elif visualize_selectbox == "Распределение по зарплате и сумме кредита":
        warnings.filterwarnings('ignore')
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.subplot(2, 3, 1)
        sns.set(style='darkgrid')
        sns.distplot(data['ConfirmedMonthlyIncome (Target)'])
        plt.title('Distribution of Salary', fontsize=20)
        plt.xlabel('Range of Salary')
        plt.ylabel('Count')
        plt.subplot(2, 3, 2)
        sns.set(style='whitegrid')
        sns.distplot(data['CreditSum'], color='red')
        plt.title('Distribution of CreditSum', fontsize=20)
        plt.xlabel('Range of CreditSum')
        plt.ylabel('Count')
        st.pyplot(plt)
             
    elif visualize_selectbox=="Распределение по гендеру":
        labels = ['Женский', 'Мужской']
        size = data1['sex'].value_counts()
        colors = ['red', 'blue']
        explode = [0, 0.1]
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
        plt.title('Гендер', fontsize=10)
        plt.axis('off')
        st.pyplot(plt, use_container_width=True)
        
    elif visualize_selectbox =="Распределение по возрасту":
        plt.rcParams['figure.figsize'] = (15, 8)
        sns.countplot(data['age'], palette='coolwarm')
        plt.title('Распределение по возрасту', fontsize=20)
        st.pyplot(plt)
        
    elif visualize_selectbox=="График кореляции":
        plt.rcParams['figure.figsize'] = (15, 8)
        sns.heatmap(data.corr(), cmap='Oranges', annot=True)
        plt.title('График колеляции', fontsize=20)
        st.pyplot(plt)
        
    elif visualize_selectbox=="Сумма кредита и количество детей (младше 18)":
        plt.rcParams['figure.figsize'] = (18, 7)
        sns.boxenplot(data['CreditSum'], data['kolichestvo_detej_mladshe_18'], palette='Oranges')
        plt.title('Сумма кредита и количество детей (младше 18)', fontsize=20)
        st.pyplot(plt)
        
elif todo_selectbox == "Кластеризация":
    methods_selectbox = st.sidebar.selectbox(
        "",
        ("Кластеризовать методом K-means (сумма кредита от зарплаты)","Кластеризовать методом K-means (сумма кредита и количество детей)",
         "K-means 3D"))
    if methods_selectbox=="Кластеризовать методом K-means (сумма кредита от зарплаты)":
        x_income = data.iloc[:, [0, 4]].values
        km = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_means = km.fit_predict(x_income)
        plt.scatter(x_income[y_means == 0, 0], x_income[y_means == 0, 1], s=100, c='red')
        plt.scatter(x_income[y_means == 1, 0], x_income[y_means == 1, 1], s=100, c='yellow')
        plt.scatter(x_income[y_means == 2, 0], x_income[y_means == 2, 1], s=100, c='green')
        plt.scatter(x_income[y_means == 3, 0], x_income[y_means == 3, 1], s=100, c='pink')
        plt.scatter(x_income[y_means == 4, 0], x_income[y_means == 4, 1], s=100, c='orange')
        plt.scatter(x_income[y_means == 5, 0], x_income[y_means == 5, 1], s=100, c='lightblue')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='darkblue')
        plt.title('метод K-means', fontsize=20)
        plt.xlabel('Сумма кредита')
        plt.ylabel('Зарплата')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    elif methods_selectbox=="Кластеризовать методом K-means (сумма кредита и количество детей)":
        x_income = data.iloc[:, [0, 3]].values
        km = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_means = km.fit_predict(x_income)
        plt.scatter(x_income[y_means == 0, 0], x_income[y_means == 0, 1], s=100, c='red')
        plt.scatter(x_income[y_means == 1, 0], x_income[y_means == 1, 1], s=100, c='yellow')
        plt.scatter(x_income[y_means == 2, 0], x_income[y_means == 2, 1], s=100, c='green')
        plt.scatter(x_income[y_means == 3, 0], x_income[y_means == 3, 1], s=100, c='pink')
        plt.scatter(x_income[y_means == 4, 0], x_income[y_means == 4, 1], s=100, c='orange')
        plt.scatter(x_income[y_means == 5, 0], x_income[y_means == 5, 1], s=100, c='lightblue')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='darkblue')
        plt.title('метод K-means', fontsize=20)
        plt.xlabel('Сумма кредита')
        plt.ylabel('Количество детей (младше 18)')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        
    elif methods_selectbox=="K-means 3D":
        x = data[['CreditSum', 'age', 'ConfirmedMonthlyIncome (Target)']].values
        km = KMeans(n_clusters=6, init='k-means++', max_iter=500, n_init=10, random_state=0)
        km.fit(x)
        labels = km.labels_
        centroids = km.cluster_centers_
        data['labels'] = labels
        trace1 = go.Scatter3d(x=data['CreditSum'],y=data['age'],z=data['ConfirmedMonthlyIncome (Target)'],mode='markers',
            marker=dict(color=data['labels'],size=10,line=dict(color=data['labels'],width=12),opacity=0.8))
        df = [trace1]
        layout = go.Layout(title='',margin=dict(l=0,r=0,b=0,t=0),
            scene=dict(xaxis=dict(title='CreditSum'),yaxis=dict(title='age'),zaxis=dict(title='ConfirmedMonthlyIncome (Target)')))
        fig = go.Figure(data=df, layout=layout)
        st.plotly_chart(fig)
        st.markdown('где:')
        st.markdown('x - Сумма кредита')
        st.markdown('y - возраст клиента')
        st.markdown('z - количество детей лиента (младше 18)')