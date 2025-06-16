**Разработка полностью связанной нейронной сети**

**Цель:** получить базовые навыки работы с одной из библиотек глубокого
обучения (Caffe, Torch, TensorFlow или MXNet на выбор) на примере
полностью связанных нейронных сетей.

**Задачи:** Выполнение практической работы предполагает решение следующих задач:
1. выбор библиотеки для выполнения практических работ курса;
2. установка выбранной библиотеки на кластере;
3. проверка корректности установки библиотеки;
4. разработка и запуск тестового примера сети, соответствующей логистической регрессии, для решения задачи классификации рукописных цифр набора данных MNIST;
5. выбор практической задачи компьютерного зрения для выполнения практических работ;
6. разработка программ/скриптов для подготовки тренировочных и тестовых данных в формате, который обрабатывается выбранной библиотекой;
7. разработка нескольких архитектур полностью связанных нейронных сетей (варьируются количество слоев и виды функций активации на каждом слое) в формате, который принимается выбранной библиотекой;
8. обучение разработанных глубоких моделей;
8. тестирование обученных глубоких моделей;
9. сделать вывод относительно разработанных архитектур;
10. подготовка отчета, содержащего минимальный объем информации по каждому этапу выполнения работы.


**Часть 1. Загрузка необходимых библиотек**

```import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.offline as pyo
import gc

# Библиотека TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
import tensorflow.keras.preprocessing.image as tf_image
