# Description: This file contains utility functions for the project

import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

pop_a = mpatches.Patch(color='#BB6B5A', label='High')
pop_b = mpatches.Patch(color='#E5E88B', label='Medium')
pop_c = mpatches.Patch(color='#8CCB9B', label='Low')


def colormap(risk_list):
    cols = []
    for l in risk_list:
        if l == 0:
            cols.append('#BB6B5A')
        elif l == 2:
            cols.append('#E5E88B')
        elif l == 1:
            cols.append('#8CCB9B')
    return cols


def two_d_compare(y_test, y_pred, model_name, x_test):
    label_encoder = LabelEncoder()
    y_pred = label_encoder.fit_transform(y_pred)
    y_test = label_encoder.fit_transform(y_test)
    area = (12 * np.random.rand(40))**2
    plt.subplots(ncols=2, figsize=(10, 4))
    plt.suptitle('Actual vs Predicted data : ' + model_name +
                 '. Accuracy : %.2f' % accuracy_score(y_test, y_pred))

    plt.subplot(121)
    plt.scatter(x_test['ESTINCOME'], x_test['DAYSSINCELASTTRADE'],
                alpha=0.8, c=colormap(y_test), s=area)
    plt.title('Actual')
    plt.legend(handles=[pop_a, pop_b, pop_c])

    plt.subplot(122)
    plt.scatter(x_test['ESTINCOME'], x_test['DAYSSINCELASTTRADE'],
                alpha=0.8, c=colormap(y_pred), s=area)
    plt.title('Predicted')
    plt.legend(handles=[pop_a, pop_b, pop_c])

    plt.show()


def three_d_compare(y_test, y_pred, model_name, x_test):
    x = x_test['TOTALDOLLARVALUETRADED']

    y = x_test['ESTINCOME']
    z = x_test['DAYSSINCELASTTRADE']

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Actual vs Predicted (3D) data : ' + model_name +
                '. Accuracy : %.2f' % accuracy_score(y_test, y_pred))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x, y, z, c=colormap(y_test), marker='o')
    ax.set_xlabel('TOTAL DOLLAR VALUE TRADED')
    ax.set_ylabel('ESTIMATED INCOME')
    ax.set_zlabel('DAYS SINCE LAST TRADE')
    plt.legend(handles=[pop_a, pop_b, pop_c])
    plt.title('Actual')

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(x, y, z, c=colormap(y_pred), marker='o')
    ax.set_xlabel('TOTAL DOLLAR VALUE TRADED')
    ax.set_ylabel('ESTIMATED INCOME')
    ax.set_zlabel('DAYS SINCE LAST TRADE')
    plt.legend(handles=[pop_a, pop_b, pop_c])
    plt.title('Predicted')

    plt.show()


def model_metrics(y_test, y_pred):
    print("Decoded values of Churnrisk after applying inverse of label encoder : " +
          str(np.unique(y_pred)))

    skplt.metrics.plot_confusion_matrix(
        y_test, y_pred, text_fontsize="small", cmap='Greens', figsize=(6, 4))
    plt.show()

    print("The classification report for the model : \n\n" +
          classification_report(y_test, y_pred))
