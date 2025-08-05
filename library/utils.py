###############################################################################
# This library provide some general utilities
###############################################################################
import numpy as np
import pandas as pd
import math
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

##########################################################################
# Input functions
##########################################################################
def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    filename += ".csv"
    dataframe = pd.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.array(dataset[:, :]).astype(np.float32)
    return HEADER, DATA


def read_XY(filename, nY=1):
    # Format data for training
    # Function read_training_data
    _, XY = read_csv(filename).astype(np.float32)
    XY = np.array(XY, dtype= np.float32)
    nX = XY.shape[1] - nY
    X = XY[:, :nX]
    Y = XY[:, nX:]
    return X, Y


def find_medium(mediumname, x_name):
    path = "Dataset_input/" + mediumname + ".csv"
    data = pd.read_csv(path, encoding="utf-8")
    data = pd.DataFrame(data)
    med_name = data.columns.str.contains(x_name)
    mediumsize = np.count_nonzero(data.columns.str.contains(x_name))
    columns = []
    name = data.columns[med_name]
    for col in name:
        bol = data[col].unique().size != 1
        columns.append(bol)
    fixedmediumsize = np.size(columns) - np.count_nonzero(columns)
    Xsize = data.shape[1] - 1
    return fixedmediumsize, mediumsize, Xsize

def random_pick(name = "Dataset_input/biolog_iML1515_EXP.csv"):
    from pandas import read_csv
    H = read_csv(name)
    frac = H.sample(frac=0.10, random_state=42)
    frac.to_csv('Dataset_input/biolog_iML1515_EXP_0.1.csv', index=False)
    return

#########################################################################
# Custome metric function
#########################################################################
def get_r2(y_true, y_pred):  # maybe can remove
    return r2_score(y_true, y_pred, multioutput="variance_weighted")


def get_rmse(y_true, y_pred):
    a = int(0)
    for i in range(len(y_true)):
        a = a + (y_pred[i] - y_true[i])**2
    r = math.sqrt(a)/len(y_true)*100
    return r


def transform(y, threshold):
    # transform y from growth-rate into detection signal (0 = not growth, 1 = growth)
    yp = np.copy(y)
    if isinstance(y,np.ndarray) != True:
        yp = np.array(yp)
    for i in range(yp.shape[0]):
        if yp[i] >= threshold:
            yp[i] = 1
        else:
            yp[i] = 0
    return yp


def get_accuracy(y_true, y_pred, threshold):
    # copy y to prevent changing the original y,
    # all the binary transformation only happen in this function

    yt, yp = np.copy(y_true), np.copy(y_pred)
    yt, yp = transform(yt, threshold), transform(yp, threshold)
    acc = accuracy_score(yt, yp)
    return acc


def measure_performance(y_t, y, regresion = True, threshold = 0.1):
    """
    Measure the performance of predicted values compared to true values.

    Parameters:
    - y_true (numpy.ndarray): True values.
    - y_predicted (numpy.ndarray): Predicted values.
    - regression (bool): If True, treat the problem as a regression task; if False, treat it as a classification task.
    - threshold (float): Threshold for classification 

    Returns:
    - measure (float): The performance measure, which is either the R-squared score (regression=True)
                      or the accuracy score (regression=False).
    """
    if regresion:
            measure = get_r2(y_t,y)
    else:
            measure = get_accuracy(y_t,y, threshold)

    return measure
#########################################################################
# Plotting function
#########################################################################
def plot_r2_curve(y_true, y_pred, zoom = False):
    r2 = get_r2(y_true, y_pred)
    TRUE, PRED = y_true.flatten(), y_pred.flatten()

    if zoom == True:
        limit = max(TRUE)*1.2 # allow 20% out zoom
        boolean_mask = [item > -0.2 and item < limit for item in PRED]
        PRED = [value for value, mask in zip(PRED, boolean_mask) if mask]
        TRUE = [value for value, mask in zip(TRUE, boolean_mask) if mask]

    sns.set(
        font="arial",
        palette="colorblind",
        style="whitegrid",
        font_scale=1.5,
        rc={"figure.figsize": (5, 5), "axes.grid": False},
    )
    sns.regplot(
        x=TRUE,
        y=PRED,
        fit_reg=0,
        marker="+",
        color="black",
        scatter_kws={"s": 40, "linewidths": 0.7},
    )
    plt.plot([min(TRUE), max(TRUE)], [min(TRUE), max(TRUE)], 
             linestyle='--', 
             color='blue',
             linewidth=1)
    plt.xlabel("Measured growth rate (." + r"$\mathregular{hr^{-1}}$" + ")")
    plt.ylabel("Cobra growth rate (." + r"$\mathregular{hr^{-1}}$" + ")")
    plt.title(f'R2: {r2:.2f}', fontsize=14)
    plt.xlim(min(TRUE) - 0.01, max(TRUE) + 0.05)
    plt.ylim(min(PRED) - 0.02, max(PRED) + 0.05)
    plt.show()


def plot_accuracy(y_true, y_pred, label):
    Y = {"label": label[0].tolist(), "y_pred": y_pred, "y_true": y_true}
    df = pd.DataFrame(Y)

    # calculate accuracy by group
    a = df[df["label"] == "N"]
    b = df[df["label"] == "C"]
    accuracy = {
        "All": accuracy_score(df["y_true"], df["y_pred"]),
        "Nitrogen": accuracy_score(a["y_true"], a["y_pred"]),
        "Carbon": accuracy_score(b["y_true"], b["y_pred"]),
    }
    courses = list(accuracy.keys())
    values = list(accuracy.values())

    # fig size
    plt.figure(figsize=(8, 5))

    # creating the bar plot
    plt.bar(courses, values, color="orange", width=0.4)
    plt.ylabel("Accuracy %")
    plt.title("Accuracy by group")
    plt.show()


def plot_confusion(y,y_cobra,threshold):
    yt, yc = y.copy(), y_cobra.copy()
    yt, yc = transform(yt,threshold), transform(yc,threshold)
    cm = confusion_matrix(yt, yc, normalize= 'all')*100
    acc = round(accuracy_score(yt, yc)*100,2)

    #plot
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Cobra predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'Confusion Matrix : accuracy {acc} %'); 
    ax.xaxis.set_ticklabels(['not-growth', 'growth']); ax.yaxis.set_ticklabels(['not-growth', 'growth'])
    plt.show()


def plot_interation(name, fba_result = None,  metric = 'R2', color='b'):
    #input data
    df = pd.read_excel(name)

    # calculate mean and std of each interation
    result = df.groupby('interation')['tc'].agg(['mean', 'std']).reset_index()

    #add result before AL if there is any
    if fba_result != None:
        noAL = {'interation':0, 'mean':fba_result, 'std':0} # add original result before active learning
        result = pd.concat([pd.DataFrame(noAL, index=[0]), result], ignore_index=True)

    #plot
    x = result['interation']
    y = result['mean']
    std = result['std']
    plt.plot(x, y, label = 'mean of '+metric, color=color)
    plt.fill_between(x, 
                     y - std, y + std, 
                     color=color, 
                     label = 'std of '+metric,
                     alpha=0.2)
    plt.xlabel('Loops of active learning')
    plt.ylabel(metric)
    plt.legend(loc='lower right')
    plt.grid(False)
    # print mean value at each interation
#    for i, j in zip(x, y):
#        plt.text(i, j, f'{j:.2f}', ha='center', va='bottom', fontsize=10)
    plt.show()