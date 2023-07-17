
def addlabels(pl, x, y):
    for i in range(len(x)):
        pl.text(i, y[i], y[i], ha='center')

