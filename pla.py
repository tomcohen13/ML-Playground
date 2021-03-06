import pandas as pd
import numpy as np
from plot_db import visualize_scatter
import sys


def main():
    '''The file should run as python pla.py data1.csv results1.csv or, generally, python pla.py [data_file] [results_file]'''

    results = []
    _, data, output = sys.argv

    try:
        df = pd.read_csv(data, header = None)

    except:
        raise Exception("Invalid csv file")

    w, b = np.zeros(2), 0

    def f(x):
        #computer y-hat using b + x·w
        return 1 if b + x.dot(w) > 0 else -1

    #loop until convergence
    while True:

        errors = 0
        for row in df.iterrows():
            arr = [item[1] for item in row[1].items()]
            xi, yi = np.array(arr[:2]), arr[2]

            if yi*f(xi)<=0:
                w += (yi - f(xi))*xi
                b += yi
                errors += 1

        results.append([w[0],w[1],b])

        #convergence
        if errors == 0:
            break

    file = open(output, 'w+')
    for line in results:
        file.write(str(line))
        file.write("\n")

    visualize_scatter(df, weights=np.concatenate((w, [b])))


if __name__ == "__main__":
    main()