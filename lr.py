import numpy as np
import pandas as pd
from plot_db import visualize_3d
import sys

def main():

    '''
    Implementation of Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Writes to an output csv file the outcome betas for each (alpha, # of iterations) setting.
    Please run the file as follows: python lr.py data2.csv, results2.csv

    '''

    _, input, output = sys.argv

    data = pd.read_csv(input, header = None)
    n = len(data)
    first_line = np.ones(n)
    data.insert(0,0,first_line, True)
    data.columns = ['x0','x1','x2','y']

    #standartization & normalization
    feat1, feat2 = np.array(data['x1']), np.array(data['x2'])
    feat1,feat2 = feat1 - feat1.mean(axis=0), feat2 - feat2.mean(axis=0)
    feat1, feat2 = feat1 / feat1.std(axis=0), feat2 / feat2.std(axis=0)
    data['x1'], data['x2'] = feat1, feat2

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    MAX_ITER = 60

    # helper function
    def f(x, betas):
        '''return the dot product of the betas and the example (beta_0 + beta_1*x1 + beta_2*x2)'''
        x = np.array(x)
        return betas.dot(x)

    results = []

    def grad_desc(alpha, data, iterations=MAX_ITER) -> list:

        betas = np.zeros(3)

        for i in range(iterations):

            total_errors = np.zeros(3)

            for row in data.iterrows():
                xi, yi = np.array(row[1]['x0':'x2']), row[1]['y']
                total_errors += (f(xi, betas) - yi)*xi

            for j in range(3):
                betas[j] -= alpha*(1/n)*total_errors[j]

        return betas

    for alpha in alphas:
        betas = grad_desc(alpha, data, MAX_ITER)
        results.append([alpha, MAX_ITER, betas[0], betas[1], betas[2]])

    my_alpha = 0.75
    my_iter = 40

    my_betas = grad_desc(my_alpha, data, my_iter)
    results.append([my_alpha, my_iter, my_betas[0], my_betas[1], my_betas[2]])


    with open(output,'w') as out:
        for line in results:
            out.write(str(line))
            out.write('\n')

    visualize_3d(data[['x1','x2','y']], lin_reg_weights = my_betas, feat1='x1', feat2='x2', labels = 'y', alpha = my_alpha)

if __name__ == "__main__":
    main()