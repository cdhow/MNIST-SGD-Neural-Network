import numpy as np
import load_data as ld
import sys
import neural_network as nn


# This function runs the provided mnst data
def run_mnst(n_input, n_hidden, n_output, ftrain_x, ftrain_y, ftest_x, ftest_y):
    tr_x, tr_y, t_x, t_y = ld.load_data(ftrain_x, ftrain_y, ftest_x, ftest_y)

    network = nn.NeuralNetwork(sizes=[784, 30, 10]) # TODO change to params
    network.SGD(tr_x, tr_y, t_x, t_y)


# This function runs the network on the assignment sheet
def run_simple_network(n_input, n_hidden, n_output):
    train_x = np.array([[0.1, 0.1], [0.1, 0.2]])
    train_y = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.array([0.1, 0.1])
    b2 = np.array([0.1, 0.1])
    w1 = np.array([[0.1, 0.1], [0.2, 0.1]])
    w2 = np.array([[0.1, 0.1], [0.1, 0.2]])

    params = (w1, w2, b1, b2)
    network = nn.NeuralNetwork(sizes=[n_input, n_hidden, n_output], epochs=1, batch_size=2, params=params)

    network.SGD(train_x, train_y)


def main():
    n_input = sys.argv[1]
    n_hidden = sys.argv[2]
    n_output = sys.argv[3]

    # To run MNST data from the data files, run the program with file name arguments
    if len(sys.argv) > 3:
        # ftrain_x = sys.argv[4]
        # ftrain_y = sys.argv[5]
        # ftest_x = sys.argv[6]
        # ftest_y = sys.argv[7]
        run_mnst(n_input, n_hidden, n_output, "data/TrainDigitX.csv.gz", "data/TrainDigitY.csv.gz",
                 "data/TestDigitX.csv.gz", "data/TestDigitY.csv.gz")
    else:
        run_simple_network(n_input, n_hidden, n_output)


    # TODO Option to save biases and weights to csv file
    #   implement a load data function to load biases and weights
    #   this means we can also so the part 1 baises and weights to a
    #   csv


if __name__ == "__main__":
    main()