import numpy as np
import load_data as ld
import sys
import csv
import gzip
import neural_network as nn
from matplotlib import pyplot as plt


# This function runs the network on the assignment sheet (part 1)
def run_simple_network(n_input, n_hidden, n_output):
    train_x = np.array([[0.1, 0.1], [0.1, 0.2]])
    train_y = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.array([0.1, 0.1])
    b2 = np.array([0.1, 0.1])
    w1 = np.array([[0.1, 0.1], [0.2, 0.1]])
    w2 = np.array([[0.1, 0.1], [0.1, 0.2]])

    params = (w1, w2, b1, b2)
    network = nn.NeuralNetwork(sizes=[n_input, n_hidden, n_output], epochs=1,
                               l_rate=0.1, batch_size=2, params=params)

    network.SGD(train_x, train_y)


# This function takes a prediction vector and outputs it to the
# specified file
def save_prediction(prediction, outfile):
    with gzip.open(outfile, "w", newline='') as f:
        csv_w = csv.writer(f)
        for p in prediction:
            csv_w.writerow([p])


# This function takes 2 2D vector of data and 2
# 1D array of corresponding labels and plots the data
# on two subplots (one for l_rate and one for batch_size)
def plot_data(l_accuracies, l_labels, bs_accuracies, bs_labels, epoch=30):
    # Accuracy against Epoch, where epoch is the
    # accuracy's index
    fig, axs = plt.subplots(2)
    fig.suptitle("Accuracy vs Epoch")
    x = range(1, epoch+1)

    # For Learning rates
    for y, label in zip(l_accuracies, l_labels):
        axs[0].plot(x, y, label=label)
    axs[0].set_title("Learning Rates")
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')

    # For Batch sizes
    for y, label in zip(bs_accuracies, bs_labels):
        axs[1].plot(x, y, label=label)
    axs[1].set_title("Batch Sizes")
    axs[1].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    plt.show()


def main():
    n_input = int(sys.argv[1])
    n_hidden = int(sys.argv[2])
    n_output = int(sys.argv[3])


    # Input files
    # ftrain_x = sys.argv[4]
    # ftrain_y = sys.argv[5]
    # ftest_x = sys.argv[6]
    # ftest_y = sys.argv[7]

    # Output prediction files
    fpred_x = "predictions/PredictDigitY.tar.gz"
    fpred_x2 = "predictions/PredictDigitY2.tar.gz"
    # TODO, remove in favour of program arguments
    ftrain_x = "data/TrainDigitX.csv.gz"
    ftrain_y = "data/TrainDigitY.csv.gz"
    ftest_x = "data/TestDigitX.csv.gz"
    ftest_y = "data/TestDigitY.csv.gz"
    ftest_x2 = "data/TestDigitX2.csv.gz"

    # Load the data
    train_x, train_y, test_x, test_y, test_x2 = ld.load_data(ftrain_x, ftrain_y, ftest_x, ftest_y, ftest_x2)

    # To store the accuracy of a network for each epoch, for each network
    l_accuracies = []
    l_labels = []

    # 1. Neural network with sizes=[784, 30, 10], epochs=30, batch_size=20 and l_rate=3.0
    print("Part 1:")
    network = nn.NeuralNetwork(sizes=[784, 30, 10], l_rate=3.0,
                               batch_size=20, epochs=30)  # TODO change sizes to params
    network.SGD(train_x, train_y, test_x, test_y)
    l_accuracies.append(network.accuracy_per_epoch)
    l_labels.append("l_rate=3.0")

    # TODO
    # # Predict testX and testX2 and save results to file
    # prediction = network.predict(test_x)
    # print("Saving prediction for '"+ftest_x+"' to '"+fpred_x+"'")
    # save_prediction(prediction, fpred_x)

    # TODO
    # prediction = network.predict(test_x2)
    # print("Saving prediction for '" + ftest_x2 + "' to '" + fpred_x2 + "'")
    # save_prediction(prediction, fpred_x2)

    # 2. Run network with same settings a (1) except with
    # learning rates [0.001, 0.1, 1.0, 10.0, 100.0]
    print("Part 2: training with learning rates [0.001, 0.1, 1.0, 10.0, 100.0]")
    for l_rate in [0.001, 0.1, 1.0, 10.0, 100.0]:
        network = nn.NeuralNetwork(sizes=[784, 30, 10], l_rate=l_rate,
                                   batch_size=20)  # TODO change sizes to params
        network.SGD(train_x, train_y, test_x, test_y)
        # Record accuracies
        l_accuracies.append(network.accuracy_per_epoch)
        l_labels.append("l_rate="+str(l_rate))

    # 3. Run network with same settings a (1) except with
    # batch_sizes [1, 5, 10, 20, 100]
    print("Part 3: training with batch sizes [1, 5, 10, 20, 100]")
    bs_accuracies = []
    bs_labels = []
    for batch_size in [1, 5, 10, 20, 100]:
        network = nn.NeuralNetwork(sizes=[784, 30, 10], l_rate=3.0,
                                   batch_size=batch_size)  # TODO change sizes to params
        network.SGD(train_x, train_y, test_x, test_y)
        # Record accuracies
        bs_accuracies.append(network.accuracy_per_epoch)
        bs_labels.append("batch_size="+str(l_rate))

    # Plot results of 2. and 3.
    plot_data(l_accuracies, l_labels, bs_accuracies, bs_labels)

    # Last section, run the network with the params from
    # part one of the assessment
    print("Last section: manual calculation confirmation.")
    run_simple_network(n_input, n_hidden, n_output)



    # TODO Option to save biases and weights to csv file
    #   implement a load data function to load biases and weights
    #   this means we can also so the part 1 baises and weights to a
    #   csv


if __name__ == "__main__":
    main()