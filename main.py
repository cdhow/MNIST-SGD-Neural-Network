import numpy as np
import load_data as ld
import sys
import csv
import gzip
import neural_network as nn
from matplotlib import pyplot as plt


# -------------- UTILITY FUNCTIONS -------------------------#

# This function takes a prediction vector and outputs it to the
# specified file
def save_prediction(prediction, outfile):
    with gzip.open(outfile, "wt", newline='') as f:
        csv_w = csv.writer(f)
        for p in prediction:
            csv_w.writerow([p])


# This function takes 2 2D vector of data and 2
# 1D array of corresponding labels and plots the data
# on two subplots (one for l_rate and one for batch_size)
def plot_data(l_y=None, l_labels=None, bs_y=None, bs_labels=None, epoch=30, cost_type=None):
    # Accuracy, cross entropy or quadratic cost
    # against Epoch, where epoch is the y's index
    fig, axs = plt.subplots(2)
    x = range(1, epoch + 1)

    # For Learning rates
    if l_y is not None:
        for y, label in zip(l_y, l_labels):
            axs[0].plot(x, y, label=label)
        axs[0].set_title("Learning Rates")
        axs[0].legend()
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel(cost_type)

    # For Batch sizes
    if bs_y is not None:
        for y, label in zip(bs_y, bs_labels):
            axs[1].plot(x, y, label=label)
        axs[1].set_title("Batch Sizes")
        axs[1].legend()
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel(cost_type)

    plt.tight_layout()
    print("Saving graph to plots/EpochsVS{}.png".format(cost_type).replace(" ", ""))
    plt.savefig('plots/EpochsVS{}.png'.format(cost_type).replace(" ", ""))


# For saving basic plots (used in optional hyper-params section)
def save_basic_plot(epochs, y_data, y_label):
    x = range(1, epochs + 1)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.plot(x, y_data)
    print("Saving graph to plots/EpochsVS{}.png".format(y_label))
    plt.savefig("plots/EpochsVS{}.png".format(y_label))


# -----------FUNCTIONS FOR EACH PART OF THE ASSESSMENT ---------#

# This function runs the network on the assignment sheet (part 1)
# For confirmation of the manual calculations
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


# This function trains the network and then runs the test data
# to output predictions to the corresponding prediction file
def predict_test_data(train_x, train_y, test_x, test_y, test_x2, n_input, n_hidden, n_output):
    # 1. Neural network with sizes=[784, 30, 10], epochs=30, batch_size=20 and l_rate=3.0
    print("Part 1:")
    network = nn.NeuralNetwork(sizes=[n_input, n_hidden, n_output],
                               l_rate=3.0,
                               batch_size=20,
                               epochs=30)
    network.SGD(train_x, train_y, test_x, test_y)

    # Output prediction files
    fpred_x = "predictions/PredictDigitY.csv.gz"
    fpred_x2 = "predictions/PredictDigitY2.csv.gz"

    # Predict testX and testX2 and save results to file
    prediction = network.predict(test_x)
    print("Saving prediction for 'TestDigitX.csv.gz' to '"+fpred_x+"'")
    save_prediction(prediction, fpred_x)

    prediction = network.predict(test_x2)
    print("Saving prediction for 'TestDigitX2.csv.gz' to '" + fpred_x2 + "'")
    save_prediction(prediction, fpred_x2)


# This function trains the network with different learning rates and
# outputs the results to three graphs, accuracy vs epoch, quadratic cost vs epoch,
# and cross entropy cost vs epoch
def compare_l_rates(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, plot_costs=False):
    # 2. Run network with same settings a (1) except with
    # learning rates [0.001, 0.1, 1.0, 10.0, 100.0]

    # To store the accuracy, cross entropy, and quadratic cost
    # of a network for each epoch, for each network
    accuracies = []
    q_costs = []
    ce_costs = []
    labels = []

    print("Part 2: training with learning rates [0.001, 0.1, 1.0, 10.0, 100.0]")
    for l_rate in [0.001, 0.1, 1.0, 10.0, 100.0]:
        network = nn.NeuralNetwork(sizes=[n_input, n_hidden, n_output],
                                   l_rate=l_rate,
                                   batch_size=20)
        network.SGD(train_x, train_y, test_x, test_y)

        # Record costs
        accuracies.append(network.accuracy_per_epoch)
        q_costs.append(network.quadratic_cost_per_epoch)
        ce_costs.append(network.cross_entropy_per_epoch)
        labels.append("l_rate=" + str(l_rate))

    if plot_costs:
        plot_data(l_y=accuracies, l_labels=labels, cost_type="Accuracies")
        plot_data(l_y=q_costs, l_labels=labels, cost_type="Quadratic Cost")
        plot_data(l_y=ce_costs, l_labels=labels, cost_type="Cross Entropy Cost")

    return labels, accuracies, q_costs, ce_costs


# This function trains the network with different batch_sizes (l_rate=3.0) and
# outputs the results to three graphs, accuracy vs epoch, quadratic cost vs epoch,
# and cross entropy cost vs epoch
def compare_batch_sizes(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, plot_costs=False):
    # 2. Run network with same settings a (1) except with
    # batch_sizes [1, 5, 10, 20, 100]

    # To store the accuracy, cross entropy, and quadratic cost
    # of a network for each epoch, for each network
    accuracies = []
    q_costs = []
    ce_costs = []
    labels = []

    print("Part 3: training with batch sizes [1, 5, 10, 20, 100]")

    for batch_size in [1, 5, 10, 20, 100]:
        network = nn.NeuralNetwork(sizes=[n_input, n_hidden, n_output], l_rate=3.0,
                                   batch_size=batch_size)
        network.SGD(train_x, train_y, test_x, test_y)

        # Record costs
        accuracies.append(network.accuracy_per_epoch)
        q_costs.append(network.quadratic_cost_per_epoch)
        ce_costs.append(network.cross_entropy_per_epoch)
        labels.append("batch_size=" + str(batch_size))

    if plot_costs:
        plot_data(bs_y=accuracies, bs_labels=labels, cost_type="Accuracies")
        plot_data(bs_y=q_costs, bs_labels=labels, cost_type="Quadratic Cost")
        plot_data(bs_y=ce_costs, bs_labels=labels, cost_type="Cross Entropy Cost")

    return labels, accuracies, q_costs, ce_costs


# This function allows the user to choose their own hyper params to train the network
def optional_hyper_params(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output):
    l_rate = float(input("learning rate:"))
    batch_size = int(input("batch size:"))
    epochs = int(input("epochs:"))
    o = input("Plot cost results? (y/n):")
    pl = o == "y"
    network = nn.NeuralNetwork(sizes=[n_input, n_hidden, n_output], l_rate=l_rate,
                               batch_size=batch_size, epochs=epochs)
    network.SGD(train_x, train_y, test_x, test_y)

    if pl:
        save_basic_plot(epochs, network.accuracy_per_epoch, "Accuracy")
        save_basic_plot(epochs, network.quadratic_cost_per_epoch, "Quadratic Cost")
        save_basic_plot(epochs, network.cross_entropy_per_epoch, "Cross Entropy Cost")


# This is the main program wrapper
def run(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, test_x2=None):
    print("\nWelcome to the Stochastic Gradient Descent Neural Network by Caleb Howard.")
    print("Note: To run the small network for manual calculation confirmation,"
          " run program with arguments (2 2 2)")
    opt = -1
    while opt != '6':
        print("\nPlease select from the following options:")
        print("1. Train Neural Network and output Predictions to file.")
        print("2. Train Neural Network with varying learning rates.")
        print("3. Train Neural Network with varying batch sizes.")
        print("4. Both (2) and (1) and graph costs against epochs.")
        print("5. Train network with optional parameters.")
        print("6. Exit program.")

        opt = input()

        if opt == '1':
            predict_test_data(train_x, train_y, test_x, test_y, test_x2, n_input, n_hidden, n_output)

        elif opt == '2':
            o = input("Plot cost results? (y/n):")
            pl = o == "y"
            compare_l_rates(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, pl)

        elif opt == '3':
            o = input("Plot cost results? (y/n):")
            pl = o == "y"
            compare_batch_sizes(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, pl)

        elif opt == '4':
            l_labels, l_acur, l_q_cost, l_ce_cost = compare_l_rates(train_x, train_y, test_x, test_y,
                                                                    n_input, n_hidden, n_output)
            bs_labels, bs_acur, bs_q_cost, bs_ce_cost = compare_batch_sizes(train_x, train_y, test_x, test_y,
                                                                            n_input, n_hidden, n_output)
            # TODO Remove
            file_debug(l_labels, l_acur, "Accuracies")
            file_debug(l_labels, l_q_cost, "Q Cost")
            file_debug(l_labels, l_ce_cost, "CE cost")

            file_debug(bs_labels, bs_acur, "Accuracies")
            file_debug(bs_labels, bs_q_cost, "Q Cost")
            file_debug(bs_labels, bs_ce_cost, "CE cost")


            # For accuracies
            plot_data(l_y=l_acur, l_labels=l_labels, bs_y=bs_acur,
                      bs_labels=bs_labels, cost_type="Accuracies")
            # For quadratic cost
            plot_data(l_y=l_q_cost, l_labels=l_labels, bs_y=bs_q_cost,
                      bs_labels=bs_labels, cost_type="Quadratic Cost")
            # For cross entropy
            plot_data(l_y=l_ce_cost, l_labels=l_labels, bs_y=bs_ce_cost,
                      bs_labels=bs_labels, cost_type="Cross Entropy")

        elif opt == '5':
            optional_hyper_params(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output)

        elif opt != '6':
            print("Invalid option: " + opt)


def file_debug(label, accuracies, header):
    with open('debug.txt', 'a') as f:
        f.write(header)
        for l, y in zip(label, accuracies):
            f.write("%s, " % l)
            for i in y:
                f.write("%s, " % i)
            f.write("\n")
        f.write("\n")


def main():
    n_input = int(sys.argv[1])
    n_hidden = int(sys.argv[2])
    n_output = int(sys.argv[3])

    if len(sys.argv) > 3:

        # Input files
        # ftrain_x = sys.argv[4]
        # ftrain_y = sys.argv[5]
        # ftest_x = sys.argv[6]
        # ftest_y = sys.argv[7]

        # TODO, remove in favour of program arguments
        ftrain_x = "data/TrainDigitX.csv.gz"
        ftrain_y = "data/TrainDigitY.csv.gz"
        ftest_x = "data/TestDigitX.csv.gz"
        ftest_y = "data/TestDigitY.csv.gz"
        ftest_x2 = "data/TestDigitX2.csv.gz"

        # Load the data
        train_x, train_y, test_x, test_y, test_x2 = ld.load_data(ftrain_x, ftrain_y, ftest_x, ftest_y, ftest_x2)
        run(train_x, train_y, test_x, test_y, 784, 30, 10, test_x2)  # Todo: update to take program arguments
    else:
        # Run the network with the params from
        # part one of the assessment
        print("Last section: manual calculation confirmation.")
        run_simple_network(n_input, n_hidden, n_output)


if __name__ == "__main__":
    main()
