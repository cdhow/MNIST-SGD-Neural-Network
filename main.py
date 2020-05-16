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
    with gzip.open(outfile, "w+", newline='') as f:
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
    plt.savefig('EpochsVS{}.png'.format(cost_type))
    plt.show()


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
    fpred_x = "predictions/PredictDigitY.tar.gz"
    fpred_x2 = "predictions/PredictDigitY2.tar.gz"

    # TODO
    # # Predict testX and testX2 and save results to file
    # prediction = network.predict(test_x)
    # print("Saving prediction for 'TestDigitX.csv.gz' to '"+fpred_x+"'")
    # save_prediction(prediction, fpred_x)

    # TODO
    # prediction = network.predict(test_x2)
    # print("Saving prediction for 'TestDigitX2.csv.gz' to '" + fpred_x2 + "'")
    # save_prediction(prediction, fpred_x2)


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
    # batch_sizes rates [0.001, 0.1, 1.0, 10.0, 100.0]

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


# This is the main program wrapper
def run(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, test_x2=None):
    print("Welcome to the Stochastic Gradient Descent Neural Network by Caleb Howard.")
    print("Please select from the following options:")
    print("1. Train Neural Network and output Predictions to file.")
    print("2. Train Neural Network with varying learning rates.")
    print("3. Train Neural Network with varying batch sizes.")
    print("4. Both (2) and (1) and graph costs against epochs.")
    print("Note: To run the small network for manual calculation confirmation,"
          " run program with arguments (2 2 2)")
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
        # For accuracies
        plot_data(l_y=l_acur, l_labels=l_labels, bs_y=bs_acur,
                  bs_labels=bs_labels, cost_type="Accuracies")
        # For quadratic cost
        plot_data(l_y=l_q_cost, l_labels=l_labels, bs_y=bs_q_cost,
                  bs_labels=bs_labels, cost_type="Quadratic Cost")
        # For cross entropy
        plot_data(l_y=l_ce_cost, l_labels=l_labels, bs_y=bs_ce_cost,
                  bs_labels=bs_labels, cost_type="Cross Entropy")


def file_debug(label, accuracies, header):
    with open('predictions/debug.txt', 'a') as f:
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
    # # l_acur = [[0.9332, 0.9466, 0.9493, 0.9525, 0.9524, 0.953, 0.9537, 0.9544, 0.9543, 0.9544, 0.9545, 0.955, 0.9548, 0.9546, 0.9556, 0.9562, 0.9564, 0.9569, 0.9568, 0.9573, 0.9566, 0.9565, 0.9566, 0.9574, 0.9578, 0.9579, 0.958, 0.9575, 0.9575, 0.9574],
    # # [0.1433, 0.2077, 0.2339, 0.2566, 0.2711, 0.2872, 0.302, 0.3187, 0.3344, 0.351, 0.3695, 0.3861, 0.4069, 0.4248, 0.443, 0.4617, 0.4813, 0.4985, 0.5132, 0.528, 0.5409, 0.5533, 0.5643, 0.5742, 0.5836, 0.592, 0.6015, 0.6091, 0.6166, 0.6235],
    # # [0.8322, 0.8802, 0.8965, 0.9048, 0.9099, 0.9144, 0.9179, 0.9208, 0.923, 0.924, 0.9256, 0.9273, 0.9286, 0.9299, 0.9313, 0.9323, 0.9334, 0.9355, 0.9364, 0.9374, 0.9375, 0.9378, 0.9392, 0.9398, 0.9404, 0.941, 0.9414, 0.9418, 0.9423, 0.9426],
    # # [0.9211, 0.9355, 0.9427, 0.946, 0.9493, 0.9514, 0.9522, 0.9538, 0.955, 0.9556, 0.9558, 0.9568, 0.9573, 0.9582, 0.9588, 0.9592, 0.9595, 0.9602, 0.9601, 0.9604, 0.9608, 0.9617, 0.9614, 0.9615, 0.9618, 0.9616, 0.9617, 0.9618, 0.9618, 0.9619],
    # # [0.9268, 0.933, 0.9386, 0.9404, 0.942, 0.9439, 0.9458, 0.9418, 0.9469, 0.9443, 0.9472, 0.946, 0.9471, 0.9475, 0.9499, 0.9477, 0.9489, 0.9501, 0.9521, 0.9499, 0.9508, 0.9517, 0.9502, 0.9463, 0.9496, 0.9495, 0.9527, 0.9507, 0.9497, 0.9509],
    # # [0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955]]
    # #
    # # bs_aurr = [[0.7863, 0.7322, 0.8356, 0.8272, 0.8429, 0.8358, 0.8544, 0.8551, 0.842, 0.8541, 0.837, 0.8688, 0.8498, 0.8669, 0.8833, 0.883, 0.8571, 0.8651, 0.8748, 0.8662, 0.8857, 0.8711, 0.8726, 0.8839, 0.8793, 0.874, 0.878, 0.8789, 0.8878, 0.8864],
    # #  [0.9332, 0.9371, 0.9426, 0.9427, 0.9426, 0.9471, 0.9483, 0.9468, 0.9508, 0.9499, 0.9475, 0.9498, 0.9528, 0.9488, 0.9485, 0.951, 0.9447, 0.9488, 0.9499, 0.9463, 0.9507, 0.9513, 0.9507, 0.9541, 0.952, 0.9508, 0.9544, 0.9524, 0.9519, 0.9499],
    # #  [0.9374, 0.9459, 0.9482, 0.9493, 0.9492, 0.9531, 0.9531, 0.9544, 0.9528, 0.9539, 0.9534, 0.9538, 0.9549, 0.9557, 0.9561, 0.9553, 0.9567, 0.956, 0.956, 0.9541, 0.9544, 0.9558, 0.9567, 0.9578, 0.9548, 0.9568, 0.9552, 0.9542, 0.9551, 0.9543],
    # # [0.9305, 0.9389, 0.9438, 0.946, 0.9493, 0.9515, 0.953, 0.9542, 0.9527, 0.953, 0.9529, 0.9532, 0.9528, 0.9523, 0.9535, 0.9538, 0.9545, 0.9543, 0.9541, 0.9536, 0.955, 0.9543, 0.9543, 0.9533, 0.9537, 0.9537, 0.9542, 0.9539, 0.954, 0.9531],
    # # [0.9083, 0.9234, 0.9295, 0.9344, 0.9384, 0.941, 0.9431, 0.9438, 0.9455, 0.9479, 0.9489, 0.9504, 0.9516, 0.9531, 0.9531, 0.953, 0.9539, 0.9546, 0.9547, 0.9553, 0.9556, 0.956, 0.9562, 0.9567, 0.957, 0.9574, 0.9578, 0.9583, 0.9588, 0.9593]]
    # #
    # # l_labels = ["l_rate=3.0", "l_rate=0.001", "l_rate=0.001", "l_rate=0.1", "l_rate=1.0", "l_rate=10.0", "l_rate=100.0"]
    # # bs_labels = ["batch_size=1", "batch_size=5", "batch_size=10", "batch_size=20", "batch_size=100"]
    #
    # plot_data(l_acur, l_labels, bs_aurr, bs_labels)
    # return
    if len(sys.argv) > 4:

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
        run(train_x, train_y, test_x, test_y, n_input, n_hidden, n_output, test_x2)
    else:
        # Run the network with the params from
        # part one of the assessment
        print("Last section: manual calculation confirmation.")
        run_simple_network(n_input, n_hidden, n_output)

    # TODO Option to save biases and weights to csv file
    #   implement a load data function to load biases and weights
    #   this means we can also so the part 1 baises and weights to a
    #   csv


if __name__ == "__main__":
    main()
