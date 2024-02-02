import math
from sklearn.metrics import ConfusionMatrixDisplay
# from Main import *
from tkinter import PhotoImage
from tkinter import ttk
from random import random
from tkinter import *
from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random


data = pd.read_csv("Dry_Bean_Dataset.csv")


# Sigmoid and its derivative


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    result = []
    for i in x:
        result.append(1 - tanh(i) ** 2)
    return result


def fetch_features_and_classes():
    data['MinorAxisLength'] = data['MinorAxisLength'].interpolate(
        limit_direction='both', limit_area='inside', method='linear')

    class_mapping = {'BOMBAY': [1, 0, 0], 'CALI': [0, 1, 0], 'SIRA': [0, 0, 1]}
    data['Class'] = data['Class'].map(class_mapping)

    class1_rows = []
    class2_rows = []
    class3_rows = []

    for row_num in range(len(data)):
        if data['Class'][row_num] == [1, 0, 0]:
            class1_rows.append(data.iloc[row_num])
        elif data['Class'][row_num] == [0, 1, 0]:
            class2_rows.append(data.iloc[row_num])
        elif data['Class'][row_num] == [0, 0, 1]:
            class3_rows.append(data.iloc[row_num])

    # randomly select 30 rows from each class
    training_rows_class1 = random.sample(class1_rows, 30)
    training_rows_class2 = random.sample(class2_rows, 30)
    training_rows_class3 = random.sample(class3_rows, 30)

    # append all samples in one list
    training_selected_rows = training_rows_class1 + training_rows_class2 + training_rows_class3
    random.shuffle(training_selected_rows)

    y_train = [row['Class'] for row in training_selected_rows]
    x_train = [row[0:5] for row in training_selected_rows]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # drop the training data from the original to get the test data ####################################################

    df_class1_rows = pd.DataFrame(class1_rows)
    df_class2_rows = pd.DataFrame(class2_rows)
    df_class3_rows = pd.DataFrame(class3_rows)

    df_training_class1_rows = pd.DataFrame(training_rows_class1)
    df_training_class2_rows = pd.DataFrame(training_rows_class2)
    df_training_class3_rows = pd.DataFrame(training_rows_class3)

    class1_test_rows = df_class1_rows.drop(df_training_class1_rows.index).values.tolist()
    class2_test_rows = df_class2_rows.drop(df_training_class2_rows.index).values.tolist()
    class3_test_rows = df_class3_rows.drop(df_training_class3_rows.index).values.tolist()

    ###################################################################################

    total_test_rows = class1_test_rows + class2_test_rows + class3_test_rows

    random.shuffle(total_test_rows)

    df_total_test = pd.DataFrame(total_test_rows, columns=df_training_class1_rows.columns)

    x_test_data = df_total_test.iloc[:, 0:5].values.tolist()  # Assuming the features are in the first 5 columns
    y_test_data = df_total_test['Class'].values.tolist()

    return x_train, y_train, x_test_data, y_test_data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Propagation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def forward_propagation(x, weights_matrix, act_func, bias_list):
    hidden_layers_list = []
    first_layer_outputs = []
    # Input to the first layer
    input_to_first_layer_weights = weights_matrix[0]
    if act_func == "sigmoid":
        first_layer_outputs = [sigmoid(np.dot(x, input_to_first_layer_weights[w])+bias_list[0][w]) for w in range(len(input_to_first_layer_weights))]
    elif act_func == "tanh":
        first_layer_outputs = [tanh(np.dot(x, input_to_first_layer_weights[w])+bias_list[0][w]) for w in range(len(input_to_first_layer_weights))]
    else:
        print("Wrong activation function name")
    hidden_layers_list.append(first_layer_outputs)

    # Hidden layers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input = first_layer_outputs
    layer_outputs = []
    for i in range(1, len(weights_matrix) - 1):
        layer_weights = weights_matrix[i]
        if act_func == "sigmoid":
            layer_outputs = [sigmoid(np.dot(input, layer_weights[w])+bias_list[i][w]) for w in range(len(layer_weights))]
        elif act_func == "tanh":
            layer_outputs = [tanh(np.dot(input, layer_weights[w])+bias_list[i][w]) for w in range(len(layer_weights))]
        else:
            print("Wrong activation function name")

        hidden_layers_list.append(layer_outputs)
        input = layer_outputs

    # Output layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    output_weights = weights_matrix[-1]
    last_layer_outputs = []

    for i in range(3):
        if act_func == "sigmoid":
            output_neuron = sigmoid(np.dot(input, output_weights[i]))
        elif act_func == "tanh":
            output_neuron = tanh(np.dot(input, output_weights[i]))
        else:
            print("Wrong activation function name")
        last_layer_outputs.append(output_neuron)
    hidden_layers_list.append(last_layer_outputs)
    return last_layer_outputs, hidden_layers_list

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def train_network(num_of_hidden_layers, neurons, eta, epochs, bias, activation, features, target):
    # activation function choice
    if activation == "sigmoid":
        act_func = sigmoid
    elif activation == "tanh":
        act_func = tanh

    def initialize_weights(neurons, input_size):
        weights_matrix = []
        input_weights = []
        hidden_layers_bias = []
        if bias: # if we use bias I will update it , if not, it will remain 0, so I initialized it with 0
            for i in neurons:
                hidden_layers_bias.append(np.zeros(i))

        # Input to the first layer weights
        for i in range(neurons[0]):
            weights = np.random.uniform(-1.0, 1.0, input_size)
            input_weights.append(weights)
        weights_matrix.append(input_weights)

        # Hidden layer weights
        for i in range(1, len(neurons)):
            layer_weights = []
            for j in range(neurons[i]):
                neuron_weights = np.random.uniform(-1.0, 1.0, neurons[i - 1])
                layer_weights.append(neuron_weights)
            weights_matrix.append(layer_weights)

        # Output layer weights
        output_layer_weights = []
        for i in range(3):
            output_weights = np.random.uniform(-1.0, 1.0, neurons[-1])
            output_layer_weights.append(output_weights)
        weights_matrix.append(output_layer_weights)

        return weights_matrix, hidden_layers_bias

    w_matrix, hidden_layers_bias_list = initialize_weights(neurons, 5)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Propagation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def clac_signal_error_for_neuron(previous_layer_signal_errors, w_matrix, f_net, hidden_layer_number):
        layer_signal_error = []
        new_weight_matrix = np.transpose(w_matrix[hidden_layer_number])
        for i in range(len(new_weight_matrix)):
            w_mult_sig_err = np.dot(previous_layer_signal_errors, new_weight_matrix[i])
            net = f_net[hidden_layer_number - 1][i]
            if activation == "sigmoid":
                neuron_signal_error = w_mult_sig_err * net * (1 - net)
                layer_signal_error.append(neuron_signal_error)
            elif activation == "tanh":
                neuron_signal_error = w_mult_sig_err * (1 - tanh(net) ** 2)
                layer_signal_error.append(neuron_signal_error)
            else:
                print("Wrong activation function name")

        return layer_signal_error

    def update_weights(signal_errors, input_layer, layer_outputs, original_weights, learning_rate, bias_list):
        updated_weights = []
        layer_updated_weights = []
        layer_updated_bias = []
        inp = input_layer
        updated_bias_list = []
        for i in range(len(original_weights)):
            layer_weights = original_weights[i]
            signal_error_matrix = signal_errors[::-1]

            for x in range(len(layer_outputs[i])):
                weight_updating_term = learning_rate * np.dot(signal_error_matrix[i][x], inp)
                updated_neuron_weights = layer_weights[x] + weight_updating_term
                layer_updated_weights.append(updated_neuron_weights)
                # original_weights length = 4, and we have 3 bias lists 0 based then our boundaries are form 0 to 2
                if bias and i < len(original_weights)-1:
                    layer_updated_bias.append(learning_rate * signal_error_matrix[i][x] + bias_list[i][x])
                elif bias == False:
                    updated_bias_list = bias_list
            updated_weights.append(layer_updated_weights)
            updated_bias_list.append(layer_updated_bias)
            layer_updated_weights = []
            layer_updated_bias = []
            inp = layer_outputs[i]
        return updated_weights, updated_bias_list

    def backward_propagation(target, last_layer_actual_output, weight_matrix, hidden_layers_list, num_of_hidden_layers):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ compute the signal error at the output layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        hidden_layers_signal_errors = []
        last_layer_actual_output = np.array(last_layer_actual_output)
        output_error = np.subtract(target, last_layer_actual_output)

        if activation == "sigmoid":
            signal_error = output_error * last_layer_actual_output * (1 - last_layer_actual_output)
        elif activation == "tanh":
            signal_error = output_error * tanh_derivative(last_layer_actual_output)
        else:
            print("Wrong activation function name")
        hidden_layers_signal_errors.append(signal_error)

        # ~~~~~~~~~~~~~~~~~~~~~~ compute signal error for each neuron at each hidden layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sig_error = signal_error
        updated_weights = []
        for i in range(num_of_hidden_layers, 0, -1):
            next_sig_error = clac_signal_error_for_neuron(sig_error, weight_matrix, hidden_layers_list, i)
            sig_error = next_sig_error
            hidden_layers_signal_errors.append(sig_error)
        return signal_error, hidden_layers_signal_errors

    # Training loop
    new_weights = w_matrix
    new_bias = hidden_layers_bias_list
    for epoch in range(epochs):
        total_error = 0

        for i in range(len(features)):
            # Forward Propagation
            last_layer_outputs, hidden_layers_list = forward_propagation(features[i], new_weights, activation, new_bias)

            # Backward Propagation
            output_layer_signal_error, signal_errors = backward_propagation(target[i], last_layer_outputs, new_weights,
                                                                            hidden_layers_list, num_of_hidden_layers)
            # Update weights
            new_weights, new_bias = update_weights(signal_errors, features[i], hidden_layers_list, new_weights, eta,
                                                   new_bias)

    return new_weights, new_bias


import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(matrix, classes):
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def confusion_matrix(predictions, targets, classes):
    # Initialize confusion matrix
    num_classes = len(classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for pred, target in zip(predictions, targets):
        # Convert predicted and target vectors to class indices
        pred_index = np.argmax(pred)
        target_index = np.argmax(target)

        # Update confusion matrix
        matrix[target_index, pred_index] += 1

    return matrix


def tst_neural_network(weights_matrix, activation, features, target, classes, bias_list):
    correct_predictions = 0
    predictions = []

    for i in range(0, len(features)):
        # Forward Propagation
        last_layer_outputs, hidden_layers_list = forward_propagation(features[i], weights_matrix, activation, bias_list)

        # Compute error
        output_error = np.subtract(target[i], last_layer_outputs[0])

        # Convert the output to predicted class
        predicted_class = [0, 0, 0]
        for j in range(len(output_error)):
            if max(output_error) == output_error[j]:
                predicted_class[j] = 1
                break

        predictions.append(predicted_class)

        # Check if the prediction is correct
        if predicted_class == target[i]:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(target) * 100

    confusion_mat = confusion_matrix(predictions, target, classes)

    # Plot confusion matrix
    plot_confusion_matrix(confusion_mat, classes)
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def train_neural_network():
    learning_rate = float(learning_rate_entry.get())
    num_epochs = int(num_epochs_entry.get())
    num_hidden_layers = int(num_hidden_layers_entry.get())
    neurons_per_layer = [int(x) for x in num_neurons_entry.get().split(',')]
    activation_function = activation_function_var.get()
    add_bias = add_bias_var.get()
    x_train, y_train, x_test, y_test = fetch_features_and_classes()
    w, bias_list = train_network(num_hidden_layers, neurons_per_layer, learning_rate, num_epochs, add_bias,
                                 activation_function, x_train, y_train)
    classes = ["BOMBAY", "CALI", "SIRA"]
    test_acc = tst_neural_network(w, activation_function, x_test, y_test, classes, bias_list)
    # Create a new window for displaying results
    result_window = tk.Toplevel(root)
    result_window.title("Neural Network Results")

    # Display accuracy in a label
    accuracy_label = tk.Label(result_window, text=f"Test Accuracy: {test_acc:.2f}%", font=('Helvetica', 16))
    accuracy_label.pack(pady=20)


# Create the main window
root = tk.Tk()
root.title("Neural Network GUI")

# Label and Entry for Learning Rate
tk.Label(root, text="Learning Rate:", background='white').grid(row=0, column=0, padx=10, pady=5)
learning_rate_entry = tk.Entry(root)
learning_rate_entry.grid(row=0, column=1, padx=10, pady=5)

# Label and Entry for Number of Epochs
tk.Label(root, text="Number of Epochs:", background='white').grid(row=1, column=0, padx=10, pady=5)
num_epochs_entry = tk.Entry(root)
num_epochs_entry.grid(row=1, column=1, padx=10, pady=5)

# Label and Entry for Number of Hidden Layers
tk.Label(root, text="Number of Hidden Layers:", background='white').grid(row=2, column=0, padx=10, pady=5)
num_hidden_layers_entry = tk.Entry(root)
num_hidden_layers_entry.grid(row=2, column=1, padx=10, pady=5)

# Label and Entry for Number of Neurons
tk.Label(root, text="Number of Neurons (comma-separated):", background='white').grid(row=3, column=0, padx=10, pady=5)
num_neurons_entry = tk.Entry(root)
num_neurons_entry.grid(row=3, column=1, padx=10, pady=5)

# Label and Dropdown for Activation Function
tk.Label(root, text="Select Your Activation Function:", background='white').grid(row=4, column=0, padx=10, pady=5)
activation_functions = ["sigmoid", "tanh"]
activation_function_var = tk.StringVar(root)
activation_function_var.set(activation_functions[0])
activation_function_dropdown = ttk.Combobox(root, textvariable=activation_function_var, values=activation_functions)
activation_function_dropdown.grid(row=4, column=1, padx=10, pady=5)

# Label and Checkbox for Add Bias
add_bias_var = tk.BooleanVar(root)
add_bias_var.set(True)
add_bias_checkbox = tk.Checkbutton(root, text="Add Bias", variable=add_bias_var, background='white')
add_bias_checkbox.grid(row=5, column=0, columnspan=2, pady=5)

# Button to initiate training
train_button = tk.Button(root, text="Train Neural Network", command=train_neural_network, background='#4CAF50',
                         fg='white')
train_button.grid(row=6, column=0, columnspan=2, pady=10)

# Run the main loop
root.mainloop()
