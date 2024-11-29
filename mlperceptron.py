import random
import time

E = 2.71828


def initialize_weight(previous_layer, next_layer):
    rows = []
    for _ in range(next_layer):
        cols = []
        for _ in range(previous_layer):
            cols.append(random.random())
        rows.append(cols)
    return rows


def initialize_helper(input_layer, hidden_layer, output_layer):
    # initialize weights from input to hidden layers with random value, 0 <= val < 1
    hidden_weights = initialize_weight(input_layer, hidden_layer)
    # initialize weights from hidden to output layer
    output_weights = initialize_weight(hidden_layer, output_layer)

    # set 0 for both hidden biases and output biases, update biases in backpropagation
    hidden_bias = [0.0 for _ in range(hidden_layer)]
    output_bias = [0.0 for _ in range(output_layer)]

    return hidden_weights, output_weights, hidden_bias, output_bias


def calc_ouput_helper(prev_layer_weights, next_layer_weights, biases):
    # x = (Sigma (Xi x Wi))  + bias
    layer_values = []
    for layer_weight, bias in zip(next_layer_weights, biases):
        weight_sum = 0
        for input, weight in zip(prev_layer_weights, layer_weight):
            weight_sum += input * weight

        layer_values.append(weight_sum + bias)
    return layer_values


def sigmoid(x):
    return 1 / (1 + (E ** (-x)))


def forward_propagation(
    inputs, hidden_weights, hidden_bias, output_weights, output_bias
):
    # step 1: calculate the outputs of all neurons in the hidden layer
    hidden_layer_values = calc_ouput_helper(inputs, hidden_weights, hidden_bias)

    hidden_layer_output = [sigmoid(x) for x in hidden_layer_values]

    # step 2: calculate the outputs of all neuron(s) in the output layer
    # similar for hidden layer, we do the same for output layer
    output_layer_values = calc_ouput_helper(
        hidden_layer_output, output_weights, output_bias
    )

    output_layer_output = [sigmoid(x) for x in output_layer_values]

    return hidden_layer_output, output_layer_output


def update_weight_and_bias_helper(
    weights, biases, deltas, layer_outputs, learning_rate
):
    # w = w + (learning rate x O/X x delta)
    # update weights
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weights[i][j] += learning_rate * deltas[i] * layer_outputs[j]
    # update biases
    # b = b + (learning rate * delta)
    for i in range(len(biases)):
        biases[i] += learning_rate * deltas[i]


def back_propagation(
    inputs,
    targets,
    hidden_layer_output,
    output_layer_output,
    hidden_weights,
    hidden_bias,
    output_weights,
    output_bias,
    learning_rate,
):
    # step 3: calculate output error
    # delta k = Ok x (1 - Ok) x (target - Ok)
    output_deltas = []
    for target, output in zip(targets, output_layer_output):
        err = target - output
        delta = output * (1 - output) * err
        output_deltas.append(delta)

    # step 4: update weight between hidden-output layer
    update_weight_and_bias_helper(
        output_weights, output_bias, output_deltas, hidden_layer_output, learning_rate
    )

    # step 5: calculate hidden error
    # delta j = Oj x (1 - Oj) x (Sigma (delta k x Wjk))
    hidden_deltas = []
    for i in range(len(output_weights[0])):
        sigma_val = 0
        for j in range(len(output_weights)):
            sigma_val += output_weights[j][i] * output_deltas[j]

        hidden_delta = hidden_layer_output[i] * (1 - hidden_layer_output[i]) * sigma_val
        hidden_deltas.append(hidden_delta)

    # step 6: update weight between input-hidden layer, similar to step 4
    update_weight_and_bias_helper(
        hidden_weights, hidden_bias, hidden_deltas, inputs, learning_rate
    )


def train_multilayer_perceptron(
    train_data, targets, input_layer, hidden_layer, output_layer, epochs, learning_rate
):
    # initialize weights and biases using helper func
    hidden_weights, output_weights, hidden_bias, output_bias = initialize_helper(
        input_layer, hidden_layer, output_layer
    )

    paired_data = list(zip(train_data, targets))

    # start training
    for _ in range(epochs):
        sum_squared_errors = 0
        for pair in paired_data:
            inputs = pair[0]
            target = pair[1]

            hidden_layer_output, output_layer_output = forward_propagation(
                inputs, hidden_weights, hidden_bias, output_weights, output_bias
            )

            print("Input\t\tDesired output\t\tActual output\t\tError")
            for desired, actual in zip(target, output_layer_output):
                err = desired - actual
                print(f"{inputs}\t\t\t{desired}\t\t{actual}\t{err}")
                sum_squared_errors += err**2

            back_propagation(
                inputs,
                target,
                hidden_layer_output,
                output_layer_output,
                hidden_weights,
                hidden_bias,
                output_weights,
                output_bias,
                learning_rate,
            )

        print(f"Sum of squared errors: {sum_squared_errors}\n")

    return hidden_weights, hidden_bias, output_weights, output_bias


def main():
    train_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = list()

    print("==============Multilayer Perceptron==============")
    while True:
        print("There are some problems you can choose to train.")
        print("x. Xor problem")
        print("a. And problem")
        print("o. Or problem")
        print("n. Nand problem")

        choice = input("What type of problem do you want to train: ")
        if choice == "x" or choice == "X":
            targets = [[0], [1], [1], [0]]
        elif choice == "a" or choice == "A":
            targets = [[0], [0], [0], [1]]
        elif choice == "o" or choice == "O":
            targets = [[0], [1], [1], [1]]
        elif choice == "n" or choice == "N":
            targets = [[1], [1], [1], [0]]
        else:
            print("Invalid option. Please key in valid option!")

        if len(targets) > 0:
            break

    # hyperparameters
    input_layer = 2
    hidden_layer = 4
    output_layer = 1
    epochs = 10000
    learning_rate = 0.1

    start = time.time()
    # train multilayer perceptron
    hidden_weights, hidden_bias, output_weights, output_bias = (
        train_multilayer_perceptron(
            train_data,
            targets,
            input_layer,
            hidden_layer,
            output_layer,
            epochs,
            learning_rate,
        )
    )
    end = time.time()
    duration = end - start
    print(
        f"Time to train the multilayer perceptron with {input_layer} input"
        + f"layers, {hidden_layer} hidden layers, and {output_layer} output "
        + f"layer(s) in {epochs} epochs: {duration:.5f} seconds"
    )

    print("\n===============================:")
    print("Test the multilayer perceptron!")
    first = int(input("Enter value for first input(0 or 1): "))
    second = int(input("Enter value for second input(0 or 1): "))

    _, output = forward_propagation(
        [first, second], hidden_weights, hidden_bias, output_weights, output_bias
    )
    print(f"Input: [{first}, {second}], Output: {output}")


main()
