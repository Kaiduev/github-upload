# Step 1 : initialize the network
from random import seed
from random import random


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()

    # get 2 random numbers for inputs
    input_layer = [{'inputs': [random() for i in range(n_inputs)]}]
    network.append(input_layer)

    # get 8 random numbers for weights between the input layer and the hidden layer
    input_layer = [{'weights_Hidden': [random() for i in range(n_hidden)]} for i in range(n_inputs)]
    network.append(input_layer)

    # get 4 random numbers for neuron outputs in the hidden layer
    hidden_layer = [{'NeuronOutput': [random() for i in range(n_hidden)]}]  #
    network.append(hidden_layer)

    # get 8 random numbers for weight between the hidden layer and the output layer
    hidden_layer = [{'weights_Output': [random() for i in range(n_outputs)]} for i in range(n_hidden)]
    network.append(hidden_layer)

    # get 2 random numbers for current output
    output_layer = [{'CurrentOutput': [random() for i in range(n_outputs)]}]
    network.append(output_layer)

    # get 2 random numbers for target output
    output_layer = [{'TargetOutput': [random() for i in range(n_outputs)]}]
    network.append(output_layer)

    return network


seed(1)
network = initialize_network(2, 4, 2)
for layer in network:
    print(layer)


def CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate):
    def CalError1layer(CurrentOutput, TargetOutput):
        ErrorX = CurrentOutput[0] * (1 - CurrentOutput[0]) * (TargetOutput[0] - CurrentOutput[0])
        return ErrorX

    def CalError2layer(CurrentOutput, TargetOutput):
        ErrorY = CurrentOutput[1] * (1 - CurrentOutput[1]) * (TargetOutput[1] - CurrentOutput[1])
        return ErrorY

    ErrorX = CalError1layer(CurrentOutput, TargetOutput)
    ErrorY = CalError2layer(CurrentOutput, TargetOutput)

    # 2. Change the weight of hidden layers

    # calculate the update weight between the hidden layer and output layer
    Updated_Weight9 = CurrentWeights[8] + LearningRate * ErrorX * NeuronOutput[3]
    Updated_Weight10 = CurrentWeights[9] + LearningRate * ErrorY * NeuronOutput[3]
    Updated_Weight11 = CurrentWeights[10] + LearningRate * ErrorX * NeuronOutput[4]
    Updated_Weight12 = CurrentWeights[11] + LearningRate * ErrorY * NeuronOutput[4]
    Updated_Weight13 = CurrentWeights[12] + LearningRate * ErrorX * NeuronOutput[5]
    Updated_Weight14 = CurrentWeights[13] + LearningRate * ErrorY * NeuronOutput[5]
    Updated_Weight15 = CurrentWeights[14] + LearningRate * ErrorX * NeuronOutput[6]
    Updated_Weight16 = CurrentWeights[15] + LearningRate * ErrorY * NeuronOutput[6]

    # 3. Calculate the Errors for the hidden layer neurons
    # Errors from the output neurons should be taken and ran back through the weights to get the hidden layer errors.
    # It is because direct calculation is not possible cuz we don't have a Target.

    # calulate the errors for neurons in hidden layer
    Error_HiddenM = NeuronOutput[2] * (1 - NeuronOutput[2]) * (ErrorX * CurrentWeights[8] + ErrorY * CurrentWeights[9])
    Error_HiddenN = NeuronOutput[3] * (1 - NeuronOutput[3]) * (
                ErrorX * CurrentWeights[10] + ErrorY * CurrentWeights[11])
    Error_HiddenO = NeuronOutput[4] * (1 - NeuronOutput[4]) * (
                ErrorX * CurrentWeights[12] + ErrorY * CurrentWeights[13])
    Error_HiddenP = NeuronOutput[5] * (1 - NeuronOutput[5]) * (
                ErrorX * CurrentWeights[14] + ErrorY * CurrentWeights[15])

    ##4.Change the weights based on Errors of M, N, O, P
    Updated_Weight1 = CurrentWeights[0] + LearningRate * Error_HiddenM * NeuronOutput[0]
    Updated_Weight2 = CurrentWeights[1] + LearningRate * Error_HiddenN * NeuronOutput[0]
    Updated_Weight3 = CurrentWeights[2] + LearningRate * Error_HiddenO * NeuronOutput[0]
    Updated_Weight4 = CurrentWeights[3] + LearningRate * Error_HiddenP * NeuronOutput[0]

    Updated_Weight5 = CurrentWeights[4] + LearningRate * Error_HiddenM * NeuronOutput[1]
    Updated_Weight6 = CurrentWeights[5] + LearningRate * Error_HiddenN * NeuronOutput[1]
    Updated_Weight7 = CurrentWeights[6] + LearningRate * Error_HiddenO * NeuronOutput[1]
    Updated_Weight8 = CurrentWeights[7] + LearningRate * Error_HiddenP * NeuronOutput[1]

    UpdateWeights = [Updated_Weight1, Updated_Weight2, Updated_Weight3, Updated_Weight4, Updated_Weight5,
                     Updated_Weight6, Updated_Weight7, Updated_Weight8,
                     Updated_Weight9, Updated_Weight10, Updated_Weight11, Updated_Weight12, Updated_Weight13,
                     Updated_Weight14, Updated_Weight15, Updated_Weight16]
    return UpdateWeights

    # UpdateWeights = CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate)
    # print('OldWeights =', CurrentWeights)
    # print('UpdateWeights =', UpdateWeights)
    # print('Total number of weights =', len(UpdateWeights))


LearningRate = 0.5  # given
CurrentOutput = [0.9391491627785106, 0.38120423768821243]  # import from step 1
TargetOutput = [0.21659939713061338, 0.4221165755827173]  # import from step 1
# import from step 1
NeuronOutput = [0.13436424411240122, 0.8474337369372327, 0.8357651039198697, 0.43276706790505337, 0.762280082457942,
                0.0021060533511106927]
# import from step 1
CurrentWeights = [0.763774618976614, 0.2550690257394217, 0.49543508709194095, 0.4494910647887381, 0.651592972722763,
                  0.7887233511355132, 0.0938595867742349, 0.02834747652200631, 0.4453871940548014, 0.7215400323407826,
                  0.22876222127045265, 0.9452706955539223, 0.9014274576114836, 0.030589983033553536, 0.0254458609934608,
                  0.5414124727934966]

UpdateWeights = CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate)
print('OldWeights =', CurrentWeights)
print('UpdateWeights =', UpdateWeights)
print('Total number of weights =', len(UpdateWeights))