from random import seed
from random import random

def initialize_network():
    network = list()

    # get 2 random numbers for inputs
    input_layer = [{'inputs': [random() for i in range(2)]}]
    network.append(input_layer)

    # get 8 random numbers for weights between the input layer and the hidden layer
    input_layer = [{'weights_Hidden': [random() for i in range(4)]} for i in range(2)]
    network.append(input_layer)

    # get 4 random numbers for neuron outputs in the hidden layer
    hidden_layer = [{'NeuronOutput': [random() for i in range(4)]}]  #
    network.append(hidden_layer)

    # get 8 random numbers for weight between the hidden layer and the output layer
    hidden_layer = [{'weights_Output': [random() for i in range(2)]} for i in range(4)]
    network.append(hidden_layer)

    # get 2 random numbers for current output
    output_layer = [{'CurrentOutput': [random() for i in range(2)]}]
    network.append(output_layer)

    # get 2 random numbers for target output
    output_layer = [{'TargetOutput': [random() for i in range(2)]}]
    network.append(output_layer)
    return network


seed(1)
network = initialize_network()
for layer in network:
    print(layer)


# Update the weights by back propagation
def CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate):

    def CalErrorsLayer(CurrentOutput, TargetOutput):
        errors = list()
        i = 0
        while i<len(CurrentOutput):
            errors.append(CurrentOutput[i] * (1 - (CurrentOutput[i]) * (TargetOutput[i] - CurrentOutput[i])))
            i += 1
        return errors

    def UpdateWeightsOutputs(CurrentWeights, LearningRate, Errors, CurrentOutput):
        uptated_weights_outputs = list()
        i, j = 8, 0
        while i<=15:
            uptated_weights_outputs.append(CurrentWeights[i] + LearningRate * Errors[j] * CurrentOutput[j])
            i+=1
            j+=1
            if j>1:
                j=0
        return uptated_weights_outputs

    def ErrorsHiddens(NeuronOutput, CurrentWeights, Errors):
        errors_hiddens = list()
        i, j, k, m = 0, 8, 0, 2
        while i<4:
            if i == 3:
                i = 4
            error_hidden = NeuronOutput[i] * (1 - NeuronOutput[m]) * (
                        Errors[k] * CurrentWeights[j] + Errors[k + 1] * CurrentWeights[i + 1])
            errors_hiddens.append(error_hidden)
            i += 1
            m += 1
            j += 2
        return errors_hiddens

    def UpdateweightsHiddens(CurrentWeights, LearningRate, ErrorsHiddens, NeuronOutput):
        UpdateWeightsHiddens = list()
        i, k, j = 0, 0, 0
        while i<8:
            update_weights_hidden = CurrentWeights[i] + LearningRate * ErrorsHiddens[k] * NeuronOutput[j]
            UpdateWeightsHiddens.append(update_weights_hidden)
            if j > 1:
                j = 0
            if i % 2 != 0:
                k += 1
            i += 1
        return UpdateWeightsHiddens

    Errors = CalErrorsLayer(CurrentOutput, TargetOutput)
    UpdateWeights = UpdateWeightsOutputs(CurrentWeights, LearningRate, Errors, CurrentOutput)
    errors_hiddens = ErrorsHiddens(NeuronOutput, CurrentWeights, Errors)
    UpdateWeightsHiddens = UpdateweightsHiddens(CurrentWeights, LearningRate, errors_hiddens, NeuronOutput)
    UpdatedWeights = UpdateWeightsHiddens + UpdateWeights
    return UpdatedWeights


LearningRate = 0.5
CurrentOutput = [random() for i in range(2)]
TargetOutput = [random() for i in range(2)]
NeuronOutput = [random() for i in range(6)]
CurrentWeights = [random() for i in range(16)]
UpdateWeights = CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate)
print("OldWeights: ", CurrentWeights)
print("UpdateWeights: ", UpdateWeights)
print("UpdateWeights: ", UpdateWeights)
print("Total number of weights: ", len(UpdateWeights))

# LearningRate = 0.5  # given
# CurrentOutput = [0.9391491627785106, 0.38120423768821243]  # import from step 1
# TargetOutput = [0.21659939713061338, 0.4221165755827173]  # import from step 1
# # import from step 1
# NeuronOutput = [0.13436424411240122, 0.8474337369372327, 0.8357651039198697, 0.43276706790505337, 0.762280082457942,
#                 0.0021060533511106927]
# # import from step 1
# CurrentWeights = [0.763774618976614, 0.2550690257394217, 0.49543508709194095, 0.4494910647887381, 0.651592972722763,
#                   0.7887233511355132, 0.0938595867742349, 0.02834747652200631, 0.4453871940548014, 0.7215400323407826,
#                   0.22876222127045265, 0.9452706955539223, 0.9014274576114836, 0.030589983033553536, 0.0254458609934608,
#                   0.5414124727934966]
