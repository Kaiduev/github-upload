from random import random

def CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate):

    # calulate the errors for neurons in output layers
    def CalErrorsLayer(CurrentOutput, TargetOutput):
        errors = list()
        i = 0
        while i<len(CurrentOutput):
            errors.append(CurrentOutput[i] * (1 - (CurrentOutput[i]) * (TargetOutput[i] - CurrentOutput[i])))
            i += 1
        return errors

    # calulate the update weight between the hidden layer and output layer
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

    # calulate the errors for neurons in hidden layer
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

    # calulate the update weight between the input layer and hidden layer
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
    ErrorsHiddens = ErrorsHiddens(NeuronOutput, CurrentWeights, Errors)
    UpdateWeightsHiddens = UpdateweightsHiddens(CurrentWeights, LearningRate, ErrorsHiddens, NeuronOutput)
    UpdatedWeights = UpdateWeightsHiddens + UpdateWeights
    return UpdatedWeights


LearningRate = 0.5 # given
CurrentOutput = [random() for i in range(2)] # initialize with the  random function
TargetOutput = [random() for i in range(2)]
NeuronOutput = [random() for i in range(6)]
CurrentWeights = [random() for i in range(16)]
UpdateWeights = CustomBPweight(TargetOutput, CurrentOutput, NeuronOutput, CurrentWeights, LearningRate)
print("OldWeights: ", CurrentWeights)
print("UpdateWeights: ", UpdateWeights)
print("Total number of weights: ", len(UpdateWeights))
