/*
 * Iterative Version of Neural Network
 * Hari Shankar
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct Node {
    char* id;

    double bias;
    double z;
    double activation;

    double dBias;
    double dZ;
    double dActivation;

    int toSize;
    struct Edge** to;

    int fromSize;
    struct Edge** from;
};

struct Edge {
    char id;
    double weight;
    double dWeight;
    struct Node *start, *end;
};

//make the activation function of a layer part of its property with function pointer.
struct Layer {
    struct Node** nodes;
    int size;
    double (*activationFn)(double, bool);
};

char* createId(int counter) {
    int length = 4;
    char* id = (char*) calloc(length + 1, sizeof(char));
    for (int i = 0; i < length; i++) {
        int placeValue = pow(26, (length - i - 1));
        id[i] = 'a' + (counter / placeValue);
        counter = counter % placeValue;
    }
    printf("%s\n", id);
    return id;
}

void initializeLayer(struct Node** layer, int size, double (*activationFn)(double, bool), const double initialBiasVal[size], int* cp) {
    for (int i = 0; i < size; i++) {
        layer[i] = (struct Node*) calloc(1, sizeof(struct Node));
        layer[i]->id = createId(*cp);
        layer[i]->bias = initialBiasVal[i];
        (*cp)++;
    }
}

void connectNodes(struct Node* start, struct Node* end, double initialWeightVal) {
    struct Edge* newEdge = malloc(sizeof(struct Edge));
    newEdge->weight = initialWeightVal;
    newEdge->start = start;
    newEdge->end = end;

    (start->toSize)++;
    start->to = realloc(start->to, start->toSize * sizeof(struct Edge*));
    start->to[start->toSize - 1] = newEdge;

    (end->fromSize)++;
    end->from = realloc(end->from, end->fromSize * sizeof(struct Edge*));
    end->from[end->fromSize - 1] = newEdge;
}

void fullyConnectLayers(struct Node** layer1, int layer1Size, struct Node** layer2, int layer2Size, double initialWeightVals[layer1Size][layer2Size]) {
    for (int i = 0; i < layer1Size; i++) {
        for (int j = 0; j < layer2Size; j++) {
            connectNodes(layer1[i], layer2[j], initialWeightVals[i][j]);
        }
    }
}

void printLayer(struct Node** layer, int size) {
    for (int i = 0; i < size; i++) {
        printf("Node %s: Value: %F, Travels to %d nodes [",
               layer[i]->id, layer[i]->activation, layer[i]->toSize);
        for (int j = 0; j < layer[i]->toSize; j++) {
            printf("%s, ", layer[i]->to[j]->end->id);
        }
        printf("], Destination from %d nodes [", layer[i]->fromSize);
        for (int j = 0; j < layer[i]->fromSize; j++) {
            printf("%s, ", layer[i]->from[j]->start->id);
        }
        printf("].\n");

    }
}

void printEdge(struct Edge edge) {
    printf("Edge travelling from node %s to node %s. Weight: %f, dWeight: %f\n",
           edge.start->id, edge.end->id, edge.weight, edge.dWeight);
}

void printNode(struct Node node) {
    printf("Node %s: Activation: %f, Z: %f, Bias: %f, dActivation: %f, dZ: %f, dBias: %f. Connected to edges:\n",
           node.id, node.activation, node.z, node.bias, node.dActivation, node.dZ, node.dBias);
    for (int i = 0; i < node.toSize; i++) {
        printEdge(*(node.to[i]));
    }
}

void printEverything(struct Layer** layers) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < layers[i]->size; j++) {
            printNode(*layers[i]->nodes[j]);
        }
    }
    for (int j = 0; j < layers[2]->size; j++) {
        printNode(*layers[2]->nodes[j]);
    }
}


void loadData(const double* inputData, int inputDataSize, struct Node** inputLayer) {
    for (int i = 0; i < inputDataSize; i++) {
        inputLayer[i]->activation = inputData[i];
    }
}

void step(struct Layer* layer) {
    //summation of weight * activation
    for (int i = 0; i < layer->size; i++) {
        struct Node* currentNode = layer->nodes[i];
        for (int j = 0; j < currentNode->fromSize; j++) {
            printf("i: %d j: %d | currentNode Value: %f | currentNode Edge Weight: %f | target value currently at : %f\n", i, j,  currentNode->activation, currentNode->from[j]->weight, currentNode->from[j]->start->activation);
            currentNode->z += currentNode->from[j]->start->activation * currentNode->from[j]->weight;
        }
    }

    //add bias
    for (int i = 0; i < layer->size; i++) {
        layer->nodes[i]->z += layer->nodes[i]->bias;
    }
}


double reluActivationFunction(double val, bool derive) {
    if (derive) {
        return val > 0;
    }
    return val > 0 ? val : 0;
}

double sigmoidActivationFunction(double val, bool derive) {
    if (derive) {
        return (exp(-val) / (pow(1 + exp(-val), 2)));
    }
    return 1 / (1 + exp(-1 * val));
}

void applyActivation(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        layer->nodes[i]->activation = layer->activationFn(layer->nodes[i]->z, false);
    }
}

//precondition: layers>=2
void forwardPass(double* inputData, int inputDataSize, struct Layer** layers, int numLayers) {
    loadData(inputData, inputDataSize, layers[0]->nodes);
    for (int i = 1; i < numLayers; i++) {
        step(layers[i]);
        applyActivation(layers[i]);
    }
}

double calculateError(struct Layer* output, double answer[]) {
    double sum = 0;
    for (int i = 0; i < output->size; i++) {
        sum += pow((output->nodes[i]->activation - answer[i]), 2);
    }
    return sum / output->size;
}

void deriveLastLayerActivations(struct Layer* output, const double answer[]) {
    for (int i = 0; i < output->size; i++) {
        output->nodes[i]->dActivation = 2 * (output->nodes[i]->activation - answer[i]);
    }
}

void deriveWeights(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        for (int j = 0; j < layer->nodes[i]->fromSize; j++) {
            layer->nodes[i]->from[j]->dWeight = layer->nodes[i]->from[j]->start->activation * layer->nodes[i]->dZ * layer->nodes[i]->dActivation;
        }
    }
}

void deriveActivations(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
         for (int j = 0; j < layer->nodes[i]->toSize; j++) {
             layer->nodes[i]->dActivation += layer->nodes[i]->to[j]->weight * layer->nodes[i]->to[j]->end->dZ * layer->nodes[i]->to[j]->end->dActivation;
         }
    }
}

void deriveZ(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        layer->nodes[i]->activation = (*(layer->activationFn))(layer->nodes[i]->z, true);
    }
}

void deriveBias(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        layer->nodes[i]->bias = layer->nodes[i]->dZ * layer->nodes[i]->dActivation;
    }
}

void resetLayerNodes(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        layer->nodes[i]->activation = 0;
        layer->nodes[i]->z = 0;

        layer->nodes[i]->dActivation = 0;
        layer->nodes[i]->dZ = 0;
        layer->nodes[i]->dBias = 0;
    }
}

void resetLayerEdges(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        for (int j = 0; i < layer->nodes[i]->toSize; j++) {
            layer->nodes[i]->to[j]->dWeight = 0;
        }
    }
}

void resetNetwork(struct Layer** layers, int numLayers) {
    for (int i = 0; i < numLayers - 1; i++) {
        resetLayerNodes(layers[i]);
        resetLayerEdges(layers[i]);
    }
    resetLayerNodes(layers[numLayers - 1]);
}

void calculate_gradient(struct Layer** layers, int numLayers, double answer[]) {

    deriveLastLayerActivations(layers[numLayers - 1], answer);
    deriveZ(layers[numLayers - 1]);
    deriveBias(layers[numLayers - 1]);
    deriveWeights(layers[numLayers - 1]);

    for (int i = numLayers - 2; i >= 0; i--) {
        deriveActivations(layers[1]);
        deriveZ(layers[1]);
        deriveBias(layers[1]);
        deriveWeights(layers[1]);
    }
}

void applyGradient(struct Layer** layers, int numLayers, double answer[], int alpha) {
    for (int i = 0; i < numLayers; i++) {
        for (int j = 0; j < layers[i]->size; j++) {
            layers[i]->nodes[j]->bias -= layers[i]->nodes[j]->dBias * alpha;
        }
    }
}

int main() {

    int counter = 0;
    int* cp = &counter;

    int inputLayerSize = 3;
    double initialBiasVals[] = {.1, .2, .3, .4};

    struct Node* inputs[inputLayerSize];
    initializeLayer(inputs, inputLayerSize, NULL, initialBiasVals, cp);

    int hiddenLayerSize = 4;
    struct Node* hidden[hiddenLayerSize];
    initializeLayer(hidden, hiddenLayerSize, reluActivationFunction, initialBiasVals, cp);

    int outputLayerSize = 2;
    struct Node* outputs[outputLayerSize];
    initializeLayer(outputs, outputLayerSize, sigmoidActivationFunction, initialBiasVals, cp);



    fullyConnectLayers(inputs, inputLayerSize, hidden, hiddenLayerSize, );
    fullyConnectLayers(hidden, hiddenLayerSize, outputs, outputLayerSize);

    struct Layer inputLayer;
    inputLayer.nodes = inputs;
    inputLayer.size = inputLayerSize;
    inputLayer.activationFn = NULL;

    struct Layer hiddenLayer;
    hiddenLayer.nodes = hidden;
    hiddenLayer.size = hiddenLayerSize;
    hiddenLayer.activationFn = reluActivationFunction;

    struct Layer outputLayer;
    outputLayer.nodes = outputs;
    outputLayer.size = outputLayerSize;
    outputLayer.activationFn = sigmoidActivationFunction;

    struct Layer* allLayers[3];
    allLayers[0] = &inputLayer;
    allLayers[1] = &hiddenLayer;
    allLayers[2] = &outputLayer;

    double data[3] = {.1, .2, .3};
    forwardPass(data, 3, allLayers, 3);

    double ans[2] = {1.0, 2.0};
    double error = calculateError(allLayers[2], ans);

    printf("Input Layer\n");
    printLayer(inputs, inputLayerSize);
    printf("\nHidden Layer\n");
    printLayer(hidden, hiddenLayerSize);
    printf("\nOutput Layer\n");
    printLayer(outputs, outputLayerSize);

    printf("\nMean-Squared Error: %f\n", error);

    calculate_gradient(allLayers, 3, ans);

    printEverything(allLayers);

    resetNetwork(allLayers, 3);

    return 0;
}
