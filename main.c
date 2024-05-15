/*
 * Iterative Version of Neural Network
 * Hari Shankar
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct Node {
    char* id;
    double value;
    int toSize;
    struct Edge** to;
    int fromSize;
    struct Edge** from;
    double bias;
};

struct Edge {
    char id;
    double weight;
    struct Node *start, *end;
};

struct Layer {
    struct Node** nodes;
    int size;
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
void initializeLayer(struct Node** layer, int size, int* cp) {
    for (int i = 0; i < size; i++) {
        layer[i] = (struct Node*) malloc(sizeof(struct Node));
        layer[i]->value = 0;
        layer[i]->toSize = 0;
        layer[i]->fromSize = 0;
        layer[i]->to = malloc(0);
        layer[i]->from = malloc(0);
        layer[i]->id = createId(*cp);
        layer[i]->bias = .1;
        (*cp)++;
    }
}

void connectNodes(struct Node* start, struct Node* end) {
    struct Edge* newEdge = malloc(sizeof(struct Edge));
    newEdge->weight = 1;
    newEdge->start = start;
    newEdge->end = end;

    (start->toSize)++;
    start->to = realloc(start->to, start->toSize * sizeof(struct Edge*));
    start->to[start->toSize - 1] = newEdge;

    (end->fromSize)++;
    end->from = realloc(end->from, end->fromSize * sizeof(struct Edge*));
    end->from[end->fromSize - 1] = newEdge;
}

void fullyConnectLayers(struct Node** layer1, int layer1Size, struct Node** layer2, int layer2Size) {
    for (int i = 0; i < layer1Size; i++) {
        for (int j = 0; j < layer2Size; j++) {
            connectNodes(layer1[i], layer2[j]);
        }
    }
}

void printLayer(struct Node** layer, int size) {
    for (int i = 0; i < size; i++) {
        printf("Node %s: Value: %F, Travels to %d nodes [",
               layer[i]->id, layer[i]->value, layer[i]->toSize);
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

void loadData(double* inputData, int inputDataSize, struct Node** inputLayer) {
    for (int i = 0; i < inputDataSize; i++) {
        inputLayer[i]->value = inputData[i];
    }
}

void step(struct Node** layer, int size) {
    for (int i = 0; i < size; i++) {
        struct Node* currentNode = layer[i];
        for (int j = 0; j < currentNode->toSize; j++) {
            printf("i: %d j: %d | currentNode Value: %f | currentNode Edge Weight: %f | target value currently at : %f\n", i, j,  currentNode->value, currentNode->to[j]->weight, currentNode->to[j]->end->value);
            currentNode->to[j]->end->value += currentNode->value * currentNode->to[j]->weight;
        }
    }
}

void reluActivationFunction(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        double val = layer->nodes[i]->value;
        if (val < 0) {
            val = 0;
        }
        layer->nodes[i]->value =  val;
    }
}

void sigmoidActivationFunciton(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        double val = layer->nodes[i]->value;
        val = 1 / (1 + exp(-1 * val));
        layer->nodes[i]->value =  val;
    }
}

void useBias(struct Layer* layer) {
    for (int i = 0; i < layer->size; i++) {
        layer->nodes[i]->value -= layer->nodes[i]->bias;
    }
}

void forwardPass(double* inputData, int inputDataSize, struct Layer** layers) {
    loadData(inputData, inputDataSize, layers[0]->nodes);

    step(layers[0]->nodes, layers[0]->size);
    useBias(layers[1]);
    reluActivationFunction(layers[1]);

    step(layers[1]->nodes, layers[1]->size);
    useBias(layers[2]);
    sigmoidActivationFunciton(layers[2]);
}

double calculateError(struct Layer* output, double answer[]) {
    double sum = 0;
    for (int i = 0; i < output->size; i++) {
        sum += pow((output->nodes[i]->value - answer[i]), 2);
    }
    return sum / output->size;
}


int main() {

    int counter = 0;
    int* cp = &counter;

    int inputLayerSize = 3;

    struct Node* inputs[inputLayerSize];
    initializeLayer(inputs, inputLayerSize, cp);

    int hiddenLayerSize = 4;
    struct Node* hidden[hiddenLayerSize];
    initializeLayer(hidden, hiddenLayerSize, cp);

    int outputLayerSize = 2;
    struct Node* outputs[outputLayerSize];
    initializeLayer(outputs, outputLayerSize, cp);

    fullyConnectLayers(inputs, inputLayerSize, hidden, hiddenLayerSize);
    fullyConnectLayers(hidden, hiddenLayerSize, outputs, outputLayerSize);

    struct Layer inputLayer;
    inputLayer.nodes = inputs;
    inputLayer.size = inputLayerSize;

    struct Layer hiddenLayer;
    hiddenLayer.nodes = hidden;
    hiddenLayer.size = hiddenLayerSize;

    struct Layer outputLayer;
    outputLayer.nodes = outputs;
    outputLayer.size = outputLayerSize;

    struct Layer* allLayers[3];
    allLayers[0] = &inputLayer;
    allLayers[1] = &hiddenLayer;
    allLayers[2] = &outputLayer;

    double data[3] = {0.1, .2, .3};
    forwardPass(data, 3, allLayers);

    double ans[2] = {1.0, 2.0};
    double error = calculateError(allLayers[2], ans);

    printf("Input Layer\n");
    printLayer(inputs, inputLayerSize);
    printf("\nHidden Layer\n");
    printLayer(hidden, hiddenLayerSize);
    printf("\nOutput Layer\n");
    printLayer(outputs, outputLayerSize);

    printf("Mean-Squared Error: %f", error);

    return 0;
}
