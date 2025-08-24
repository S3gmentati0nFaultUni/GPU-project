#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

#include "../MST_hk.cuh"
#include "Graph.cuh"
#include "../Common/common.h"
#include "../Common/sharedMacros.h"
#include "ReadGraph.h"

using namespace std;

Graph *initializeGraph(string line, bool GPUenabled) {
    stringstream ss(line);
    string element;
    uint nodeSize;
    uint edgeSize;

    ss >> element;
    nodeSize = stoi(element);
    ss >> element;
    edgeSize = stoi(element);

    Graph *graph = new Graph(nodeSize, edgeSize, GPUenabled);
    return graph;
}

void readEdge(string line, vector<uint> *edges, vector<int> *weights, GraphStruct *str) {
    stringstream ss(line);
    string element;
    node source, destination;
    int weight;

    ss >> element;
    source = stoi(element) - 1;
    ss >> element;
    destination = stoi(element) - 1;
    ss >> element;
    weight = stoi(element);

    str->cumDegs[source + 1]++;

    edges[source].push_back(destination);
    weights[source].push_back(weight);
}

void readEdgeCPU(string line, CPUGraph *graph, uint pos) {
    stringstream ss(line);
    string element;
    node source, destination;
    int weight;

    ss >> element;
    source = stoi(element) - 1;
    ss >> element;
    destination = stoi(element) - 1;
    ss >> element;
    weight = stoi(element);

    graph->edges[pos] = Edge(source, destination, weight);
}

Graph *rgraph(string path, bool GPUenabled) {
    ifstream inFile;
    string line;
    Graph *graph;

    inFile.open(path);

    if (!inFile) {
        cout << "Unable to open file";
        exit(1);
    }

    getline(inFile, line);
    graph = initializeGraph(line, GPUenabled);
    GraphStruct *str = graph->getStruct();
    uint nodeSize = str->nodeSize;
    uint edgeSize = str->edgeSize;
    printf("%d\t%d\n", nodeSize, edgeSize);
    vector<uint> *edges = new vector<uint>[edgeSize];
	  vector<int> *weights = new vector<int>[edgeSize];

    while (getline(inFile, line)) {
        readEdge(line, edges, weights, str);
    }

    for (node i = 0; i < nodeSize; ++i) {
        str->cumDegs[i + 1] += str->cumDegs[i];
    }

    for (node i = 0; i < nodeSize; ++i) {
        memcpy((str->neighs + str->cumDegs[i]), edges[i].data(), sizeof(uint) * edges[i].size());
		    memcpy((str->weights + str->cumDegs[i]), weights[i].data(), sizeof(int) * weights[i].size());
    }

    printf("Closing the file and freeing memory\n");

    inFile.close();
    delete[] edges;
    edges = NULL;
    delete[] weights;
    weights = NULL;

/*
    graph->print(true);
    print_d <<<1, 1>>> (str, 1);
    CHECK(cudaDeviceSynchronize());
*/
    return graph;
}

CPUGraph *rgraphCPU(string path) {
    ifstream inFile;
    string line;

    inFile.open(path);

    if (!inFile) {
        cout << "Unable to open file";
        exit(1);
    }

    getline(inFile, line);
    stringstream ss(line);
    string element;
    uint nodeSize;
    uint edgeSize;

    ss >> element;
    nodeSize = stoi(element);
    ss >> element;
    edgeSize = stoi(element);
    CPUGraph *graph = new CPUGraph(nodeSize, edgeSize);

    uint i = 0;
    while (getline(inFile, line)) {
        readEdgeCPU(line, graph, i);
        i++;
    }

    printf("Closing the file and freeing memory\n");

    inFile.close();

    return graph;
}