#ifndef READ_GRAPH_H
#define READ_GRAPH_H

#include <string>
#include <iostream>
#include <vector>

#include "Graph.cuh"
#include "../Common/common.h"
#include "../Common/sharedMacros.h"

using namespace std;

struct Edge {
    int src, dest, weight;

    Edge() {
        this->src = 0;
        this->dest = 0;
        this->weight = 0;
    }

    Edge(int src, int dest, int weight) {
        this->src = src;
        this->dest = dest;
        this->weight = weight;
    }

    void print() {
        cout << this->src << " " << this->dest << " " << this->weight << endl;
    }
};



struct CPUGraph {
    int nodeSize, edgeSize;
    vector<Edge> edges;

    CPUGraph(int nodeSize, int edgeSize) {
        this->nodeSize = nodeSize;
        this->edgeSize = edgeSize;
        this->edges = vector<Edge>(edgeSize - 1);
    }

    void print() {
        cout << this->nodeSize << "   " << this->edgeSize << endl;
        for (int i = 0; i < this->edgeSize - 1; i++) {
            this->edges[i].print();
        }
    }
};

Graph *initializeGraph(string line, bool GPUenabled);
void readEdge(string line, vector<uint> *edges, vector<int> *weights, GraphStruct *str);
void readEdgeCPU(string line, vector<Edge> *edges, uint pos);
Graph *rgraph(string path, bool GPUenabled);
CPUGraph *rgraphCPU(string path);

#endif
