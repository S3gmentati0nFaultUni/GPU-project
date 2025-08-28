#include "DataStructure/ReadGraph.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <fstream>

using namespace std;

struct Subset {
    int parent, rank;
};

// Find the root of the node 'i' with path compression
int find(vector<Subset>& subsets, int i) {
    if (subsets[i].parent != i) {
        subsets[i].parent = find(subsets, subsets[i].parent);
    }
    return subsets[i].parent;
}

// Union of two subsets by rank
void unionSets(vector<Subset>& subsets, int u, int v) {
    int rootU = find(subsets, u);
    int rootV = find(subsets, v);

    if (subsets[rootU].rank < subsets[rootV].rank) {
        subsets[rootU].parent = rootV;
    } else if (subsets[rootU].rank > subsets[rootV].rank) {
        subsets[rootV].parent = rootU;
    } else {
        subsets[rootV].parent = rootU;
        subsets[rootU].rank++;
    }
}

// Borůvka's Algorithm to find MST
unsigned long long int boruvkaMST(int V, vector<Edge> edges) {
    vector<Subset> subsets(V);
    for (int v = 0; v < V; v++) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    unsigned long long int mstWeight = 0;
    int numComponents = V;

    while (numComponents > 1) {
        vector<int> cheapest(V, -1);

        // Find cheapest edge for each component
        for (int i = 0; i < edges.size(); i++) {
            int u = edges[i].src;
            int v = edges[i].dest;
            int weight = edges[i].weight;
            int setU = find(subsets, u);
            int setV = find(subsets, v);

            if (setU != setV) {
                if (cheapest[setU] == -1 || edges[cheapest[setU]].weight > weight) {
                    cheapest[setU] = i;
                }
                if (cheapest[setV] == -1 || edges[cheapest[setV]].weight > weight) {
                    cheapest[setV] = i;
                }
            }
        }

        // Add the selected edges to the MST
        for (int i = 0; i < V; i++) {
            if (cheapest[i] != -1) {
                int u = edges[cheapest[i]].src;
                int v = edges[cheapest[i]].dest;
                int setU = find(subsets, u);
                int setV = find(subsets, v);

                if (setU != setV) {
                    mstWeight += edges[cheapest[i]].weight;
                    unionSets(subsets, setU, setV);
                    numComponents--;
                }
            }
        }
    }

    // Print the MST
    cout << "Minimum Spanning Tree weight: " << mstWeight << endl;
    return mstWeight;
}

int main() {


    const std::vector<std::string> test = { "nw", "cal", "lks", "bay", "ne", "west", "usa", "col", "east", "ctr", "fla", "ny" };
    const std::filesystem::path sandbox{TEST};
    CPUGraph *graph;
    std::size_t i {0};

    for (auto const& dir_entry : std::filesystem::directory_iterator{sandbox}) {
        if (dir_entry.path().extension() != ".txt") {
            continue;
        }
        cout << dir_entry.path() << endl;

        // Generation of the graph
        string path(dir_entry.path());
        printf("Generating graph from file\n");
        graph = rgraphCPU(path);
        double go = seconds();
        unsigned long long int weight = boruvkaMST(graph->nodeSize, graph->edges);
        float CPUtime = seconds() - go;
        cout << "Time elapsed for CPU computation: " << CPUtime << endl;

        path = LOGPATH + test[i] + string("_cpu");

        ofstream logfile(path, ios_base::out);

        if (logfile.is_open()){
            cout << "Writing to file " << path << endl;
            logfile << weight << "\n" << CPUtime << "\n";
            logfile.close();
        }
        else {
            cout << "Unable to open file";
        }

        delete graph;
        ++i;
    }
    

    return 0;
}
