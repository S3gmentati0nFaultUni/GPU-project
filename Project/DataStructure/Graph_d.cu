#include "../Common/common.h"
#include "Graph.h"
#include <cstdio>

using namespace std;

void Graph::deallocGPU() {
  CHECK(cudaFree(str->neighs));
  CHECK(cudaFree(str->weights));
  CHECK(cudaFree(str->cumDegs));
  CHECK(cudaFree(str));
}

/**
 * Set the CUDA Unified Memory for nodes and edges
 * @param memType node or edge memory type
 */
void Graph::memsetGPU(uint nn, string memType) {
  if (!memType.compare("nodes")) {
    CHECK(cudaMallocManaged(&str, sizeof(GraphStruct)));
    CHECK(cudaMallocManaged(&(str->cumDegs), (nn + 1) * sizeof(node)));
  } else if (!memType.compare("edges")) {
    CHECK(cudaMallocManaged(&(str->neighs), str->edgeSize * sizeof(node)));
    CHECK(cudaMallocManaged(&(str->weights), str->edgeSize * sizeof(int)));
  }
}

void Graph::memsetGPU(uint nn, uint ne) {
  CHECK(cudaMallocManaged(&str, sizeof(GraphStruct)));
  CHECK(cudaMallocManaged(&(str->cumDegs), (nn + 1) * sizeof(node)));
  CHECK(cudaMallocManaged(&(str->neighs), ne * sizeof(node)));
  CHECK(cudaMallocManaged(&(str->weights), ne * sizeof(int)));
}

/**
 * Print the graph on device (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
__global__ void print_d(GraphStruct *str, bool verbose) {
  printf("** Graph (num node: %d, num edges: %d)\n", str->nodeSize,
         str->edgeSize);

  if (verbose) {
    for (int i = 0; i < str->nodeSize; i++) {
      printf("  node(%d)[%d]-> ", i, str->cumDegs[i + 1] - str->cumDegs[i]);
      for (int j = 0; j < str->cumDegs[i + 1] - str->cumDegs[i]; j++) {
        printf("%d(%d) ", str->neighs[str->cumDegs[i] + j],
               str->weights[str->cumDegs[i] + j]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

__global__ void hello_d() { printf("Helloworld\n"); }
