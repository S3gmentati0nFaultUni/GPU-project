#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits.h>
#include <cstdio>
#include "Graph.cuh"
#include "../Common/common.h"

using namespace std;

void Graph::graphConstruction(uint nnn, uint nne, node *nNeighbours, uint *neWeights, uint *nCumDegs)
{
	str->edgeSize = nne;
	memsetGPU(nnn, "edges");
	memcpy(str->neighs, nNeighbours, nne * sizeof(node));
	memcpy(str->weights, neWeights, nne * sizeof(uint));
	memcpy(str->cumDegs, nCumDegs, (nnn + 1) * sizeof(node));
}

Graph::~Graph()
{
	if (GPUEnabled)
	{
		deallocGPU();
	}
	else
	{
		delete str;
	}
}

/**
 * Generate an Erdos random graph
 * @param n number of nodes
 * @param density probability of an edge (expected density)
 * @param eng seed
 */
void Graph::setup(uint nn)
{
	if (GPUEnabled)
	{
		memsetGPU(nn, string("nodes"));
	}
	else
	{
		str = new GraphStruct();
		str->cumDegs = new node[nn + 1]{}; // starts by zero
	}
	str->nodeSize = nn;
}

void Graph::setup(uint nn, uint ne)
{
	if (GPUEnabled)
	{
		memsetGPU(nn, ne);
	}
	else
	{
		str = new GraphStruct();
		str->cumDegs = new node[nn + 1]{}; // starts by zero
		str->neighs = new uint[ne]{};
		str->weights = new int[ne]{};
	}
	str->nodeSize = nn;
	str->edgeSize = ne;
}

/**
 * Generate a new random graph
 * @param eng seed
 */
void Graph::randGraph(float prob, bool weighted, int weight_limit, std::default_random_engine &eng)
{
	if (prob < 0 || prob > 1)
	{
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n");
	}
	if (!weighted && weight_limit != -1)
	{
		printf("[Graph] Error: Is the graph weighted or not?\n");
	}
	if (weight_limit == -1)
	{
		weight_limit = UINT_MAX;
	}

	uniform_real_distribution<> randR(0.0, 1.0);
	node n = str->nodeSize;

	// gen edges
	vector<int> *edges = new vector<int>[n];
	vector<int> *weights = new vector<int>[n];
	for (node i = 0; i < n - 1; i++)
	{
		for (node j = i + 1; j < n; j++)
			if (randR(eng) < prob)
			{
				edges[i].push_back(j);
				edges[j].push_back(i);
				str->cumDegs[i + 1]++;
				str->cumDegs[j + 1]++;
				str->edgeSize += 2;
				if (weighted)
				{
					int weight = (int)(weight_limit * randR(eng));
					weights[i].push_back(weight);
					weights[j].push_back(weight);
				}
			}
	}
	for (node i = 0; i < n; i++)
	{
		str->cumDegs[i + 1] += str->cumDegs[i];
	}

	// max, min, mean deg
	maxDeg = 0;
	minDeg = n;
	for (uint i = 0; i < n; i++)
	{
		if (str->deg(i) > maxDeg)
			maxDeg = str->deg(i);
		if (str->deg(i) < minDeg)
			minDeg = str->deg(i);
	}
	density = (float)str->edgeSize / (float)(n * (n - 1));
	meanDeg = (float)str->edgeSize / (float)n;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	// manage memory for edges with CUDA Unified Memory
	if (GPUEnabled)
	{
		memsetGPU(n, "edges");
	}
	else
	{
		str->neighs = new node[str->edgeSize]{};
		str->weights = new int[str->edgeSize]{};
	}
	for (node i = 0; i < n; i++)
	{
		memcpy((str->neighs + str->cumDegs[i]), edges[i].data(), sizeof(int) * edges[i].size());
		memcpy((str->weights + str->cumDegs[i]), weights[i].data(), sizeof(int) * weights[i].size());
	}

	// Memory deallocation
	delete[] edges;
	delete[] weights;
}

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void Graph::print(bool verbose)
{
	node n = str->nodeSize;
	cout << "** Graph (num node: " << n << ", num edges: " << str->edgeSize
		 << ")" << endl;
	cout << "         (min deg: " << minDeg << ", max deg: " << maxDeg
		 << ", mean deg: " << meanDeg << ", connected: " << connected << ")"
		 << endl;

	if (verbose)
	{
		for (node i = 0; i < n; i++)
		{
			cout << "   node(" << i << ")" << "["
				 << str->deg(i) << "]-> ";
			for (node j = 0; j < str->deg(i); j++)
			{
				cout << str->getNeigh(i, j) << "(" << str->getWeight(i, j) << ") ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}

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
