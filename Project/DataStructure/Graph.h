#ifndef GRAPH_H
#define GRAPH_H

#include <random>
#include <string>

using node = unsigned int; // graph node
using uint = unsigned int;
using sint = short int;
using lint = long int;

/**
 * Base structure (array 1D format) of a graph
 */
struct GraphStruct
{
	uint nodeSize{0};		   // num of graph nodes
	uint edgeSize{0};		   // num of graph edges
	uint *cumDegs{nullptr};	   // cumsum of node degrees
	node *neighs{nullptr};	   // list of neighbors for all nodes (edges)
	int *weights{nullptr};

	~GraphStruct()
	{
		delete[] neighs;
		delete[] cumDegs;
		delete[] weights;
	}

	// check whether node j is a neighbor of node i
	int isNeighbor(node i, node j)
	{
		for (uint k = 0; k < deg(i); k++)
		{
			if (neighs[cumDegs[i] + k] == j)
			{
				return k;
			}
		}
		return -1;
	}

	// finds the weight associated to a certain neighbour
	bool findWeight(node i, node j)
	{
		for (uint k = 0; k < deg(i); k++)
		{
			if (neighs[cumDegs[i] + k] == j)
			{
				return getWeight(i, k);
			}
		}
		return -1;
	}

	// Getter for the weight associated to the j-th neighbour of i
	int getWeight(node i, uint j)
	{
		return weights[cumDegs[i] + j];
	}

	// Getter for the j-th neighbour of i
	node getNeigh(node i, uint j)
	{
		return neighs[cumDegs[i] + j];
	}

	// return the degree of node i
	uint deg(node i)
	{
		return cumDegs[i + 1] - cumDegs[i];
	}

	// Function that saves to file the contents of the current graph
	void graphWriter(char *path)
	{
		FILE *fptr;

		// Open the file in writing mode
		fptr = fopen(path, "w");
		if (!fptr)
		{
			printf("There was an error while opening the file\n");
		}

		// Write down the size of the graph
		fprintf(fptr, "%d\n", nodeSize);

		// Write the adjacency lists
		for (node i = 0; i < nodeSize; ++i)
		{
			for (node j = 0; j < deg(i); j++)
			{
				fprintf(fptr, "%d,%d,", neighs[cumDegs[i] + j], weights[cumDegs[i] + j]);
			}
			fprintf(fptr, "\n");
		}

		// Close the file
		fclose(fptr);
	}
};

/**
 * It manages a graph for CPU & GPU
 */
class Graph
{
	float density{0.0f};	   // Probability of an edge (Erdos graph)
	GraphStruct *str{nullptr}; // graph structure
	uint maxDeg{0};
	uint minDeg{0};
	float meanDeg{0.0f};
	bool connected{true};
	bool GPUEnabled{true};

public:
	Graph(uint nn, uint ne, bool GPUEnb) : GPUEnabled{GPUEnb}
	{
		setup(nn, ne);
	}
	Graph(uint nn, bool GPUEnb) : GPUEnabled{GPUEnb} { setup(nn); }
	void copyConstructor(uint nnn, uint nne, node *nNeighbours, uint *neWeights, uint *nCumDegs)
	{
		if (GPUEnabled)
		{
			deallocGPU();
		}
		else
		{
			delete str;
		}
		setup(nnn);
		graphConstruction(nnn, nne, nNeighbours, neWeights, nCumDegs);
	};

	~Graph();
	void setup(uint);		// CPU/GPU mem setup
	void setup(uint, uint); // CPU/GPU mem setup
	void graphConstruction(uint nnn, uint nne, node *nNeighbours, uint *neWeights, uint *nCumDegs);
	void randGraph(float, bool, int, std::default_random_engine &); // generate an Erdos random graph
	void print(bool);
	void print_d(GraphStruct *, bool);
	GraphStruct *getStruct() { return str; }
	void memsetGPU(uint, std::string); // use UVA memory on CPU/GPU
	void memsetGPU(uint, uint);		   // use UVA memory on CPU/GPU
	void deallocGPU();
	bool isConnected() { return connected; };
};

#endif
