// Header file di C++
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <filesystem>

// Header file C
#include <time.h>
#include <limits.h>

// Custom files
#include "DataStructure/Graph_d.h"
#include "DataStructure/Graph.h"
#include "Common/common.h"
#include "Common/sharedMacros.h"
#include "DataStructure/ReadGraph.h"

using namespace std;

/*****
* Device function that gets the degree of a certain node
* @param str - The structure of the graph
* @param i - The node we are interested in
*****/
__device__ node d_deg (GraphStruct *str, node i) {
    return str->cumDegs[i + 1] - str->cumDegs[i];
}

/*****
* Device function that gets the weight of a certain edge
* @param str - The structure of the graph
* @param i - The source node of the edge
* @param offset - The offset of the destination node in the adjacency list of
*                 the source
*****/
__device__ int d_getWeight (GraphStruct *str, node i, uint offset) {
    return str->weights[str->cumDegs[i] + offset];
}

/*****
* Device function that gets the neighbour of a certain node
* @param str - The structure of the graph
* @param i - The source node of the edge
* @param offset - The offset of the destination node in the adjacency list of
*                 the source
*****/
__device__ node d_getNeigh (GraphStruct *str, node i, uint offset) {
    return str->neighs[str->cumDegs[i] + offset];
}

__device__ uint d_getRoot (uint i, uint *d_flag, uint *d_colors) {
    return max(0, d_flag[d_colors[i]]);
}

uint getRoot (uint i, uint *flag, uint *colors) {
    return max(0, flag[colors[i]]);
}


/*****
* Kernel that finds the cheapest edge in the adjacency list of every node
* @param str - The structure of the graph
* @param d_candidates - The device-level array of candidates to become part of
*                       the spanning tree (edges saved as offsets in the CSR
*                       representation of the graph)
*****/
__global__ void findCheapest (GraphStruct *str, uint *d_candidates) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is out of bounds returns immediately
    if (idx >= str->nodeSize) {
        return;
    }

    // Initialize the minimum value
    uint minimum = UINT_MAX;
    int minimumWeight = INT_MAX;

    // Find the cheapest edge in each adjacency list
    for (uint i = 0; i < d_deg(str, idx); i++) {
        int edgeWeight = d_getWeight(str, idx, i);
        if (edgeWeight < minimumWeight) {
            minimumWeight = edgeWeight;
            minimum = i;
        }
        else if (edgeWeight == minimumWeight &&
                 d_getNeigh(str, idx, i) < d_getNeigh(str, idx, minimum)) {
            minimumWeight = edgeWeight;
            minimum = i;
        }
    }

    // Update the return vector
    d_candidates[idx] = minimum;
}


/*****
* Kernel that removes the mirrored edges from the graph. A mirrored edge is
* simply an edge pointing from the source to the destination and vice versa in
* an oriented graph, the removal logic is to cut the edge with the lowest source
* @param str - The structure of the graph
* @param d_candidates - The device-level array of candidates to become part of
*                       the spanning tree (edges saved as offsets in the CSR
*                       representation of the graph)
*****/
__global__ void mirroredEdgesRemoval (GraphStruct *str, uint *d_candidates, unsigned long long int *d_weight) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is out of bounds returns immediately
    if (idx >= str->nodeSize) {
        return;
    }

    uint destinationOffset = d_candidates[idx];
    node destination = d_getNeigh(str, idx, destinationOffset);
    if (idx < destination) {
        uint sourceOffset = d_candidates[destination];
        node destinationNeigh = d_getNeigh(str, destination, sourceOffset);

        // The vertex cannot be a candidate anymore because it would create a cycle
        if (destinationNeigh == idx) {
            d_candidates[idx] = UINT_MAX;
        }
    }

    if (d_candidates[idx] != UINT_MAX) {
        atomicAdd(d_weight, d_getWeight(str, idx, d_candidates[idx]));
    }
}


/*****
* Helper device function that recursively colors the nodes of the graph
* @param str - The structure of the graph
* @param d_candidates - The device-level array of candidates to become part of
*                       the spanning tree (edges saved as offsets in the CSR
*                       representation of the graph)
* @param i - The index of the node to be colored
* @param d_colors - The device-level array of colors assigned to each vertex
*****/
__device__ uint *d_recursiveColorationHelper (GraphStruct *str, uint *d_candidates, node i, uint *d_colors) {
    uint color = UINT_MAX;
    if (d_candidates[i] == UINT_MAX) {
        color = i;
    }
    else {
        node neigh = d_getNeigh(str, i, d_candidates[i]);
        color = d_recursiveColorationHelper(str, d_candidates, neigh, d_colors)[neigh];
    }

    if (color != UINT_MAX) {
        d_colors[i] = color;
    }
    return d_colors;
}


/*****
* Kernel that recognizes the connected components in the graph and colors them
* @param str - The structure of the graph
* @param d_candidates - The device-level array of candidates to become part of
*                       the spanning tree
* @param d_colors - The device-level array of colors assigned to each vertex
*****/
__global__ void colorationProcess(GraphStruct *str, uint *d_candidates, uint *d_colors) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is out of bounds returns immediately
    if (idx >= str->nodeSize) {
        return;
    }

    d_recursiveColorationHelper(str, d_candidates, idx, d_colors);
}











//*** SCAN FUNCTIONS ***//

__global__ void prescan(uint *g_odata, uint *g_idata, uint *aux, int n, int smemSize)
{
  extern __shared__ int temp[];// allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;
  int idx = blockIdx.x * blockDim.x + thid;

  // load input into shared memory
  temp[2 * thid] =  (2 * idx < n) ? g_idata[2 * idx] : 0;
  temp[2 * thid + 1] = (2 * idx + 1 < n) ? g_idata[2 * idx + 1] : 0;

  for (int d =smemSize>>1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d)
    {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      if (bi < smemSize && ai < smemSize) {
        temp[bi] += temp[ai];
      }
    }
    offset *= 2;
  }

  if (thid == 0)
  {
    aux[blockIdx.x] = temp[smemSize - 1];
    temp[smemSize - 1] = 0;
  }

  for (int d = 1; d < smemSize; d *= 2) // traverse down tree & build scan
  {
      offset >>= 1;
    __syncthreads();

    if (thid < d)
    {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      if (bi < smemSize && ai < smemSize) {
        int t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
  }


  __syncthreads();
  if (idx <= (n / 2)) {
      g_odata[2*idx] = temp[2*thid]; // write results to device memory
      g_odata[2*idx+1] = temp[2*thid+1];
  }
}


void cpuScan(uint *array, int start, int end) {
    if (end - start <= 1) {
        return;
    }

    int temp = array[start + 1];
    array[start + 1] = array[start];
    array[start] = 0;

    for (uint i = start + 1; i < end - 1; i++) {
        int sum = array[i] + temp;
        temp = array[i + 1];
        array[i + 1] = sum;
    }
}


__global__ void cfinal_sum(uint *g_odata, uint *aux, uint n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (blockIdx.x == 0 || 2 * idx >= n) {
      return;
  }

  uint sum = 0;
  for (uint i = 0; i < blockIdx.x; ++i) {
      sum += aux[i];
  }

  if (2 * idx == n - 1) {
      g_odata[2 * idx] += sum;
      return;
  }

  g_odata[2 * idx] += sum;
  g_odata[2 * idx + 1] += sum;
}

//****************//






__global__ void cumulatedDegreeUpdate(GraphStruct *str, uint *d_cumDegs, uint *d_colors, uint *d_flag) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= str->nodeSize) {
        return;
    }

    uint color = d_colors[idx];
    node svSuccessor = d_getRoot(idx, d_flag, d_colors);
    uint sum = 0;

    for (uint i = 0; i < d_deg(str, idx); i++) {
        node neigh = d_getNeigh(str, idx, i);
        uint neighColor = d_colors[neigh];

        if (color != neighColor) {
            sum++;
        }
    }

    atomicAdd(&(d_cumDegs[svSuccessor]), sum);
}







__global__ void graphContraction(GraphStruct *str, uint *d_colors, uint *d_flag,
                                 uint *d_cumDegs, node *d_newNeighs, uint *d_newWeights) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is out of bounds returns immediately
    if (idx >= str->nodeSize) {
        return;
    }

    uint color = d_colors[idx];
    node superVertex = d_getRoot(idx, d_flag, d_colors);

    for (uint i = 0; i < d_deg(str, idx); i++) {
        node neigh = d_getNeigh(str, idx, i);
        uint neighColor = d_colors[neigh];

        if (color != neighColor) {
            int weight = d_getWeight(str, idx, i);
            uint position = atomicAdd(&(d_cumDegs[superVertex]), 1);
            d_newNeighs[position] = d_getRoot(neigh, d_flag, d_colors);
            d_newWeights[position] = weight;
        }
    }
}






__global__ void svIdentification (GraphStruct *str, uint *d_colors, uint *d_candidates, uint *d_flag) {
    // Initialize one thread per node
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is out of bounds returns immediately
    if (idx >= str->nodeSize) {
        return;
    }

    if (d_colors[idx] == idx) {
        d_flag[idx] = 1;
    }
}







void computeMST (std::string fpath, std::string tname) {
    // Generation of a random graph
    std::random_device rd;
    std::default_random_engine eng(FIXED_SEED);
    uint maxWeight = MAX_WEIGHT;
    float prob = .5;
    bool GPUEnabled = 1;
    Graph *graphPointer;

    // Generation of the graph
    if (TESTING) {
        string path(fpath);
        printf("Generating graph from file\n");
        graphPointer = rgraph(path, true);
    }
    else {
        graphPointer = new Graph(SIZE, GPUEnabled);
        graphPointer->randGraph(prob, true, maxWeight, eng);

        if (!graphPointer->isConnected()) {
            cout << "The graph is not connected" << endl;
            return;
        }
    }


    uint iterations = 0;




    // Configuration of the GPU kernel
    uint blockDim = BLOCK_SIZE;
    uint *candidates;




    // Events to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    float spliTime = 0;
    float totalTime = 0;




    // Variables calculating the MST weight
    unsigned long long int mstWeight = 0;
    unsigned long long int *d_mstWeight;
    CHECK(cudaMalloc((void **)&d_mstWeight, sizeof(unsigned long long int)));
    CHECK(cudaMemcpy(d_mstWeight, &mstWeight, sizeof(unsigned long long int), cudaMemcpyHostToDevice));


    // Main block of the algorithm
    while (graphPointer->getStruct()->nodeSize > 1) {
        // Initialization of the variables associated with the graph
        GraphStruct *str = graphPointer->getStruct();
        uint size = str->nodeSize;
        uint edgeSize = str->edgeSize;
        cout << "Processing a graph of size: " << size << " with " << edgeSize << " edges.\n\n";
        uint gridDim = (size + blockDim - 1) / blockDim;
        if (DEBUGGING && size < 15 && str->edgeSize < 100) {
            graphPointer->print(true);
            print_d<<<1, 1>>>(str, 1);
            CHECK(cudaDeviceSynchronize());
        }
        candidates = new uint[size];



        // First setp of the algorithm
        uint *d_candidates;
        CHECK(cudaMalloc((void**)&d_candidates, (size) * sizeof(uint)));
        CHECK(cudaMemset(d_candidates, 0, (size) * sizeof(uint)));
        cout << "Launching kernel FIND CHEAPEST -- (" << blockDim << ", 1, 1) -- (" << gridDim << ", 1, 1)" << endl;
        cudaEventRecord(start);
        findCheapest<<<gridDim, blockDim>>>(str, d_candidates);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("Finding the cheapest edge for every vertex took: %.5f seconds\n\n", spliTime);
        totalTime += spliTime;




        // ~Debugging~ print the cheapest edge for every vertex
        if (DEBUGGING && size < 15) {
            cout << "The cheapest edge for every vertex" << endl;
            CHECK(cudaMemcpy(candidates, d_candidates, (size) * sizeof(uint), cudaMemcpyDeviceToHost));
            for (uint i = 0; i < size; i++) {
                cout << "node (" << i << ") -> " << str->getNeigh(i, candidates[i]) << "("
                    << str->getWeight(i, candidates[i]) << ")" << endl;
            }
            cout << "\n\n\n";
        }



        // Second step of the algorithm
        cout << "Launching kernel MIRRORED EDGES REMOVAL -- (" << blockDim << ", 1, 1) -- (" << gridDim << ", 1, 1)" << endl;
        cudaEventRecord(start);
        mirroredEdgesRemoval<<<gridDim, blockDim>>>(str, d_candidates, d_mstWeight);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaMemcpy(candidates, d_candidates, (size) * sizeof(uint), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&mstWeight, d_mstWeight, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("Removing the mirrored edges required: %.5f seconds\n\n", spliTime);
        totalTime += spliTime;




        // ~Debugging~ print the cheapest edge for every vertex update
        if (DEBUGGING && size < 15) {
            cout << "Update of the cheapest edge for every vertex" << endl;
            for (uint i = 0; i < size; i++) {
                cout << "node (" << i << ") -> ";
                if (candidates[i] != UINT_MAX) {
                    cout << str->getNeigh(i, candidates[i]) << "("
                    << str->getWeight(i, candidates[i]) << ")" << endl;
                }
                else {
                    cout << "NULL" << endl;
                }
            }
            printf ("%llu\n", mstWeight);
        }



        cout << "The MST weight at the end of iteration " << iterations + 1 << " is: " << mstWeight << endl;



        // Third step of the algorithm
        cout << "Launching kernel COLORATION PROCESS -- (" << blockDim << ", 1, 1) -- (" << gridDim << ", 1, 1)" << endl;

        // Initialize the color array
        uint *colors = new uint[size];
        uint *d_colors;
        CHECK(cudaMalloc((void**)&d_colors, size * sizeof(uint)));
        CHECK(cudaMemset(d_colors, UINT_MAX, size * sizeof(uint)));



        cudaEventRecord(start);
        colorationProcess<<<gridDim, blockDim>>>(str, d_candidates, d_colors);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaMemcpy(colors, d_colors, size * sizeof(uint), cudaMemcpyDeviceToHost));
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("Removing the mirrored edges required: %.5f seconds\n\n", spliTime);
        totalTime += spliTime;




        // Print the coloring
        if (DEBUGGING) {
            uint *checkColoring = new uint[size];

            for (uint i = 0; i < size; i++) {
                checkColoring[i] = 0;
            }

            for (uint i = 0; i < size; i++) {
                checkColoring[colors[i]]++;
            }

            uint nonZeroColors = 0;
            for (uint i = 0; i < size; i++) {
                if (checkColoring[i] != 0) {
                    nonZeroColors++;
                }
            }

            cout << "There is a total of " << nonZeroColors << " colors" << endl;

            cout << "\n\n\n";

            delete[] checkColoring;
        }




        /**
         * If the coloring coming out of the last kernel contains only one color
         * then it means that the edge added in the last step was the one needed
         * to merge the partial trees
         **/
        uint color = colors[0];
        bool uniqueColor = true;
        for (uint i = 1; i < size; i++) {
            if (colors[i] != color) {
                uniqueColor = false;
                break;
            }
        }
        if (uniqueColor) {
            cout << "THE CALCULATION OF THE MST IS COMPLETE\n";
            cout << "THE MST WEIGHT IS: " << mstWeight << endl;
            printf("Total elapsed time: %.5f seconds\n\n", totalTime);

            // Cuda event dealloc
            CHECK(cudaEventDestroy(start));
            CHECK(cudaEventDestroy(stop));

            // Cuda memory deallocation
            CHECK(cudaFree(d_candidates));
            CHECK(cudaFree(d_colors));
            CHECK(cudaFree(d_mstWeight));

            // Host memory deallocation
            delete[] candidates;
            delete[] colors;
            delete graphPointer;

            if (TESTING) {
                string path(LOGPATH + string("GPU") + "_" + tname);

                ofstream logfile(path, ios_base::app);

                if (logfile.is_open()){
                    cout << "Writing to file " << path << endl;
                    logfile << mstWeight << "\n" << totalTime << "\n";
                    logfile.close();
                }
                else {
                    cout << "Unable to open file";
                }
            }

            return;
        }





        // Fourth step of the algorithm
        cout << "Doing a round of scan on the flag vector, size: " << size << endl;
        uint *flag = new uint[size];
        uint *cFlag = new uint[size];
        uint *d_flag, *d_ogFlag;

        // setup di d_ogFlag
        CHECK(cudaMalloc((void**)&d_ogFlag, (size) * sizeof(uint)));
        CHECK(cudaMemset(d_ogFlag, 0, (size) * sizeof(uint)));
        cudaEventRecord(start);
        svIdentification <<< gridDim, blockDim >>> (str, d_colors, d_candidates, d_ogFlag);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("Building the flag array took:   %.5f seconds\n", spliTime);
        totalTime += spliTime;

        if (DEBUGGING) {
            CHECK(cudaMemcpy(cFlag, d_ogFlag, size * sizeof(uint), cudaMemcpyDeviceToHost));
        }

        // SMEM kernel configuration
        uint smemSize = 2 * blockDim;
        uint smem = smemSize * sizeof(uint);
        uint numSmemBlock = (size + smemSize - 1) / smemSize;

        // Setup the auxiliary array
        uint *aux, *d_aux;
        aux = new uint[numSmemBlock];
        CHECK(cudaMalloc((void **) &d_aux, (numSmemBlock) * sizeof(uint)));
        CHECK(cudaMemset(d_aux, 0, (numSmemBlock) * sizeof(uint)));

        // Setup of the d_flag array
        CHECK(cudaMalloc((void**)&d_flag, (size) * sizeof(uint)));
        CHECK(cudaMemset(d_flag, 0, (size) * sizeof(uint)));

        printf("prescan procedure on the flag array of size: %d ...\n", size);
        cudaEventRecord(start);
        prescan <<<  numSmemBlock, blockDim, smem >>> (d_flag, d_ogFlag, d_aux, size, smemSize);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("The first prescan procedure took:   %.5f seconds\n", spliTime);

        totalTime += spliTime;

        // Put everything together
        printf("final summation procedure...\n");
        cudaEventRecord(start);
        cfinal_sum <<< gridDim, blockDim >>> (d_flag, d_aux, size);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("The final summation procedure took:   %.5f seconds\n\n", spliTime);

        totalTime += spliTime;

        CHECK(cudaMemcpy(flag, d_flag, size * sizeof(uint), cudaMemcpyDeviceToHost));

        if (DEBUGGING) {
            cpuScan(cFlag, 0, size);

            for (uint i = 0; i < size - 1; i++) {
                if (cFlag[i] != flag[i]) {
                    cout << "I due array sono diversi in posizione " << i << "   " << cFlag[i] << "   " << flag[i] << endl;
                    return;
                }
            }

            delete[] cFlag;
        }
        cout << "The contracted graph will contain " << flag[size - 1] << " supervertices\n\n" << endl;
        // Spring Cleaning
        delete[] (aux);
        CHECK(cudaFree(d_aux));
        CHECK(cudaFree(d_ogFlag));





        // Fifth step of the algorithm

        // Allocating resources for the new cumulated degrees array
        uint newNodeSize = flag[size - 1];
        uint cumDegSize = newNodeSize + 1;
        uint *cumDegs = new uint[cumDegSize];
        uint *d_cumDegs;
        CHECK(cudaMalloc((void**)&d_cumDegs, (cumDegSize) * sizeof(uint)));
        CHECK(cudaMemset(d_cumDegs, 0, (cumDegSize) * sizeof(uint)));

        cout << "Launching kernel CUMULATED DEGREE UPDATE -- (" << blockDim << ", 1, 1) -- (" << gridDim << ", 1, 1)" << endl;

        cudaEventRecord(start);
        cumulatedDegreeUpdate<<<gridDim, blockDim>>>(str, d_cumDegs, d_colors, d_flag);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("Doing the computation of the cumulated degrees took: %.5f seconds\n\n", spliTime);

        totalTime += spliTime;

        CHECK(cudaMemcpy(cumDegs, d_cumDegs, cumDegSize * sizeof(uint), cudaMemcpyDeviceToHost));




        // Perform another prefix sum on the cumDegrees array
        // Setup aux
        aux = new uint[numSmemBlock];
        CHECK(cudaMalloc((void **) &d_aux, (numSmemBlock) * sizeof(uint)));
        CHECK(cudaMemset(d_aux, 0, (numSmemBlock) * sizeof(uint)));

        uint *cCumDegs;

        if (DEBUGGING) {
            cCumDegs = new uint[cumDegSize];
            for (uint i = 0; i < cumDegSize; i++) {
                cCumDegs[i] = cumDegs[i];
            }
        }

        // Setup d_ogCumDegs
        uint *d_ogCumDegs;
        CHECK(cudaMalloc((void **) &d_ogCumDegs, cumDegSize * sizeof(uint)));
        CHECK(cudaMemcpy(d_ogCumDegs, cumDegs, (cumDegSize) * sizeof(uint), cudaMemcpyHostToDevice));

        printf("prescan procedure on the cumDegs array of size: %d ...\n", size);
        cudaEventRecord(start);
        prescan <<<  numSmemBlock, blockDim, smem >>> (d_cumDegs, d_ogCumDegs, d_aux, cumDegSize, smemSize);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("The first prescan procedure took:   %.5f seconds\n", spliTime);

        totalTime += spliTime;

        // Put everything together
        printf("final summation procedure...\n");
        cudaEventRecord(start);
        cfinal_sum <<< gridDim, blockDim >>> (d_cumDegs, d_aux, cumDegSize);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("The final summation procedure took:   %.5f seconds\n\n", spliTime);

        totalTime += spliTime;

        CHECK(cudaMemcpy(cumDegs, d_cumDegs, cumDegSize * sizeof(uint), cudaMemcpyDeviceToHost));
        if (DEBUGGING) {
            cpuScan(cCumDegs, 0, cumDegSize);

            for (uint i = 0; i < cumDegSize - 1; i++) {
                if (cCumDegs[i] != cumDegs[i]) {
                    cout << "I due array sono diversi in posizione " << i << endl;
                    cout << cCumDegs[i] << "   " << cumDegs[i];
                    return;
                }
            }
        }
        cout << "The contracted graph will contain " << cumDegs[cumDegSize - 1] << " edges" << endl;
        cout << "The old graph structure contained " << str->edgeSize << " edges\n\n" << endl;

        // Spring cleaning
        free(aux);
        CHECK(cudaFree(d_aux));
        CHECK(cudaFree(d_ogCumDegs));




        // Allocating space for the arrays in the newly contracted graph
        uint newEdgeSize = cumDegs[cumDegSize - 1];
        node *newNeighs = new node[newEdgeSize];
        uint *newWeights = new uint[newEdgeSize];

        uint *d_newNeighs, *d_newWeights;
        CHECK(cudaMalloc((void **)&d_newNeighs, newEdgeSize * sizeof(node)));
        CHECK(cudaMalloc((void **)&d_newWeights, newEdgeSize * sizeof(uint)));
        CHECK(cudaMemset(d_newNeighs, 0, newEdgeSize * sizeof(node)));
        CHECK(cudaMemset(d_newWeights, 0, newEdgeSize * sizeof(uint)));

        cout << "Launching kernel GRAPH CONTRACTION -- (" << blockDim << ", 1, 1) -- (" << gridDim << ", 1, 1)" << endl;
        cudaEventRecord(start);
        graphContraction<<<gridDim, blockDim>>>(str, d_colors, d_flag, d_cumDegs, d_newNeighs, d_newWeights);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        spliTime = milliseconds / 1000.0;
        printf("The contratcion of the new neighbour and weight arrays took: %.5f seconds\n\n", spliTime);
        CHECK(cudaMemcpy(newNeighs, d_newNeighs, newEdgeSize * sizeof(node), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(newWeights, d_newWeights, newEdgeSize * sizeof(uint), cudaMemcpyDeviceToHost));

        if (DEBUGGING) {
            node *checkNewNeighs = new node[newEdgeSize];
            uint *checkNewWeights = new uint[newEdgeSize];
            // Copy the contents of cumDegs into a new array
            for (uint i = 0; i < cumDegSize; i++) {
                cCumDegs[i] = cumDegs[i];
            }

            cudaEventRecord(start);
            for (uint i = 0; i < size; i++) {
                uint color = colors[i];
                node superVertex = getRoot(i, flag, colors);

                for (uint j = 0; j < str->deg(i); j++) {
                    node neigh = str->getNeigh(i, j);
                    uint neighColor = colors[neigh];

                    if (color != neighColor) {
                        int weight = str->getWeight(i, j);
                        uint position = cCumDegs[superVertex];
                        checkNewNeighs[position] = getRoot(neigh, flag, colors);
                        checkNewWeights[position] = weight;
                        cCumDegs[superVertex]++;
                    }
                }
            }

            cout << "I due array sono uguali" << endl;
            delete[] cCumDegs;
            delete[] checkNewNeighs;
            delete[] checkNewWeights;
        }




        // Reconstructing the graph
        graphPointer->copyConstructor(newNodeSize, newEdgeSize, newNeighs, newWeights, cumDegs);

        printf("----------------------------------\n\n");




        // Updating the iteration information
        totalTime += spliTime;
        iterations++;


        // Cuda memory deallocation
        CHECK(cudaFree(d_candidates));
        CHECK(cudaFree(d_colors));
        CHECK(cudaFree(d_flag));
        CHECK(cudaFree(d_cumDegs));
        CHECK(cudaFree(d_newNeighs));
        CHECK(cudaFree(d_newWeights));



        // Host memory deallocation
        delete[] candidates;
        delete[] colors;
        delete[] flag;
        delete[] cumDegs;
        delete[] newNeighs;
        delete[] newWeights;
    }

    return;
}


int main() {
    const std::vector<std::string> test = { "nw", "cal", "lks", "bay", "ne", "west", "col", "east", "ctr", "fla", "ny" };
    const std::filesystem::path sandbox{TEST};
    std::size_t i {0};
    for (auto const& dir_entry : std::filesystem::directory_iterator{sandbox}) {
        if (dir_entry.path().extension() != ".txt") {
            continue;
        }
        cout << dir_entry.path() << endl;
        computeMST(
            dir_entry.path().string(), 
            test[i]
        );
        ++i;
    }
}

