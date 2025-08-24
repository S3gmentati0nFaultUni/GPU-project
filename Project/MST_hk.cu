#include "MST_hk.cuh"
#include "DataStructure/Graph.cuh"

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