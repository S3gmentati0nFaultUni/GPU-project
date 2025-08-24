#ifndef MST_HK_CUH
#define MST_HK_CUH

    #include "DataStructure/Graph.cuh"

    // CUDA Kernels
    /**
    * Print the graph on device (verbose = 1 for "verbose print")
    * @param verbose print the complete graph
    */
    __global__ void print_d(GraphStruct *str, bool verbose);

    /*****
    * Kernel that finds the cheapest edge in the adjacency list of every node
    * @param str - The structure of the graph
    * @param d_candidates - The device-level array of candidates to become part of
    *                       the spanning tree (edges saved as offsets in the CSR
    *                       representation of the graph)
    *****/
    __global__ void findCheapest (GraphStruct *str, uint *d_candidates);

    /*****
    * Kernel that removes the mirrored edges from the graph. A mirrored edge is
    * simply an edge pointing from the source to the destination and vice versa in
    * an oriented graph, the removal logic is to cut the edge with the lowest source
    * @param str - The structure of the graph
    * @param d_candidates - The device-level array of candidates to become part of
    *                       the spanning tree (edges saved as offsets in the CSR
    *                       representation of the graph)
    *****/
    __global__ void mirroredEdgesRemoval (
        GraphStruct *str, 
        uint *d_candidates, 
        unsigned long long int *d_weight
    );

    /*****
    * Kernel that recognizes the connected components in the graph and colors them
    * @param str - The structure of the graph
    * @param d_candidates - The device-level array of candidates to become part of
    *                       the spanning tree
    * @param d_colors - The device-level array of colors assigned to each vertex
    *****/
    __global__ void colorationProcess(GraphStruct *str, uint *d_candidates, uint *d_colors);

    // Scan functions
    __global__ void prescan(uint *g_odata, uint *g_idata, uint *aux, int n, int smemSize);
    __global__ void cfinal_sum(uint *g_odata, uint *aux, uint n);
    /////////////////

    __global__ void cumulatedDegreeUpdate(
        GraphStruct *str, 
        uint *d_cumDegs, 
        uint *d_colors, 
        uint *d_flag
    );

    __global__ void graphContraction(
        GraphStruct *str, 
        uint *d_colors, 
        uint *d_flag,        
        uint *d_cumDegs, 
        node *d_newNeighs, 
        uint *d_newWeights
    );

    __global__ void svIdentification (
        GraphStruct *str, 
        uint *d_colors, 
        uint *d_candidates, 
        uint *d_flag
    );






    // Device functions
    __device__ uint d_getRoot (uint i, uint *d_flag, uint *d_colors);

    /*****
    * Helper device function that recursively colors the nodes of the graph
    * @param str - The structure of the graph
    * @param d_candidates - The device-level array of candidates to become part of
    *                       the spanning tree (edges saved as offsets in the CSR
    *                       representation of the graph)
    * @param i - The index of the node to be colored
    * @param d_colors - The device-level array of colors assigned to each vertex
    *****/
    __device__ uint *d_recursiveColorationHelper (
        GraphStruct *str, 
        uint *d_candidates, 
        node i, 
        uint *d_colors
    );

    // Graph device functions
    __device__ node d_getNeigh (GraphStruct *str, node i, uint offset);
    __device__ int d_getWeight (GraphStruct *str, node i, uint offset);
    __device__ node d_deg (GraphStruct *str, node i);


    // CPU functions
    uint getRoot (uint i, uint *flag, uint *colors);
    void cpuScan(uint *array, int start, int end);

#endif