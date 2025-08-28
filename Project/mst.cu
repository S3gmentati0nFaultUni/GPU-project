// Header file di C++
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <filesystem>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Header file C
#include <time.h>
#include <limits.h>

// Custom files
#include "MST_hk.cuh"
#include "DataStructure/Graph.cuh"
#include "Common/common.h"
#include "Common/sharedMacros.h"
#include "DataStructure/ReadGraph.h"

using namespace std;


void computeMST (std::string fpath, std::string tname, bool useThrust) {
    // Generation of a random graph
    std::random_device rd;
    std::default_random_engine eng(FIXED_SEED);
    Graph *graphPointer;

    // Generation of the graph
    string path(fpath);
    printf("Generating graph from file\n");
    graphPointer = rgraph(path, true);


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
                string mainFileName;
                if (!useThrust) {
                    mainFileName = "GPU";
                }
                else {
                    mainFileName = "GPUthst";
                }
                string path(LOGPATH + mainFileName + "_" + tname);

                ofstream logfile(path, ios_base::out);

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
        uint *aux, *d_aux;


        // Setup of the d_flag array
        CHECK(cudaMalloc((void**)&d_flag, (size) * sizeof(uint)));
        CHECK(cudaMemset(d_flag, 0, (size) * sizeof(uint)));

        if (useThrust) {
            printf("Thrust scan procedure on the flag array of size: %d ...\n", size);
            cudaEventRecord(start);
            thrust::exclusive_scan(thrust::device, d_ogFlag, d_ogFlag + size, d_flag);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaGetLastError());
            cudaEventElapsedTime(&milliseconds, start, stop);
            spliTime = milliseconds / 1000.0;
            printf("Thrust scan took:   %.5f seconds\n", spliTime);
        }
        else {
            // Setup the auxiliary array
            aux = new uint[numSmemBlock];
            CHECK(cudaMalloc((void **) &d_aux, (numSmemBlock) * sizeof(uint)));
            CHECK(cudaMemset(d_aux, 0, (numSmemBlock) * sizeof(uint)));

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
            
            // Spring Cleaning
            delete[] (aux);
            CHECK(cudaFree(d_aux));
        }

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

        // Sprint cleaning
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

        if (useThrust) {
            printf("Thrust scan procedure on the cumDegs array of size: %d ...\n", cumDegSize);
            cudaEventRecord(start);
            thrust::exclusive_scan(thrust::device, d_ogCumDegs, d_ogCumDegs + cumDegSize, d_cumDegs);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaGetLastError());
            cudaEventElapsedTime(&milliseconds, start, stop);
            spliTime = milliseconds / 1000.0;
            printf("Thrust scan took:   %.5f seconds\n", spliTime);
        }
        else {

            // Perform another prefix sum on the cumDegrees array
            // Setup aux
            aux = new uint[numSmemBlock];
            CHECK(cudaMalloc((void **) &d_aux, (numSmemBlock) * sizeof(uint)));
            CHECK(cudaMemset(d_aux, 0, (numSmemBlock) * sizeof(uint)));

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
            
            free(aux);
            CHECK(cudaFree(d_aux));
        }
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
    const std::vector<std::string> test = { "nw", "cal", "lks", "bay", "ne", "west", "usa", "col", "east", "ctr", "fla", "ny" };
    const std::filesystem::path sandbox{TEST};
    std::size_t i {0};
    for (auto const& dir_entry : std::filesystem::directory_iterator{sandbox}) {
        if (dir_entry.path().extension() != ".txt") {
            continue;
        }
        cout << dir_entry.path() << endl;
        computeMST(
            dir_entry.path().string(), 
            test[i],
            true
        );
        ++i;
    }
}

