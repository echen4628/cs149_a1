#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;
  int threadId;
  int nThreads;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  // printf("In computeAssignments: Thread %d\n", args->threadId);
  double *minDist = new double[args->M];
  
  int M_start = args->M * ((float) args->threadId/args->nThreads);
  int M_end = args->M * (((float) (args->threadId+1) /args->nThreads));

  // Initialize arrays
  for (int m =M_start; m < M_end; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }



  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++) {
    for (int m = M_start; m < M_end; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m]) {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  delete[] minDist;
  // printf("Out computeAssignments: Thread %d\n", args->threadId);
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  // printf("In computeCentroid\n");
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }
  // printf("before free\n");

  delete[] counts;
  // printf("Out computeCentroids\n");
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  // printf("In computeCost\n");
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  delete[] accum;
  // printf("Out computeCost\n");
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  static constexpr int MAX_THREADS = 32;

  std::thread workers[MAX_THREADS];
  WorkerArgs argArray[MAX_THREADS];
  int numThreads = 8;

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  for (int i=0; i<numThreads; i++){
    argArray[i].data = data;
    argArray[i].clusterCentroids = clusterCentroids;
    argArray[i].clusterAssignments = clusterAssignments;
    argArray[i].currCost = currCost;
    argArray[i].M = M;
    argArray[i].N = N;
    argArray[i].K = K;
    argArray[i].nThreads = numThreads;
    argArray[i].threadId = i;
    argArray[i].start = 0;
    argArray[i].end = K;
  }

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    // argArray[0].start = 0;
    // argArray[0].end = K;
    for (int i=1; i<numThreads; i++){
      workers[i] = std::thread(computeAssignments, &argArray[i]);
    }

    computeAssignments(&argArray[0]);

    for (int i=1; i<numThreads; i++){
      workers[i].join();
    }
    // for (int i=0; i<20; i++){
    //   int m = i;
    //   printf("%d,\n", clusterAssignments[m]);
    // }

    // printf("------\n");
  
    // for (int i=20; i>0; i--){
    //   int m = M-1-i;
    //   printf("%d,\n", clusterAssignments[m]);
    // }
    computeCentroids(&argArray[0]);
    computeCost(&argArray[0]);

    iter++;
  }

  free(currCost);
  free(prevCost);
}
