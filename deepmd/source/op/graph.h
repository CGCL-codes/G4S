#include <functional>
#include "stdio.h"


struct Graph{
    int numNodes;
    int degree;
    const double ** edgeWeight;
    const double * states;
    double * temp;
};

int getNumNodes(struct Graph *graph){
    return graph->numNodes;
}

int getNeighbors(struct Graph *graph, int index){
    return graph->degree;
}

void GraphProcess(struct Graph *graph, double * result, std::function<void(int, int, struct Graph*, double*)> gather, std::function<void(int, struct Graph*, double *)> apply){
  int nodes = getNumNodes(graph);
  omp_set_num_threads(8);
  #pragma omp parallel for schedule(dynamic , 1)
  for(int vi = 0; vi < nodes; vi++){
    int neighbors = getNeighbors(graph, vi);
    for(int neighbor = 0; neighbor < neighbors; neighbor++){
      gather(vi, neighbor, graph, result);
    }
    apply(vi, graph, result);
  }
}