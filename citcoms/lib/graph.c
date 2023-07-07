#include "stdio.h"
#include "graph.h"
#include "global_defs.h"

int getNumNodes(struct Graph *graph){
    return graph->numNodes;
}

int getNeighbors(struct Graph *graph, int index){
    return graph->degree;
}

extern struct All_variables *tempE;
extern int tempM;
extern int tempLevel;

void getNeig(struct Graph *graph, double *u, int index){
  int count = 1;
  index+=1;
  for(int i = 0; i < 24; i+=3){
    int nodeb = tempE->IEN[tempLevel][tempM][index].node[count];
    u[i] = graph->states[tempE->ID[tempLevel][tempM][nodeb].doff[1]];
    u[i+1] = graph->states[tempE->ID[tempLevel][tempM][nodeb].doff[2]];
    u[i+2] = graph->states[tempE->ID[tempLevel][tempM][nodeb].doff[3]];
    count++;
  }
}

double neg[24] = {0};

void GraphProcess(struct Graph *graph, double * result, fun_gather gather, fun_apply apply){
  int nodes = getNumNodes(graph);
  for(int vi = 0; vi < nodes; vi++){
    int neighbors = getNeighbors(graph, vi);
    getNeig(graph, neg, vi);
    for(int neighbor = 0; neighbor < neighbors; neighbor++){
      gather(vi, neighbor, graph, neg);
    }
    apply(vi, graph, result);
  }
}