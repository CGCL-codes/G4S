struct Graph;

typedef void (*fun_gather)(int, int, struct Graph*, double*);

typedef void (*fun_apply)(int, struct Graph*, double *);

int getNumNodes(struct Graph *graph);
int getNeighbors(struct Graph *graph, int index);

void GraphProcess(struct Graph *graph, double * result, fun_gather gather, fun_apply apply);

struct Graph{
    int numNodes;
    int degree;
    const double ** edgeWeight;
    const double * states;
    double * temp;
};