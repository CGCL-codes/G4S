#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "custom_op.h"
//#include "sys/time.h"
#pragma once
#include "time.h"
#include "graph.h"
// #include "spmm.h"


using namespace tensorflow;

int Nsize;

REGISTER_OP("OptMatmul")
    .Attr("T: {double} = DT_DOUBLE")
    .Input("xx: T")
    .Input("w: T")
    .Output("res: T");

// template <typename T>
// void gather(const T *A, const T *B, T *C, int i, int j, int ARow, int BCol, int BRow){//CRow = ARow,CCol = BCol,ACol = BRow
//     C[i * BCol + j] = 0;
//     #pragma omp parallel for
//     for(int k = 0; k < BCol; k ++){
//         C[i * BCol + j] += A[i *BRow + k] * B[k * BCol + j];
//     }
// }

// template <typename T>
// void gather(const double **A, const double *B, double *C, int i, int j, int ARow, int BCol, int BRow){//CRow = ARow,CCol = BCol,ACol = BRow
//     C[i * BCol + j] = 0;
//     #pragma omp parallel for
//     for(int k = 0; k < Nsize; k ++){
//         C[i * BCol + j] += A[i][k] * B[k * BCol + j];
//     }
// }

// // template <typename T>
// void apply(const double **A, const double *B, double *C, int i, int ARow, int BCol){
//     return ;
// }


// void gather(int vi, int neighbor, Graph *graph, double* result){
//     //printf("gather\n");
//     int Col = getNeighbors(graph, vi);
//     // graph->temp[vi * Col + neighbor] = 0;
//     // //#pragma omp parallel for
//     // for(int k = 0; k < Nsize; k++){
//     //     printf("k=%d\n",k);
//     //     graph->temp[vi * Col + neighbor] += graph->edgeWeight[vi][k] * graph->states[k * Col + neighbor];
//     // }
//     result[vi * Col + neighbor] = 0;
//     for(int k = 0; k < Nsize; k++){
//         //printf("k=%d\n",k);
//         result[vi * Col + neighbor] += graph->edgeWeight[vi][k] * graph->states[k * Col + neighbor];
//     }
// }

// //template<typename T>
// void apply(int vi, Graph *graph, double* result){
//     // printf("apply\n");
//     // int neighbors = getNeighbors(graph, vi);
//     // for(int neighbor = 0; neighbor < neighbors; neighbor++){
//     //     printf("neighbor=%d\n",neighbor);
//     //     result[vi * neighbors + neighbor] = graph->temp[vi * neighbors + neighbor];
//     // }
//     return ;
// }

template <typename T>
class OptMatmulOp : public OpKernel {
 public:
  explicit OptMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //printf("------Compute--------\n");
    //读取tensor
    Graph graph;
    int context_input_index = 0;
    const Tensor& xx_tensor = context->input(context_input_index++);
    const Tensor& w_tensor = context->input(context_input_index++);
    int M = xx_tensor.shape().dim_size(0);
    int N = xx_tensor.shape().dim_size(1);
    int K = w_tensor.shape().dim_size(1);
    const T * xx = xx_tensor.flat<T>().data();
    const T * w = w_tensor.flat<T>().data();
    
    // graph->states = w_tensor.flat<T>().data();
    TensorShape res_shape ;
    res_shape.AddDim(M);
    res_shape.AddDim(K);
    Tensor* res_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, res_shape, &res_tensor));
    T * result = res_tensor->flat<T>().data();
    Nsize = N;
    //printf("获取graph->states\n");
    graph.states = w;
    graph.numNodes = M;
    graph.degree = K;
    const T* A[M];
    for(int i = 0; i < M; i++){
        A[i] = xx + i * N;
    }
    //printf("graph->edgeweight赋值\n");
    graph.edgeWeight = A;
    // for(int i = 0; i < M; i++){
    //     graph.edgeWeight[i] = xx + i * N;
    // }
    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N ;j++){
    //         printf("graph.edgeweight[%d][%d] = %lf ", i, j, graph.edgeWeight[i][j]);
    //     }
    //     printf("\n");
    // }
    // for(int i = 0; i < N * K; i++){
    //     printf("graph.states[%d] = %lf ", i, graph.states[i]);
    // }
    // printf("\n");
    //GraphProcess(&graph, result, gather, apply);
    //free(graph.edgeWeight);
    //result = graph.temp;

    // printf("\n");
    GraphProcess(&graph, result,
        [&](int e, int a, struct Graph *graph, double* result){
            int Col = getNeighbors(graph, e);
            // graph->temp[e * Col + a] = 0;
            // for(int k = 0; k < Nsize; k++){
            //     graph->temp[e * Col + a] += graph->edgeWeight[e][k] * graph->states[k * Col + a];
            // }
            result[e * Col + a] = 0;
            for(int k = 0; k < Nsize; k++){
                result[e * Col + a] += graph->edgeWeight[e][k] * graph->states[k * Col + a];
            }
        },
        [&](int e, struct Graph *graph, double *Au){
            // int Col = getNeighbors(graph, e);
            // for(int a = 0; a < Col; a++){
            //     result[e * Col + a] = graph->temp[e * Col + a];
            // }
        });
    // const T* A[M];
    // for(int i = 0; i < M; i++){
    //     A[i] = xx + i * N;
    // }
    // spmm(M, K, A, w, result, gather, apply, N);
    // for(int i = 0; i < M * K; i++){
    //     printf("result[%d] = %lf ", i, result[i]);
    // }
    // printf("\n");
    // printf("---------fininshed------------\n");
  }
  private:
    int M,N,K;
};

#define REGISTER_CPU(T)                                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OptMatmul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OptMatmulOp<T>);
//REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU
