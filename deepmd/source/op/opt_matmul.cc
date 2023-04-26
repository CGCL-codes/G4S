#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#pragma once
#include "time.h"
#include "graph.h"


using namespace tensorflow;

int Nsize;

REGISTER_OP("OptMatmul")
    .Attr("T: {double} = DT_DOUBLE")
    .Input("xx: T")
    .Input("w: T")
    .Output("res: T");

template <typename T>
class OptMatmulOp : public OpKernel {
 public:
  explicit OptMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Graph graph;
    int context_input_index = 0;
    const Tensor& xx_tensor = context->input(context_input_index++);
    const Tensor& w_tensor = context->input(context_input_index++);
    int M = xx_tensor.shape().dim_size(0);
    int N = xx_tensor.shape().dim_size(1);
    int K = w_tensor.shape().dim_size(1);
    const T * xx = xx_tensor.flat<T>().data();
    const T * w = w_tensor.flat<T>().data();
    
    TensorShape res_shape ;
    res_shape.AddDim(M);
    res_shape.AddDim(K);
    Tensor* res_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, res_shape, &res_tensor));
    T * result = res_tensor->flat<T>().data();
    Nsize = N;
    graph.states = w;
    graph.numNodes = M;
    graph.degree = K;
    const T* A[M];
    for(int i = 0; i < M; i++){
        A[i] = xx + i * N;
    }
    graph.edgeWeight = A;
    GraphProcess(&graph, result,
        [&](int e, int a, struct Graph *graph, double* result){
            int Col = getNeighbors(graph, e);
            result[e * Col + a] = 0;
            for(int k = 0; k < Nsize; k++){
                result[e * Col + a] += graph->edgeWeight[e][k] * graph->states[k * Col + a];
            }
        },
        [&](int e, struct Graph *graph, double *Au){

        });
  }
  private:
    int M,N,K;
};

#define REGISTER_CPU(T)                                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OptMatmul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OptMatmulOp<T>);
REGISTER_CPU(double);

#undef REGISTER_CPU
