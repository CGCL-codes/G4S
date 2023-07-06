from tensorflow.python.framework import ops
from deepmd.env import tf 


@ops.RegisterGradient("OptMatmul")
def _opt_matmul_grad(op,grad):
    xx = op.inputs[0]
    w = op.inputs[1]
    dxx = tf.matmul(grad, w, False, True)
    dw = tf.matmul(xx, grad, True, False) 
    return [dxx, dw]