#ifndef THP_MKLDNN_CONV_INC
#define THP_MKLDNN_CONV_INC

#include "Runtime.h"
#include "Types.h"

namespace torch { namespace mkldnn {

void mkldnn_test_op(
    void *state, mkldnnEngine_t engine, mkldnnDataType_t dataType,
    THVoidTensor *input, THVoidTensor *output);

}}  // namespace torch::mkldnn

#endif
