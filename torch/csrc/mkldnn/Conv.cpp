#include "Conv.h"

namespace torch { namespace mkldnn {

void* tensorPointer(THVoidTensor* tensor)
{
  int elementSize = sizeof(float);
  char* ptr = (char*) tensor->storage->data;
  ptr += elementSize * tensor->storageOffset;
  return ptr;
}

void mkldnn_test_op(
    void *state, mkldnnEngine_t engine, mkldnnDataType_t dataType, 
    THVoidTensor *input, THVoidTensor *output)
{
  int size = 1;
  for (int i = 0; i < input->nDimension; ++i) {
    size *= input->size[i];
  }

  float* input_ptr = (float*)tensorPointer(input);
  float* output_ptr = (float*)tensorPointer(output);
  for (int j = 0; j < size; ++j) {
    output_ptr[j] = input_ptr[j] + 1.0;
  }
}

}}  // namespace torch::mkldnn 
