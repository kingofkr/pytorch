#include "Runtime.h"

namespace torch { namespace mkldnn {

void* state;

mkldnnEngine_t getMkldnnEngine()
{
  return 0;
}

mkldnnStream_t getMkldnnStream()
{
  return 0;
}

}}  // namespace torch::mkldnn
