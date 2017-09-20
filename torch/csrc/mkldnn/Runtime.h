#ifndef THP_MKLDNN_RUNTIME_INC
#define THP_MKLDNN_RUNTIME_INC

//#include <mkldnn.h>

namespace torch { namespace mkldnn {

extern void* state;

typedef int mkldnnEngine_t;
typedef int mkldnnStream_t;

mkldnnEngine_t getMkldnnEngine();
mkldnnStream_t getMkldnnStream();

}}  // namespace torch::mkldnn

#endif
