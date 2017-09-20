#ifndef THP_MKLDNN_TYPES_INC
#define THP_MKLDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <string>
//#include <mkldnn.h>
#include "../Types.h"
#include "../THP.h"
#include <ATen/Tensor.h>

namespace torch { namespace mkldnn {

typedef enum mkldnnDataType_t {
  MKLDNN_DATA_HALF,
  MKLDNN_DATA_FLOAT,
  MKLDNN_DATA_DOUBLE
} mkldnnDataType_t;

PyObject * getTensorClass(PyObject *args);
mkldnnDataType_t getMkldnnDataType(PyObject *tensorClass);
mkldnnDataType_t getMkldnnDataType(const at::Tensor& tensor);

}}  // namespace torch::mkldnn

#endif
