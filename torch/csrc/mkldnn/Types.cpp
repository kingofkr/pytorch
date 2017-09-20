#include "Types.h"

namespace torch { namespace mkldnn {

mkldnnDataType_t getMkldnnDataType(PyObject *tensorClass)
{
  if (tensorClass == THPFloatTensorClass) {
    return MKLDNN_DATA_FLOAT;
  } else if (tensorClass == THPDoubleTensorClass) {
    return MKLDNN_DATA_DOUBLE;
  } else if (tensorClass == THPHalfTensorClass) {
    return MKLDNN_DATA_HALF;
  }
  if (!PyType_Check(tensorClass)) {
    throw std::runtime_error("getMkldnnDataType() expects a PyTypeObject");
  }
  std::string msg("getMkldnnDataType() not supported for ");
  msg += ((PyTypeObject*)tensorClass)->tp_name;
  throw std::runtime_error(msg);
}

mkldnnDataType_t getMkldnnDataType(const at::Tensor& tensor) {
  if (tensor.type().scalarType() == at::kFloat) {
    return MKLDNN_DATA_FLOAT;
  } else if (tensor.type().scalarType() == at::kDouble) {
    return MKLDNN_DATA_DOUBLE;
  } else if (tensor.type().scalarType() == at::kHalf) {
    return MKLDNN_DATA_HALF;
  }
  std::string msg("getMkldnnDataType() not supported for ");
  msg += at::toString(tensor.type().scalarType());
  throw std::runtime_error(msg);
}

PyObject * getTensorClass(PyObject *args)
{
  for (int i = 0; i < PyTuple_Size(args); i++) {
    PyObject *item = PyTuple_GET_ITEM(args, i);
    if (THPModule_isTensor(item)) {
      return (PyObject*)Py_TYPE(item);
    }
  }
  return NULL;
}

}}  // namespace torch::mkldnn
