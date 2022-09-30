#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <list>
#include <iostream>

#include "rle.h"

namespace py = pybind11;


std::pair<std::list<long>, std::list<long>> py_rle_encode(py::array_t<uint8_t> x) {
    py::buffer_info buf = x.request();
    uint8_t *data = static_cast<uint8_t *>(buf.ptr);
    return rle_encode(data, buf.size);
}


PYBIND11_MODULE(_jetnet_C, m) {
    m.doc() = "JetNet C extensions"; // optional module docstring
    m.def("rle_encode", &py_rle_encode, "Run length encode numpy array");
}