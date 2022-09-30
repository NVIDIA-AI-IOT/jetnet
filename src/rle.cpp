#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <list>
#include <iostream>

namespace py = pybind11;

std::pair<std::list<long>, std::list<long>> rle(py::array_t<uint8_t> x) {
    py::buffer_info buf = x.request();
    uint8_t *data = static_cast<uint8_t *>(buf.ptr);

    uint8_t value = 0;
    long count = 0;
    std::list<long> counts;
    std::list<long> values;

    for (long i = 0; i < buf.size; i++) {
        if (data[i] != value) {
            counts.push_back(count);
            values.push_back(value);
            count = 0;
            value = data[i];
        }
        count++;
    }

    counts.push_back(count);
    values.push_back(value);

    return {counts, values};
}

PYBIND11_MODULE(_jetnet_C, m) {
    m.doc() = "JetNet C extensions"; // optional module docstring
    m.def("rle", &rle, "Run length encode numpy array");
}