#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <string>
#include <memory>

using namespace emscripten;

#include "rle.h"

struct RLE {
    std::vector<size_t> counts;
    std::vector<uint8_t> values;
};

//std::pair<std::list<size_t>, std::list<uint8_t>>
RLE js_rle_encode(std::string x) {
    auto result = rle_encode((uint8_t*) x.data(), (size_t) x.length());
    RLE rle({
        {result.first.begin(), result.first.end()},
        {result.second.begin(), result.second.end()}
    });
    // return result;
    return rle;
}

// std::string js_rle_decode_rgba(std::vector<size_t> counts, std::vector<uint8_t> values, int size, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {

// }

// }
struct DataContainer {
    std::unique_ptr<uint32_t> _data;
    size_t _size;
    val get() {
        return val(typed_memory_view(_size * 4, (uint8_t*) _data.get()));
    }
};

DataContainer make_red_rgba_array(size_t size, uint32_t color) {
    DataContainer container = {
        std::make_unique<uint32_t>(size),
        size
    };
    
    uint32_t *data = container._data.get();
    
    for (int i = 0; i < size; i++) {
        data[i] = color;
    }

    return container;
}

EMSCRIPTEN_BINDINGS(my_module) {

    value_object<RLE>("RLE")
        .field("counts", &RLE::counts)
        .field("values", &RLE::values)
        ;

    class_<DataContainer>("DataContainer")
        .function("get", &DataContainer::get);

    function("rle_encode", &js_rle_encode);
    function("make_red_rgba_array", &make_red_rgba_array);
    
    register_vector<size_t>("vector<size_t>");
    register_vector<uint8_t>("vector<uint8_t>");
}