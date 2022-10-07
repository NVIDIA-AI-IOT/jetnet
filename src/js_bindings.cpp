#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <string>
#include <memory>
#include <map>
#include <iostream>
#include <cstring>
#include <iomanip>
using namespace emscripten;

#include "rle.h"

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

Color make_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return {r, g, b, a};
}

struct ColorArray {
    std::unique_ptr<Color[]> _data;
    size_t _size;
    val get() {
        return val(typed_memory_view(_size * 4, (uint8_t*) _data.get()));
    }
};

RLE make_rle() {
    return RLE();
}

RLE js_rle_encode(std::string x) {
    auto result = rle_encode((uint8_t*) x.data(), (size_t) x.length());
    RLE rle({
        {result.first.begin(), result.first.end()},
        {result.second.begin(), result.second.end()},
        (size_t) x.length()
    });

    return rle;
}


ColorArray rle_to_rgba_data(RLE &rle, std::map<uint8_t, Color> colorMap) {

    // create result data
    ColorArray result = {
        std::make_unique<Color[]>(rle.size),
        rle.size
    };
    Color *data = result._data.get();

    // fill result data
    int offset = 0;
    for (int i = 0; i < rle.counts.size(); i++) {
        auto count = rle.counts[i];
        auto value = rle.values[i];
        auto color = colorMap[value];
        for (int j = 0; j < count; j++) {
            data[offset] = color;
            offset++;
        }
    }

    return result;
}

ColorArray binary_rle_to_rgba_data(RLE &rle, Color bg, Color fg) {

    // create result data
    ColorArray result = {
        std::make_unique<Color[]>(rle.size),
        rle.size
    };
    Color *data = result._data.get();

    // fill result data
    int offset = 0;
    for (int i = 0; i < rle.counts.size(); i++) {
        auto count = rle.counts[i];
        auto value = rle.values[i];
        auto color = bg;
        if (value > 0) {
            color = fg;
        }
        for (int j = 0; j < count; j++) {
            data[offset] = color;
            offset++;
        }
    }

    return result;
}

ColorArray make_color_array(size_t size) {
    return {
        std::make_unique<Color[]>(size),
        size
    };
}

void fill_rgba_binary_rle(ColorArray &result, RLE &rle, Color bg, Color fg) {

    // create result data
    Color *data = result._data.get();

    // fill result data
    int offset = 0;
    for (int i = 0; i < rle.counts.size(); i++) {
        auto count = rle.counts[i];
        auto value = rle.values[i];
        auto color = bg;
        if (value > 0) {
            color = fg;
        }
        for (int j = 0; j < count; j++) {
            data[offset] = color;
            offset++;
        }
    }

}


std::map<uint8_t, Color> make_binary_colormap() {
    std::map<uint8_t, Color> result;

    result[0] = {0, 0, 0, 255};
    for (int i = 1; i < 256; i++) {
        result[i] = {255, 255, 255, 255};
    }

    return result;
}

EMSCRIPTEN_BINDINGS(my_module) {

    value_object<RLE>("RLE")
        .field("counts", &RLE::counts)
        .field("values", &RLE::values)
        .field("size", &RLE::size)
        ;

    value_object<Color>("Color")
        .field("r", &Color::r)
        .field("g", &Color::g)
        .field("b", &Color::b)
        .field("a", &Color::a)
        ;

    class_<ColorArray>("ColorArray")
        .function("get", &ColorArray::get)
        ;

    function("make_rle", &make_rle);
    function("make_color", &make_color);
    function("make_color_array", &make_color_array);
    function("rle_encode", &js_rle_encode);
    function("rle_to_rgba_data", &rle_to_rgba_data);
    function("binary_rle_or", &binary_rle_or);
    function("binary_rle_to_rgba_data", &binary_rle_to_rgba_data);
    function("fill_rgba_binary_rle", &fill_rgba_binary_rle);
    
    function("make_binary_colormap", &make_binary_colormap);
    
    register_vector<size_t>("vector<size_t>");
    register_vector<uint8_t>("vector<uint8_t>");
    register_map<uint8_t, Color>("colormap");
}