
#include "rle.h"


std::pair<std::list<size_t>, std::list<uint8_t>> rle_encode(uint8_t *data, size_t size) {

    uint8_t value = 0;
    size_t count = 0;

    std::list<size_t> counts;
    std::list<uint8_t> values;

    for (size_t i = 0; i < size; i++) {
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