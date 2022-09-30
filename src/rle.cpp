
#include "rle.h"


std::pair<std::list<long>, std::list<long>> rle_encode(uint8_t *data, size_t size) {
    
    uint8_t value = 0;
    long count = 0;

    std::list<long> counts;
    std::list<long> values;

    for (long i = 0; i < size; i++) {
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