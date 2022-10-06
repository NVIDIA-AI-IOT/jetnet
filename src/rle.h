#pragma once

#include <stdint.h>
#include <list>
#include <vector>


struct RLE {
    std::vector<size_t> counts;
    std::vector<uint8_t> values;
    size_t size;
};

std::pair<std::list<size_t>, std::list<uint8_t>> rle_encode(uint8_t *data, size_t size);

bool pick_or(size_t ac, uint8_t av, size_t bc, uint8_t bv);
size_t dec_count(std::vector<size_t> &counts, size_t index, int dec);
RLE binary_rle_or(RLE &a, RLE &b);