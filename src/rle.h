#pragma once

#include <stdint.h>
#include <list>


std::pair<std::list<size_t>, std::list<uint8_t>> rle_encode(uint8_t *data, size_t size);