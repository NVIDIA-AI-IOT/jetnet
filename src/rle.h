#pragma once

#include <stdint.h>
#include <list>


std::pair<std::list<long>, std::list<long>> rle_encode(uint8_t *data, size_t size);