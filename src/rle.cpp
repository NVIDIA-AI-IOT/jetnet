
#include "rle.h"
#include <iostream>


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

bool pick_or(size_t ac, uint8_t av, size_t bc, uint8_t bv) {
    if ((av == 0) && (bv == 0)) {
        if (ac < bc) {
            return 0;
        } else {
            return 1;
        }
    } else if ((av == 0) && (bv == 1)) {
        return 1;
    } else if ((av == 1) && (bv == 0)) {
        return 0;
    } else {
        if (ac > bc) {
            return 0;
        } else {
            return 1;
        }
    }
}

size_t dec_count(std::vector<size_t> &counts, size_t index, int dec) {
    size_t size = counts.size();
    while ((dec > 0) && (index < size)) {
        if (counts[index] > dec) {
            counts[index] -= dec;
            return index;
        } else {
            auto orig = counts[index];
            counts[index] -= dec;
            dec -= orig;
            index += 1;
        }
    }
    return index;
}

RLE binary_rle_or(RLE &a, RLE &b) {
    auto ac = a.counts;
    auto av = a.values;
    auto bc = b.counts;
    auto bv = b.values;

    size_t ia = 0;
    size_t ib = 0;

    std::vector<size_t> cc = {0};
    std::vector<uint8_t> cv = {0};

    size_t a_size = ac.size();
    size_t b_size = bc.size();

    while ((ia < a_size) && (ib < b_size)) {
        auto sel = pick_or(ac[ia], av[ia], bc[ib], bv[ib]);
        size_t count;
        uint8_t value;
        if (sel == 0) {
            count = ac[ia];
            value = av[ia];
        } else {
            count = bc[ib];
            value = bv[ib];
        }
        if (value == cv[cc.size() - 1]) {
            cc[cc.size() - 1] += count;
        } else {
            cc.push_back(count);
            cv.push_back(value);
        }
        ia = dec_count(ac, ia, count);
        ib = dec_count(bc, ib, count);
    }

    return {cc, cv, a.size};
}