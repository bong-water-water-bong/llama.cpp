#pragma once

#include <stddef.h>

#define LLAMA_UI_HAS_ASSETS 1

struct llama_ui_asset {
    const char *          name;
    const unsigned char * data;
    size_t                size;
    const char *          etag;
};

const llama_ui_asset * llama_ui_find_asset(const char * name);
