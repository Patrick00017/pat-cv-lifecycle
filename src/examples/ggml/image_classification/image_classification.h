#define _USE_MATH_DEFINES // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>
#include <iostream>


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct icls_hparams {
    // define the hparams for icls model
    // int32_t num_hidden = 128;
    // int32_t num_classes = 2;

    int32_t ftype = GGML_FTYPE_ALL_F32; // no quantization
};

struct conv2d_layer {
    struct ggml_tensor * weights;

    struct ggml_tensor * bn_weight;
    struct ggml_tensor * bias;
    struct ggml_tensor * running_mean;
    struct ggml_tensor * running_var;
    int stride = 1;
    int padding = 0;
    bool batch_normalize = true;
    bool activate = false; // true for relu, false for linear
};

struct icls_cls_head {
    // ("fc1", nn.Linear(1024, 512)),
    // ("relu1", nn.ReLU()),
    // ("fc2", nn.Linear(512, 256)),
    // ("relu2", nn.ReLU()),
    // ("fc3", nn.Linear(256, num_classes)),
    // ("output", nn.LogSoftmax(dim=1)),
    struct ggml_tensor* fc1_weight;
    struct ggml_tensor* fc1_bias;
    struct ggml_tensor* fc2_weight;
    struct ggml_tensor* fc2_bias;
    struct ggml_tensor* fc3_weight;
    struct ggml_tensor* fc3_bias;
};

struct icls_backbone {
    conv2d_layer conv1;

    std::vector<conv2d_layer> layer1_0;
    std::vector<conv2d_layer> layer1_1;
    std::vector<conv2d_layer> layer1_2;

    std::vector<conv2d_layer> layer2_0;
    std::vector<conv2d_layer> layer2_1;
    std::vector<conv2d_layer> layer2_2;
    std::vector<conv2d_layer> layer2_3;

    std::vector<conv2d_layer> layer3_0;
    std::vector<conv2d_layer> layer3_1;
    std::vector<conv2d_layer> layer3_2;
    std::vector<conv2d_layer> layer3_3;
    std::vector<conv2d_layer> layer3_4;
    std::vector<conv2d_layer> layer3_5;

    std::vector<conv2d_layer> layer4_0;
    std::vector<conv2d_layer> layer4_1;
    std::vector<conv2d_layer> layer4_2;

    struct ggml_tensor* fc_weight;
    struct ggml_tensor* fc_bias;

    icls_cls_head classifer;
};

struct icls_model {
    int input_width = 224;
    int input_height = 224;

    icls_backbone backbone;
    icls_cls_head head;

    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context *ctx;
};

struct icls_params {
    std::string model = "D:/code/forked_project/ggml/examples/image_classification/weights/icls.gguf";
    std::string fname_inp = "D:/code/forked_project/ggml/examples/image_classification/data/test/2.jpg";
    std::string fname_out = "predictions.jpg";
    int         n_threads = std::max(1U, std::thread::hardware_concurrency() / 2);
    std::string device;
};

// load the model's weights from a file
bool icls_model_init_from_file(const std::string &fname, icls_model &model);


static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

struct icls_image {
    int w, h, c;
    std::vector<float> data;

    icls_image() : w(0), h(0), c(0) {}
    icls_image(int w, int h, int c) : w(w), h(h), c(c), data(w* h* c) {}

    float get_pixel(int x, int y, int c) const {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        return data[c * w * h + y * w + x];
    }

    void set_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c * w * h + y * w + x] = val;
    }

    void add_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c * w * h + y * w + x] += val;
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }
};

bool load_image(const char* fname, icls_image& img);
void draw_box_width(icls_image& a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
icls_image letterbox_image(const icls_image& im, int w, int h);
bool save_image(const icls_image& im, const char* name, int quality);
icls_image get_label(const std::vector<icls_image>& alphabet, const std::string& label, int size);
void draw_label(icls_image& im, int row, int col, const icls_image& label, const float* rgb);

static void draw_box(icls_image& a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (int i = x1; i <= x2; ++i) {
        a.data[i + y1 * a.w + 0 * a.w * a.h] = r;
        a.data[i + y2 * a.w + 0 * a.w * a.h] = r;

        a.data[i + y1 * a.w + 1 * a.w * a.h] = g;
        a.data[i + y2 * a.w + 1 * a.w * a.h] = g;

        a.data[i + y1 * a.w + 2 * a.w * a.h] = b;
        a.data[i + y2 * a.w + 2 * a.w * a.h] = b;
    }
    for (int i = y1; i <= y2; ++i) {
        a.data[x1 + i * a.w + 0 * a.w * a.h] = r;
        a.data[x2 + i * a.w + 0 * a.w * a.h] = r;

        a.data[x1 + i * a.w + 1 * a.w * a.h] = g;
        a.data[x2 + i * a.w + 1 * a.w * a.h] = g;

        a.data[x1 + i * a.w + 2 * a.w * a.h] = b;
        a.data[x2 + i * a.w + 2 * a.w * a.h] = b;
    }
}

void draw_box_width(icls_image& a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    for (int i = 0; i < w; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

bool save_image(const icls_image& im, const char* name, int quality)
{
    uint8_t* data = (uint8_t*)calloc(im.w * im.h * im.c, sizeof(uint8_t));
    for (int k = 0; k < im.c; ++k) {
        for (int i = 0; i < im.w * im.h; ++i) {
            data[i * im.c + k] = (uint8_t)(255 * im.data[i + k * im.w * im.h]);
        }
    }
    int success = stbi_write_jpg(name, im.w, im.h, im.c, data, quality);
    free(data);
    if (!success) {
        fprintf(stderr, "Failed to write image %s\n", name);
        return false;
    }
    return true;
}

bool load_image(const char* fname, icls_image& img)
{
    int w, h, c;
    uint8_t* data = stbi_load(fname, &w, &h, &c, 3);
    if (!data) {
        return false;
    }
    c = 3;
    img.w = w;
    img.h = h;
    img.c = c;
    img.data.resize(w * h * c);
    for (int k = 0; k < c; ++k) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                int dst_index = i + w * j + w * h * k;
                int src_index = k + c * i + c * w * j;
                img.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    stbi_image_free(data);
    return true;
}

static icls_image resize_image(const icls_image& im, int w, int h)
{
    icls_image resized(w, h, im.c);
    icls_image part(w, im.h, im.c);
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (int k = 0; k < im.c; ++k) {
        for (int r = 0; r < im.h; ++r) {
            for (int c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = im.get_pixel(im.w - 1, r, k);
                }
                else {
                    float sx = c * w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * im.get_pixel(ix, r, k) + dx * im.get_pixel(ix + 1, r, k);
                }
                part.set_pixel(c, r, k, val);
            }
        }
    }
    for (int k = 0; k < im.c; ++k) {
        for (int r = 0; r < h; ++r) {
            float sy = r * h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (int c = 0; c < w; ++c) {
                float val = (1 - dy) * part.get_pixel(c, iy, k);
                resized.set_pixel(c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (int c = 0; c < w; ++c) {
                float val = dy * part.get_pixel(c, iy + 1, k);
                resized.add_pixel(c, r, k, val);
            }
        }
    }
    return resized;
}

static void embed_image(const icls_image& source, icls_image& dest, int dx, int dy)
{
    for (int k = 0; k < source.c; ++k) {
        for (int y = 0; y < source.h; ++y) {
            for (int x = 0; x < source.w; ++x) {
                float val = source.get_pixel(x, y, k);
                dest.set_pixel(dx + x, dy + y, k, val);
            }
        }
    }
}

icls_image letterbox_image(const icls_image& im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w / im.w) < ((float)h / im.h)) {
        new_w = w;
        new_h = (im.h * w) / im.w;
    }
    else {
        new_h = h;
        new_w = (im.w * h) / im.h;
    }
    icls_image resized = resize_image(im, new_w, new_h);
    icls_image boxed(w, h, im.c);
    boxed.fill(0.5);
    embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
    return boxed;
}

static icls_image tile_images(const icls_image& a, const icls_image& b, int dx)
{
    if (a.w == 0) {
        return b;
    }
    icls_image c(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, a.c);
    c.fill(1.0f);
    embed_image(a, c, 0, 0);
    embed_image(b, c, a.w + dx, 0);
    return c;
}

static icls_image border_image(const icls_image& a, int border)
{
    icls_image b(a.w + 2 * border, a.h + 2 * border, a.c);
    b.fill(1.0f);
    embed_image(a, b, border, border);
    return b;
}

icls_image get_label(const std::vector<icls_image>& alphabet, const std::string& label, int size)
{
    size = size / 10;
    size = std::min(size, 7);
    icls_image result(0, 0, 0);
    for (int i = 0; i < (int)label.size(); ++i) {
        int ch = label[i];
        icls_image img = alphabet[size * 128 + ch];
        result = tile_images(result, img, -size - 1 + (size + 1) / 2);
    }
    return border_image(result, (int)(result.h * .25));
}

void draw_label(icls_image& im, int row, int col, const icls_image& label, const float* rgb)
{
    int w = label.w;
    int h = label.h;
    if (row - h >= 0) {
        row = row - h;
    }
    for (int j = 0; j < h && j + row < im.h; j++) {
        for (int i = 0; i < w && i + col < im.w; i++) {
            for (int k = 0; k < label.c; k++) {
                float val = label.get_pixel(i, j, k);
                im.set_pixel(i + col, j + row, k, rgb[k] * val);
            }
        }
    }
}
