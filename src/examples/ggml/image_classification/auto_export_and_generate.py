import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
from torch.export.graph_signature import InputSpec, OutputSpec
from torchvision import datasets, transforms, models
from string import Template

aten_op_dict = []


class AutoGenerateUtils:
    def __init__(self, model: nn.Module, input_args):

        self.exported_program: ExportedProgram = export(model, args=input_args)
        self.graph_signature = self.exported_program.graph_signature
        self.graph = self.exported_program.graph
        self.input_specs: list[InputSpec] = self.graph_signature.input_specs
        self.output_specs: list[OutputSpec] = self.graph_signature.output_specs
        self.ggml_code: str = ""
        self.init_ggml_code()

    def init_ggml_code(self):
        """this function is used to generate some useful tools for next generation."""
        ggml_code_startup_template_str = """
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

${model_struct}

bool model_init_from_file(const std::string &fname, custom_model &model){
    struct ggml_context * tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    int num_tensors = gguf_get_n_tensors(gguf_ctx);
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(params);
    for (int i = 0; i < num_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor * dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);

        // add some log
        std::cout << "src: " << src->name << " dst: " << dst->name << std::endl;
    }
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor * cur = ggml_get_first_tensor(model.ctx); cur != NULL; cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    
    // load model.ctx tensors into backbone
    ${weight_load_process}
    
    return true;
}
        """
        ggml_startup_template = Template(ggml_code_startup_template_str)
        print(
            ggml_startup_template.substitute(
                model_struct="aaa\n", weight_load_process="bbb\n"
            )
        )

    def check_input(self):
        for i in range(5):
            print(self.input_specs[i].kind)  # parameter or buffer, the same in ggml?
            print(self.input_specs[i].arg)  # .name is the name in exported graph
            print(self.input_specs[i].target)  # weight name in state_dict
            # persistent=True means the buffer is saved to and loaded from the model's state_dict() when saving/loading checkpoints. persistent=False buffers (like num_batches_tracked) are not saved - they are recalculated at runtime.
            print(self.input_specs[i].persistent)

    def check_graph(self):
        # print(self.graph)
        print(self.graph.__str__)
        for node in self.graph.nodes:
            # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
            # print(node.op)  # call_function
            # print(node.name)  # node name
            # print(node.target)  # function name to be called, like aten.conv2d.default
            # print(node.type)  # None
            # print(
            # node.all_input_nodes
            # )  # a dict like {relu__47: None, p_layer4_2_conv3_weight: None}
            # print(node.kwargs)
            print(f"""
Node:
    name: {node.name},
    op: {node.op},
    target: {node.target},
    type: {node.type},
    input_nodes: {node.all_input_nodes},
    args: {node.args}
    kwargs: {node.kwargs}
""")

    def generate(self):
        pass


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    input_args = (torch.randn(1, 3, 256, 256),)
    tool = AutoGenerateUtils(model=model, input_args=input_args)
    # tool.check_input()
    # tool.check_graph()
