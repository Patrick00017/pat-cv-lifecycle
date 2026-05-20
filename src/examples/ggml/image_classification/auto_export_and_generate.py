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

    def generate_ggml_code(self):
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

static struct ggml_cgraph* build_graph(struct ggml_context* ctx_cgraph, const icls_model& model) {
    struct ggml_cgraph* gf = ggml_new_graph(ctx_cgraph);
    struct ggml_tensor* input = ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, model.input_width, model.input_height, 3, 1);
    ggml_set_name(input, "input");
    struct ggml_tensor* result = apply_conv2d(ctx_cgraph, input, model.backbone.conv1);
    print_shape(0, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    print_shape(1, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer1_0, true);
    print_shape(10, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer1_1, false);
    print_shape(11, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer1_2, false);
    print_shape(12, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer2_0, true);
    print_shape(20, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer2_1, false);
    print_shape(21, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer2_2, false);
    print_shape(22, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer2_3, false);
    print_shape(23, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer3_0, true);
    print_shape(30, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer3_1, false);
    print_shape(31, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer3_2, false);
    print_shape(32, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer3_3, false);
    print_shape(33, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer3_4, false);
    print_shape(34, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer3_5, false);
    print_shape(35, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer4_0, true);
    print_shape(40, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer4_1, false);
    print_shape(41, result);
    result = apply_block(ctx_cgraph, result, model.backbone.layer4_2, false);
    print_shape(42, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_AVG, 7, 7, 1, 1, 0, 0); // todo: adaptive avgpool
    print_shape(50, result); // 1 * 1 * 2048 * 1
    result = ggml_reshape_4d(ctx_cgraph, result, result->ne[2], 1, 1, 1);
    print_shape(51, result);
    result = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, model.backbone.fc_weight, result), model.backbone.fc_bias);
    print_shape(52, result);

    // start classifer
    result = ggml_relu(ctx_cgraph, ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, model.backbone.classifer.fc1_weight, result), model.backbone.classifer.fc1_bias));
    print_shape(60, result);
    result = ggml_relu(ctx_cgraph, ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, model.backbone.classifer.fc2_weight, result), model.backbone.classifer.fc2_bias));
    print_shape(61, result);
    result = ggml_relu(ctx_cgraph, ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, model.backbone.classifer.fc3_weight, result), model.backbone.classifer.fc3_bias));
    print_shape(62, result);
    result = ggml_soft_max(ctx_cgraph, result);

    ggml_set_output(result);
    ggml_set_name(result, "output");

    ggml_build_forward_expand(gf, result);

    return gf;
}
        """

        # 1. load all graph node names and state dict names
        graph_weight_names = []  # use to save the node name in the graph
        state_dict_weight_names = []  # use to save the weight name in the state dict
        for input_spec in self.input_specs:
            print(input_spec.arg.name)  # weight name will be used in the graph
            print(input_spec.target)  # state_dict weight name
            graph_weight_names.append(input_spec.arg.name)
            state_dict_weight_names.append(input_spec.target)

        # 2. generate the model structure
        ggml_model_weight_names_code = ""
        for param_name in graph_weight_names:
            ggml_model_weight_names_code += f"\tstruct ggml_tensor* {param_name};\t\n"

        model_structure_code = f"""
struct custom_model {{
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context *ctx;
    
    // generated weight names
    {ggml_model_weight_names_code}
}};
        """

        # 3. load weight from state dict
        weight_load_process_code = ""
        for index, state_dict_name in enumerate(state_dict_weight_names):
            ggml_model_weight_name = graph_weight_names[index]
            weight_load_process_code += f"""model.{ggml_model_weight_name} = ggml_get_tensor(model.ctx, "{state_dict_name}");\t\n"""

        # 4. load to the total code
        ggml_startup_template = Template(ggml_code_startup_template_str)
        ggml_startup_code = ggml_startup_template.substitute(
            model_struct=model_structure_code,
            weight_load_process=weight_load_process_code,
        )

        # 5. create compute graph
        self.ggml_code = ggml_startup_code

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
        self.generate_ggml_code()


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    input_args = (torch.randn(1, 3, 256, 256),)
    tool = AutoGenerateUtils(model=model, input_args=input_args)
    # tool.check_input()
    tool.check_graph()
