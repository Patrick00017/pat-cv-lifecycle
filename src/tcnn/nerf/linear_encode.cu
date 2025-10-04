#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <vector>

using namespace tcnn;

template <typename T, uint32_t WIDTH>
class FeedForwardNetwork: public Network<T> {
public:
    FeedForwardNetwork(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers, Activation activation, Activation output_activation);

#if !define(TCNN_NO_FWD_BWD)
    void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override;    
    std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override;  
    
    void backward_impl(
        cudaStream_t stream,
        const Context& ctx,
        const GPUMatrixDynamic<T>& input,
        const GPUMatrixDynamic<T>& output,
        const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<T>* dL_dinput = nullptr,
        bool use_inference_params = false,
        GradientMode params_gradients_mode = GradientMode::Overwrite
    ) override;
#endif
    void set_params_impl(T* params, T* inference_params, T* gradients) override;
    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;
    GPUMatrix<T, RM>& input_weight_matrix(bool inference){
        auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
        return weight_matrices.front()
    } 
    GPUMatrix<T, RM>& weight_matrix_at(bool inference, uint32_t idx){
        auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
        return weight_matrices.at(1 + idx);
    }
    GPUMatrix<T, RM>& output_weight_matrix(bool inference, uint32_t idx){
        auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
        return weight_matrices.back();
    }
    GPUMatrix<T, RM>& input_gradient_matrix(){
        return m_gradient_matrices.front();
    }
    GPUMatrix<T, RM>& gradient_matrix_at(uint32_t idx){
        return m_gradient_matrices.at(1 + idx);
    }
    GPUMatrix<T, RM>& output_gradient_matrix(){
        return m_gradient_matrices.back();
    }
    size_t n_params() const override {
        return m_total_n_params;
    }
    uint32_t input_width() const override {
        return m_input_width;
    }
    uint32_t padded_output_width() const override {
        return m_padded_output_width;
    }
    uint32_t output_width() const override {
        return m_output_width;
    }
    static uint32_t REQUIRED_ALIGNMENT() {
        return 16; // use 16x16x16 tensor ops
    }
    uint32_t required_input_alignment() const override {
        return REQUIRED_ALIGNMENT();
    }
    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
        std::vector<std::pair<uint32_t, uint32_t>> result;
        for(auto& matrix : m_weight_matrices){
            result.emplace_back(matrix.m(), matrix.n())
        }
        return result;
    }
    uint32_t width(uint32_t layer) const override {
        return WIDTH;
    }
    uint32_t num_forward_activations() const override {
        return m_n_hidden_layers;
    }
    std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
        return {forward.hidden.at(layer).data(), CM};
    }
    
private:
    std::unique_ptr<CudaRtcKernel> m_convert_params_to_jit_layout_kernel;
    std::unique_ptr<CudaRtcKernel> m_convert_params_from_jit_layout_kernel;
    struct ForwardContext : public Context {
        std::vector<GPUMatrix<T>> hidden;
        GPUMemoryArena::Allocation alloc;
    }
    std::unique_ptr<ForwardContext> allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size);

    uint32_t m_n_hidden_layers;
    uint32_t m_n_hidden_matmuls;
    uint32_t m_input_width;
    uint32_t m_network_width;
    uint32_t m_output_width;
    uint32_t m_padded_output_width;

    Activation m_activation;
    Activation m_output_activation;

    std::vector<GPUMatrix<T, RM>> m_weight_matrices;
    std::vector<GPUMatrix<T, RM>> m_weight_matrices_inference;
    size_t m_total_n_params;

    std::vector<GPUMatrix<T, RM>> m_gradient_matrices;
}   