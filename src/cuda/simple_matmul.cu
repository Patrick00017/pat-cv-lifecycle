#include <stdio.h>

// start with (16, 16) block size
void __global__ matmul(float* a, float* b, float* c, int width){
    printf("%d | %d | %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float pvalue = 0.0f;
} 

int main(int argc, char* argv[]){
    int len = 100;
    float* A = (float *)malloc(len * sizeof(float));
    float* B = (float *)malloc(len * sizeof(float));
    float* C = (float *)malloc(len * sizeof(float));
    float* d_a;
    float* d_b;
    float* d_c;

    // malloc device memory
    cudaMalloc((void**)&d_a, len * sizeof(float));
    cudaMalloc((void**)&d_b, len * sizeof(float));
    cudaMalloc((void**)&d_c, len * sizeof(float));

    // assign values for host vector
    for(int i=0;i<len;i++){
        A[i] = i*1.0;
        B[i] = i*2.0;
    }

    // copy from host
    cudaMemcpy(d_a, A, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, len * sizeof(float), cudaMemcpyHostToDevice);

    // run the kernel
    // 1 can be convert to next_maltiply func
    addVec<<<7, 16>>>(d_a, d_b, d_c, len);

    // get value from device
    cudaMemcpy(C, d_c, len * sizeof(float), cudaMemcpyDeviceToHost);

    // print it
    for(int i=0;i<len;i++){
        printf("%f ", C[i]);
    }
    printf("\n");

    // free the memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}