#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 1000
#define TILE_WIDTH 10
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))

void __global__ matmul(float* a, float* b, float* p){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("%d %d\n", row, col);
    float pvalue = 0.0f;
    for(int i=0;i<WIDTH;i++){
        pvalue += a[row*WIDTH + i] * b[i*WIDTH + col];
    }
    p[row*WIDTH + col] = pvalue;
}

void __global__ matmul_v1(float* a, float* b, float* p, int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("%d %d\n", row, col);
    float pvalue = 0.0f;
    for(int i=0;i<width;i++){
        pvalue += a[row*width + i] * b[i*width + col];
    }
    p[row*width + col] = pvalue;
}

void __global__ matmul_v2(float* a, float* b, float* p){
    __shared__ float mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // block local thread offset position
    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;

    // printf("%d %d\n", row, col);
    float pvalue = 0.0f;
    for(int ph=0;ph<WIDTH/TILE_WIDTH;ph++){
        // copy value
        mds[ty][tx] = a[row*WIDTH+ph*TILE_WIDTH+tx];
        nds[ty][tx] = b[(ph*TILE_WIDTH+ty)*WIDTH+col];
        __syncthreads();
        
        for(int k=0;k<TILE_WIDTH;k++){
            pvalue += mds[ty][k] * nds[k][tx];
        }
        __syncthreads();
    }
    p[row*WIDTH + col] = pvalue;
}



int main(int argc, char* argv[]){
    // float A[WIDTH][WIDTH];
    // float B[WIDTH][WIDTH];
    // float P[WIDTH][WIDTH];
    float* A = (float*)malloc(sizeof(float) * WIDTH * WIDTH);
    float* B = (float*)malloc(sizeof(float) * WIDTH * WIDTH);
    float* P = (float*)malloc(sizeof(float) * WIDTH * WIDTH);
    float* d_a;
    float* d_b;
    float* d_p;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float value = 0.001f;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            A[i*WIDTH + j] = value;
            B[i*WIDTH + j] = value;
            value += 0.001f;
            // printf("%f %f\n", A[i*WIDTH + j], B[i*WIDTH + j]);
        }
    }
    cudaMalloc((void**)&d_a, sizeof(float) * WIDTH * WIDTH);
    cudaMalloc((void**)&d_b, sizeof(float) * WIDTH * WIDTH);
    cudaMalloc((void**)&d_p, sizeof(float) * WIDTH * WIDTH);
    cudaMemcpy(d_a, A, sizeof(float) * WIDTH * WIDTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float) * WIDTH * WIDTH, cudaMemcpyHostToDevice);

    dim3 block(20, 20, 1);
    dim3 grid(DIV_ROUND_UP(WIDTH, 20), DIV_ROUND_UP(WIDTH, 20), 1);
    cudaEventRecord(start);
    matmul<<<grid, block>>>(d_a, d_b, d_p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("核函数执行时间: %.3f ms\n", kernelTime);

    cudaMemcpy(P, d_p, sizeof(float) * WIDTH * WIDTH, cudaMemcpyDeviceToHost);
    // for(int i=0;i<WIDTH;i++){
    //     for(int j=0;j<WIDTH;j++){
    //         printf("%f ", P[i*WIDTH + j]);
    //     }
    //     printf("\n");
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_p);
    free(A);
    free(B);
    free(P);
}