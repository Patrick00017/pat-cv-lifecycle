#include <stdio.h>
#include <cuda_runtime.h>

#define I 10
#define J 10
// #define F_SIZE 3
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))

void __global__ conv2d(float* a, float* f, float* p, int r){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // change pivot to left top, and the kernel width(height) is 2*r + 1
    int a_start_x = col - r;
    int a_start_y = row - r;
    // int f_x = 0;
    // int f_y = 0;
    float pvalue = 0.0f;
    for(int i=0;i<2*r+1;i++){
        for(int j=0;j<2*r+1;j++){
            int a_x = a_start_x + i;
            int a_y = a_start_y + i;
            if(a_x < 0 || a_x >= I || a_y < 0 || a_y >= J)
                pvalue += 0.0;
            else
                pvalue += a[a_y * I + a_x] * f[j*(2*r+1) + i];
        }
    }
    p[row*I + col] = pvalue;
}

int main(int argc, char* argv[]){
    // float A[WIDTH][WIDTH];
    // float B[WIDTH][WIDTH];
    // float P[WIDTH][WIDTH];
    int r = 1;
    int F_SIZE = 2*r + 1;
    float* A = (float*)malloc(sizeof(float) * I * J);
    float* F = (float*)malloc(sizeof(float) * F_SIZE * F_SIZE);
    float* P = (float*)malloc(sizeof(float) * I * J);
    float* d_a;
    float* d_f;
    float* d_p;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float value = 1.0f;
    for(int i=0;i<I;i++){
        for(int j=0;j<J;j++){
            A[i*I + j] = value;
            // value += 1.0f;
            // printf("%f %f\n", A[i*WIDTH + j], B[i*WIDTH + j]);
        }
    }
    // assign value for filter
    value = 1.0f;
    for(int i=0;i<F_SIZE;i++){
        for(int j=0;j<F_SIZE;j++){
            F[i*I + j] = value;
            printf("%f ", F[i*I + j]);
        }
        printf("\n");
    }
    F[r] *= 5;

    cudaMalloc((void**)&d_a, sizeof(float) * I * J);
    cudaMalloc((void**)&d_f, sizeof(float) * F_SIZE * F_SIZE);
    cudaMalloc((void**)&d_p, sizeof(float) * I * J);
    cudaMemcpy(d_a, A, sizeof(float) * I * J, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, F, sizeof(float) * F_SIZE * F_SIZE, cudaMemcpyHostToDevice);

    dim3 block(10, 10, 1);
    dim3 grid(1, 1, 1);
    cudaEventRecord(start);
    conv2d<<<grid, block>>>(d_a, d_f, d_p, r);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("核函数执行时间: %.3f ms\n", kernelTime);

    cudaMemcpy(P, d_p, sizeof(float) * I * J, cudaMemcpyDeviceToHost);
    for(int i=0;i<I;i++){
        for(int j=0;j<J;j++){
            printf("%f ", P[i*I + j]);
        }
        printf("\n");
    }
    free(A);
    free(F);
    // free(P);
    cudaFree(d_a);
    cudaFree(d_f);
    cudaFree(d_p);
}