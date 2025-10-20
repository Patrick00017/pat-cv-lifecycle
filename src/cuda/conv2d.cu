#include <stdio.h>
#include <cuda_runtime.h>

#define I 10
#define J 10
#define F_SIZE 3
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))

void __global__ conv2d(float* a, float* f, float* p){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int a_c_x = col;
    int a_c_y = row;
    int f_c_x = 1;
    int f_c_y = 1;
    float pvalue = 0.0f;
    for(int i=-1;i<F_SIZE-1;i++){
        for(int j=-1;j<F_SIZE-1;j++){
            if(a_c_x + i < 0 || a_c_x + i >= I || a_c_y + j < 0 || a_c_y + j >= J)
                pvalue += 0.0;
            else
                pvalue += a[(a_c_y + j)*I + a_c_x + i] * f[(f_c_y + j)*F_SIZE + f_c_x + i];
        }
    }
    p[row*I + col] = pvalue;
}

int main(int argc, char* argv[]){
    // float A[WIDTH][WIDTH];
    // float B[WIDTH][WIDTH];
    // float P[WIDTH][WIDTH];
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

    cudaMalloc((void**)&d_a, sizeof(float) * I * J);
    cudaMalloc((void**)&d_f, sizeof(float) * F_SIZE * F_SIZE);
    cudaMalloc((void**)&d_p, sizeof(float) * I * J);
    cudaMemcpy(d_a, A, sizeof(float) * I * J, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, F, sizeof(float) * F_SIZE * F_SIZE, cudaMemcpyHostToDevice);

    dim3 block(10, 10, 1);
    dim3 grid(1, 1, 1);
    cudaEventRecord(start);
    conv2d<<<grid, block>>>(d_a, d_f, d_p);
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

    cudaFree(d_a);
    cudaFree(d_f);
    cudaFree(d_p);
    free(A);
    free(F);
    free(P);
}