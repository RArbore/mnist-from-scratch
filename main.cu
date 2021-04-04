#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define INPUT_SIZE 784
#define LAYER_SIZE_1 16
#define LAYER_SIZE_2 16
#define LAYER_SIZE_3 10

__global__
void matmul(float *a, float *b, float *c, size_t pitch_a, size_t pitch_b, size_t pitch_c, size_t a_r, size_t a_c, size_t b_r, size_t b_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * b_c) {
        int row = i / b_c;
        int col = i % b_c;
        float *elem_c = (float *)(((char *) c) + row * pitch_c + col * sizeof(float));
        for (int iter = 0; iter < a_c; iter++) {
            float *elem_a = (float *)(((char *) a) + row * pitch_a + iter * sizeof(float));
            float *elem_b = (float *)(((char *) b) + iter * pitch_b + col * sizeof(float));
            *(elem_c) += (*(elem_a)) * (*(elem_b));
        }
    }
}

__global__
void matadd(float *a, float *b, float *c, size_t pitch_a, size_t pitch_b, size_t pitch_c, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * a_c) {
        int row = i / a_c;
        int col = i % a_c;
        float *elem_c = (float *)(((char *) c) + row * pitch_c + col * sizeof(float));
        float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
        float *elem_b = (float *)(((char *) b) + row * pitch_b + col * sizeof(float));
        *(elem_c) = (*(elem_a)) + (*(elem_b));
    }
}

__global__
void leakyrelu(float *a, float *b, float mult, size_t pitch_a, size_t pitch_b, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * a_c) {
        int row = i / a_c;
        int col = i % a_c;
        float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
        float *elem_b = (float *)(((char *) b) + row * pitch_b + col * sizeof(float));
        *(elem_b) = (*(elem_a) > 0) ? (*(elem_a)) : mult * (*(elem_a));
    }
}

__global__
void expsum(float *a, float *b, size_t pitch_a, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * a_c) {
        int row = i / a_c;
        int col = i % a_c;
        float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
        atomicAdd(b, exp(*elem_a));
    }
}

__global__
void softmax(float *a, float *b, float *sum, size_t pitch_a, size_t pitch_b, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * a_c) {
        int row = i / a_c;
        int col = i % a_c;
        float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
        float *elem_b = (float *)(((char *) b) + row * pitch_b + col * sizeof(float));
        *elem_b = exp(*elem_a) / *sum;
    }
}

// Parameters

float *w1, *b1, *w2, *b2, *w3, *b3;
float *hw1, *hb1, *hw2, *hb2, *hw3, *hb3;
size_t pitch_w1, pitch_b1, pitch_w2, pitch_b2, pitch_w3, pitch_b3;

float* params[] = {hw1, hb1, hw2, hb2, hw3, hb3};

int max_indices[] = {INPUT_SIZE * LAYER_SIZE_1,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2 + LAYER_SIZE_2,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2 + LAYER_SIZE_2 + LAYER_SIZE_2 * LAYER_SIZE_3,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2 + LAYER_SIZE_2 + LAYER_SIZE_2 * LAYER_SIZE_3 + LAYER_SIZE_3};

// Processing stages

float *stages[10], *sum;
size_t pitch_stages[10];

void initialize_weights() {

    int findex;
    int pindex;

    for (pindex = 0; pindex < 6; pindex++) {
        params[pindex] = (float *) malloc(((pindex == 0) ? max_indices[0] : max_indices[pindex] - max_indices[pindex - 1]) * sizeof(float));
    }
    pindex = 0;

    FILE *fp;
    fp = fopen("model.csv", "r");

    for (findex = 0; findex < max_indices[5]; findex++) {
        if (findex >= max_indices[pindex]) pindex++;
        float *value = params[pindex];
        fscanf(fp, "%f,", &value[findex - ((pindex == 0) ? 0 : max_indices[pindex - 1])]);
    }

    fclose(fp);

}

void inference() {

   matmul<<<16, 1>>>(w1, stages[0], stages[1], pitch_w1, pitch_stages[0], pitch_stages[1], 16, 784, 784, 1);
   matadd<<<16, 1>>>(stages[1], b1, stages[2], pitch_stages[1], pitch_b1, pitch_stages[2], 16, 1);
   leakyrelu<<<16, 1>>>(stages[2], stages[3], 0.2, pitch_stages[2], pitch_stages[3], 16, 1);

   matmul<<<16, 1>>>(w2, stages[3], stages[4], pitch_w2, pitch_stages[3], pitch_stages[4], 16, 16, 16, 1);
   matadd<<<16, 1>>>(stages[4], b2, stages[5], pitch_stages[4], pitch_b2, pitch_stages[5], 16, 1);
   leakyrelu<<<16, 1>>>(stages[5], stages[6], 0.2, pitch_stages[5], pitch_stages[6], 16, 1);

   matmul<<<10, 1>>>(w3, stages[6], stages[7], pitch_w3, pitch_stages[6], pitch_stages[7], 10, 16, 16, 1);
   matadd<<<10, 1>>>(stages[7], b3, stages[8], pitch_stages[7], pitch_b3, pitch_stages[8], 10, 1);

   expsum<<<10, 1>>>(stages[8], sum, pitch_stages[8], 10, 1);
   softmax<<<10, 1>>>(stages[8], stages[9], sum, pitch_stages[8], pitch_stages[9], 10, 1);

}

int main (int argc, char *argv[]) {

    initialize_weights();

    return 0;
}