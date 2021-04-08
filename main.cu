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

float* params[] = {w1, b1, w2, b2, w3, b3};
float* hparams[] = {hw1, hb1, hw2, hb2, hw3, hb3};
size_t* pitches[] = {&pitch_w1, &pitch_b1, &pitch_w2, &pitch_b2, &pitch_w3, &pitch_b3};

int max_indices[] = {INPUT_SIZE * LAYER_SIZE_1,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2 + LAYER_SIZE_2,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2 + LAYER_SIZE_2 + LAYER_SIZE_2 * LAYER_SIZE_3,
                     INPUT_SIZE * LAYER_SIZE_1 + LAYER_SIZE_1 + LAYER_SIZE_1 * LAYER_SIZE_2 + LAYER_SIZE_2 + LAYER_SIZE_2 * LAYER_SIZE_3 + LAYER_SIZE_3};

int param_sizes[][2] = {{LAYER_SIZE_1, INPUT_SIZE},
                        {LAYER_SIZE_1, 1},
                        {LAYER_SIZE_2, LAYER_SIZE_1},
                        {LAYER_SIZE_2, 1},
                        {LAYER_SIZE_3, LAYER_SIZE_2},
                        {LAYER_SIZE_3, 1}};

// Processing stages

float *stages[10], *hstages[10], *sum;
size_t pitch_stages[10];

int stage_sizes[][2] = {{INPUT_SIZE, 1},
                        {LAYER_SIZE_1, 1},
                        {LAYER_SIZE_1, 1},
                        {LAYER_SIZE_1, 1},
                        {LAYER_SIZE_2, 1},
                        {LAYER_SIZE_2, 1},
                        {LAYER_SIZE_2, 1},
                        {LAYER_SIZE_3, 1},
                        {LAYER_SIZE_3, 1},
                        {LAYER_SIZE_3, 1}};

void initialize_weights() {

    int findex;
    int pindex;

    for (pindex = 0; pindex < 6; pindex++) {
        size_t malloc_size = ((pindex == 0) ? max_indices[0] : max_indices[pindex] - max_indices[pindex - 1]) * sizeof(float);
        cudaMallocPitch(&params[pindex], pitches[pindex], param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][0]);
        hparams[pindex] = (float *) malloc(malloc_size);
    }
    pindex = 0;

    FILE *fp;
    fp = fopen("model.csv", "r");

    for (findex = 0; findex < max_indices[5]; findex++) {
        if (findex >= max_indices[pindex]) pindex++;
        float *value = hparams[pindex];
        fscanf(fp, "%f,", &value[findex - ((pindex == 0) ? 0 : max_indices[pindex - 1])]);
    }

    fclose(fp);

    for (pindex = 0; pindex < 6; pindex++) {
        cudaMemcpy2D(params[pindex], *pitches[pindex], hparams[pindex], param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][0], cudaMemcpyHostToDevice);
    }

}

void initialize_stages() {

    int sindex;

    for (sindex = 0; sindex < 10; sindex++) {
        cudaMallocPitch(&stages[sindex], &pitch_stages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0]);
        hstages[sindex] = (float *) calloc(stage_sizes[sindex][1] * stage_sizes[sindex][0], sizeof(float));
    }

    cudaMalloc(&sum, sizeof(float));

}

void reset_stages() {

    int sindex;

    for (sindex = 0; sindex < 10; sindex++) {
        cudaMemcpy2D(stages[sindex], pitch_stages[sindex], hstages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0], cudaMemcpyHostToDevice);
    }

    cudaMemset(sum, 0, 1);

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
    /*
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float *) malloc(A_R * A_C * sizeof(float));
    b = (float *) malloc(B_R * B_C * sizeof(float));
    c = (float *) malloc(A_R * B_C * sizeof(float));

    memcpy(a, A, A_R * A_C * sizeof(float));
    memcpy(b, A, B_R * B_C * sizeof(float));

    size_t pitch_a;
    size_t pitch_b;
    size_t pitch_c;

    cudaMallocPitch(&d_a, &pitch_a, A_C * sizeof(float), A_R);
    cudaMallocPitch(&d_b, &pitch_b, B_C * sizeof(float), B_R);
    cudaMallocPitch(&d_c, &pitch_c, B_C * sizeof(float), A_R);

    cudaMemcpy2D(d_a, pitch_a, a, A_C * sizeof(float), A_C * sizeof(float), A_R, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitch_b, b, B_C * sizeof(float), B_C * sizeof(float), B_R, cudaMemcpyHostToDevice);

    expsum<<<1, 64>>>(d_a, d_b, pitch_a, A_R, A_C);
    cudaDeviceSynchronize();
    softmax<<<1, 64>>>(d_a, d_c, d_b, pitch_a, pitch_c, A_R, A_C);

    gpuErrchk(cudaMemcpy2D(c, B_C * sizeof(float), d_c, pitch_c, B_C * sizeof(float), A_R, cudaMemcpyDeviceToHost));

    for (int i = 0; i < A_R * B_C; i++) {
        printf("%f\n", c[i]);
    }
    */

    initialize_weights();

    return 0;
}
