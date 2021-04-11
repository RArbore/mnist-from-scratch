#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

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

#define LOAD_MODEL 0

#define DATA_SIZE 100

#define RAND_GRANULARITY 1000

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
void max(float *a, float *b, size_t pitch_a, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        int row, col;
        for (row = 0; row < a_r; row++) {
            for (col = 0; col < a_c; col++) {
                float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
                if ((row == 0 && col == 0) || *b < *elem_a) *b = *elem_a;
            }
        }
    }
}

__global__
void expsum(float *a, float *b, float *maxf, size_t pitch_a, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * a_c) {
        int row = i / a_c;
        int col = i % a_c;
        float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
        atomicAdd(b, exp(*elem_a - *maxf));
    }
}

__global__
void softmax(float *a, float *b, float *sum, float *maxf, size_t pitch_a, size_t pitch_b, size_t a_r, size_t a_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_r * a_c) {
        int row = i / a_c;
        int col = i % a_c;
        float *elem_a = (float *)(((char *) a) + row * pitch_a + col * sizeof(float));
        float *elem_b = (float *)(((char *) b) + row * pitch_b + col * sizeof(float));
        *elem_b = exp(*elem_a - *maxf) / *sum;
    }
}

float *params[6], *hparams[6];
size_t pitches[6];

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

float *stages[10], *hstages[10], *hout, *sum, *maxf;
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

    int findex, pindex;

    for (pindex = 0; pindex < 6; pindex++) {
        size_t malloc_size = ((pindex == 0) ? max_indices[0] : max_indices[pindex] - max_indices[pindex - 1]) * sizeof(float);
        gpuErrchk(cudaMallocPitch(&params[pindex], &pitches[pindex], param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][0]));
        hparams[pindex] = (float *) malloc(malloc_size);
    }
    pindex = 0;

    if (LOAD_MODEL) {
        FILE *fp;
        fp = fopen("model.csv", "r");

        for (findex = 0; findex < max_indices[5]; findex++) {
            if (findex >= max_indices[pindex]) pindex++;
            float *value = hparams[pindex];
            fscanf(fp, "%f,", &value[findex - ((pindex == 0) ? 0 : max_indices[pindex - 1])]);
        }

        fclose(fp);
    }
    else {
        int layer_size;
        float stdv;
        
        for (findex = 0; findex < max_indices[5]; findex++) {
            if (findex >= max_indices[pindex]) {
                pindex++;
                layer_size = param_sizes[pindex][1];
                stdv = 1.0 / sqrt((float) layer_size);
            }
            
            float *value = hparams[pindex];
            float randu = ((float) (rand() % RAND_GRANULARITY)) / RAND_GRANULARITY;
            value[findex - ((pindex == 0) ? 0 : max_indices[pindex - 1])] = randu * stdv * 2 - stdv;
        }
    }

    for (pindex = 0; pindex < 6; pindex++) {
        cudaMemcpy2D(params[pindex], pitches[pindex], hparams[pindex], param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][0], cudaMemcpyHostToDevice);
    }

}

void initialize_stages() {

    int sindex;

    for (sindex = 0; sindex < 10; sindex++) {
        gpuErrchk(cudaMallocPitch(&(stages[sindex]), &(pitch_stages[sindex]), stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0]));
        hstages[sindex] = (float *) calloc(stage_sizes[sindex][1] * stage_sizes[sindex][0], sizeof(float));

        gpuErrchk(cudaMemcpy2D(hstages[sindex], stage_sizes[sindex][1] * sizeof(float), stages[sindex], pitch_stages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0], cudaMemcpyDeviceToHost));
    }
    hout = (float *) calloc(stage_sizes[9][1] * stage_sizes[9][0], sizeof(float));
    cudaMalloc(&sum, sizeof(float));
    cudaMalloc(&maxf, sizeof(float));

}

void reset_stages() {

    int sindex;

    for (sindex = 0; sindex < 10; sindex++) {
        cudaMemcpy2D(stages[sindex], pitch_stages[sindex], hstages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0], cudaMemcpyHostToDevice);
    }

    cudaMemset(sum, 0, sizeof(float));
    cudaMemset(maxf, 0, sizeof(float));

}

void inference() {

    matmul<<<16, 1>>>(params[0], stages[0], stages[1], pitches[0], pitch_stages[0], pitch_stages[1], 16, 784, 784, 1);
    matadd<<<16, 1>>>(stages[1], params[1], stages[2], pitch_stages[1], pitches[1], pitch_stages[2], 16, 1);
    leakyrelu<<<16, 1>>>(stages[2], stages[3], 0.2, pitch_stages[2], pitch_stages[3], 16, 1);

    matmul<<<16, 1>>>(params[2], stages[3], stages[4], pitches[2], pitch_stages[3], pitch_stages[4], 16, 16, 16, 1);
    matadd<<<16, 1>>>(stages[4], params[3], stages[5], pitch_stages[4], pitches[3], pitch_stages[5], 16, 1);
    leakyrelu<<<16, 1>>>(stages[5], stages[6], 0.2, pitch_stages[5], pitch_stages[6], 16, 1);

    matmul<<<10, 1>>>(params[4], stages[6], stages[7], pitches[4], pitch_stages[6], pitch_stages[7], 10, 16, 16, 1);
    matadd<<<10, 1>>>(stages[7], params[5], stages[8], pitch_stages[7], pitches[5], pitch_stages[8], 10, 1);

    max<<<1, 1>>>(stages[8], maxf, pitch_stages[8], 10, 1);
    expsum<<<10, 1>>>(stages[8], sum, maxf, pitch_stages[8], 10, 1);
    softmax<<<10, 1>>>(stages[8], stages[9], sum, maxf, pitch_stages[8], pitch_stages[9], 10, 1);

    int sindex, nindex;
    for (sindex = 9; sindex < 10; sindex++) {
        gpuErrchk(cudaMemcpy2D(hstages[sindex], stage_sizes[sindex][1] * sizeof(float), stages[sindex], pitch_stages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0], cudaMemcpyDeviceToHost));
        for (nindex = 0; nindex < stage_sizes[sindex][0]; nindex++) {
            printf("%f ", hstages[sindex][nindex]);
        }
        printf("\n");
    }

}

float *images[DATA_SIZE], *labels;
float *himages[DATA_SIZE], *hlabels;
size_t pitch_images[DATA_SIZE];

void load_data() {

    int iindex, pindex;

    for (iindex = 0; iindex < DATA_SIZE; iindex++) {
        cudaMallocPitch(&images[iindex], &pitch_images[iindex], stage_sizes[0][1] * sizeof(float), stage_sizes[0][0]);
        himages[iindex] = (float *) malloc(stage_sizes[0][1] * stage_sizes[0][0] * sizeof(float));

    }

    cudaMalloc(&labels, DATA_SIZE * sizeof(float));
    hlabels = (float *) malloc(DATA_SIZE * sizeof(float));

    FILE *fi, *fl;
    fi = fopen("images.csv", "r");
    fl = fopen("labels.csv", "r");

    for (iindex = 0; iindex < DATA_SIZE; iindex++) {
        for (pindex = 0; pindex < INPUT_SIZE; pindex++) {
            fscanf(fi, "%f,", himages[iindex] + pindex);
        }
        fscanf(fl, "%f,", hlabels + iindex);
    }

    fclose(fi);
    fclose(fl);

    for (iindex = 0; iindex < DATA_SIZE; iindex++) {
        cudaMemcpy2D(images[iindex], pitch_images[iindex], himages[iindex], stage_sizes[0][1] * sizeof(float), stage_sizes[0][1] * sizeof(float), stage_sizes[0][0], cudaMemcpyHostToDevice);
    }
    cudaMemcpy(labels, hlabels, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

}

int main (int argc, char *argv[]) {

    time_t t;
    srand((unsigned) time(&t));
    initialize_weights();
    initialize_stages();
    load_data();
    int i;
    for (i = 0; i < DATA_SIZE; i++) {
        reset_stages();
        gpuErrchk(cudaMemcpy2D(stages[0], pitch_stages[0], images[i], pitch_images[i], stage_sizes[0][1] * sizeof(float), stage_sizes[0][0], cudaMemcpyDeviceToDevice));
        inference();
    }

    return 0;
}
