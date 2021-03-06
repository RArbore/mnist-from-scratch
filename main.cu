#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
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
#define NUM_LAYERS 3
#define NUM_PARAMS 6
#define NUM_STAGES 10
#define NUM_GRADS 9

#define DATA_SIZE 1000
#define RAND_GRANULARITY 10000
#define LOAD_MODEL 0

#define NUM_EPOCHS 10000
#define LR 0.0001

float *params[NUM_PARAMS], *hparams[NUM_PARAMS];
size_t pitch_params[NUM_PARAMS];

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

int *d_param_sizes;

float *stages[NUM_STAGES], *hstages[NUM_STAGES], *last_stage, *hout, *sum, *maxf, *loss;
size_t pitch_stages[NUM_STAGES];

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

float *grads[NUM_GRADS];
size_t pitch_grads[NUM_GRADS];

int grad_sizes[][2] = {{LAYER_SIZE_1, INPUT_SIZE},
                       {LAYER_SIZE_1, 1},
                       {LAYER_SIZE_1, 1},
                       {LAYER_SIZE_2, LAYER_SIZE_1},
                       {LAYER_SIZE_2, 1},
                       {LAYER_SIZE_2, 1},
                       {LAYER_SIZE_3, LAYER_SIZE_2},
                       {LAYER_SIZE_3, 1},
                       {LAYER_SIZE_3, 1}};

float *images[DATA_SIZE], *labels;
float *himages[DATA_SIZE], *hlabels;
size_t pitch_images[DATA_SIZE];

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

__global__
void cross_entropy(float *pred, float *label, float *loss, size_t pitch_pred, size_t r, size_t c, int image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < r * c) {
        int row = i;
        int col = 0;
        float *elem_pred = (float *)(((char *) pred) + row * pitch_pred + col * sizeof(float));
        float mult_pred = *elem_pred * 0.99998 + 0.00001;
        float elem_loss = -(label[image] == i ? log(mult_pred) : log(1 - mult_pred));
        if (isinf(elem_loss)) elem_loss = 1000000.0f;
        atomicAdd(loss, elem_loss);

    }
}

__global__
void calc_grad_w(int layer, float *label, int image, int *d_param_sizes, float *z, float *pa, float *grad, float *grad_prev, size_t pitch_z, size_t pitch_pa, size_t pitch_grad, size_t pitch_grad_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_param_sizes[layer * 4 + 1] * d_param_sizes[layer * 4]) {
        int row = i / d_param_sizes[layer * 4 + 1];
        int col = i % d_param_sizes[layer * 4 + 1];

        float *grad_pt = (float *)(((char *) grad) + row * pitch_grad + col * sizeof(float));
        float *grad_prev_pt = (float *)(((char *) grad_prev) + row * pitch_grad_prev);
        float *pa_pt = (float *)(((char *) pa) + col * pitch_pa);
        float *z_pt = (float *)(((char *) z) + row * pitch_z);

        *grad_pt = *pa_pt * *grad_prev_pt;
        *grad_pt *= layer == NUM_LAYERS - 1 ? 1.0 : (*z_pt >= 0 ? 1.0 : 0.2);
    }
}

__global__
void calc_grad_x(int layer, float *label, int image, int *d_param_sizes, float *param, float *out, float *z, float *grad, float *grad_prev, size_t pitch_param, size_t pitch_out, size_t pitch_z, size_t pitch_grad, size_t pitch_grad_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_param_sizes[layer * 4 + 2]) {
        int row = i;

        float *grad_pt = (float *)(((char *) grad) + row * pitch_grad);
        float *out_pt = (float *)(((char *) out) + row * pitch_out);
        float *z_pt = (float *)(((char *) z) + row * pitch_z);

        if (layer == NUM_LAYERS - 1) {
            *grad_pt = *out_pt - (label[image] == row);
        }
        else {
            int m_row;
            float *grad_prev_pt, *param_pt;
            *grad_pt = 0;
            for (m_row = 0; m_row < d_param_sizes[layer * 4 + 6]; m_row++) {
                grad_prev_pt = (float *)(((char *) grad_prev) + m_row * pitch_grad_prev);
                param_pt = (float *)(((char *) param) + m_row * pitch_param + row * sizeof(float));
                *grad_pt += (layer == NUM_LAYERS - 2) ? (*grad_prev_pt * *param_pt) : (*grad_prev_pt * *param_pt * ((*z_pt >= 0) ? 1.0 : 0.2));
            }
        }
    }
}

__global__
void calc_grad_b(int layer, float *label, int image, int *d_param_sizes, float *param, float *z, float *grad, float *grad_prev, size_t pitch_param, size_t pitch_z, size_t pitch_grad, size_t pitch_grad_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_param_sizes[layer * 4]) {
        int row = i / d_param_sizes[layer * 4];

        float *grad_pt = (float *)(((char *) grad) + row * pitch_grad);
        float *grad_prev_pt = (float *)(((char *) grad_prev) + row * pitch_grad_prev);
        float *z_pt = (float *)(((char *) z) + row * pitch_z);

        *grad_pt = *grad_prev_pt;
        *grad_pt *= layer == NUM_LAYERS - 1 ? 1.0 : (*z_pt >= 0 ? 1.0 : 0.2);
    }
}

__global__
void grad_step(int param_num, int *d_param_sizes, float *param, float *grad, size_t pitch_param, size_t pitch_grad, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_param_sizes[param_num * 2] * d_param_sizes[param_num * 2 + 1]) {
        int row = i / d_param_sizes[param_num * 2 + 1];
        int col = i % d_param_sizes[param_num * 2 + 1];

        float *param_pt = (float *)(((char *) param) + row * pitch_param + col * sizeof(float));
        float *grad_pt = (float *)(((char *) grad) + row * pitch_grad + col * sizeof(float));

        *param_pt -= lr * *grad_pt;
    }
}

void initialize_weights() {

    int findex, pindex;

    for (pindex = 0; pindex < NUM_PARAMS; pindex++) {
        size_t malloc_size = ((pindex == 0) ? max_indices[0] : max_indices[pindex] - max_indices[pindex - 1]) * sizeof(float);
        cudaMallocPitch(&params[pindex], &pitch_params[pindex], param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][0]);
        hparams[pindex] = (float *) malloc(malloc_size);
    }
    pindex = 0;

    if (LOAD_MODEL) {
        FILE *fp;
        fp = fopen("model.csv", "r");

        for (findex = 0; findex < max_indices[NUM_PARAMS - 1]; findex++) {
            if (findex >= max_indices[pindex]) pindex++;
            float *value = hparams[pindex];
            fscanf(fp, "%f,", &value[findex - ((pindex == 0) ? 0 : max_indices[pindex - 1])]);
        }

        fclose(fp);
    }
    else {
        int layer_size;
        float stdv;
        layer_size = param_sizes[pindex][1];
        stdv = 1.0 / sqrt((float) layer_size);
        for (findex = 0; findex < max_indices[NUM_PARAMS - 1]; findex++) {
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

    for (pindex = 0; pindex < NUM_PARAMS; pindex++) {
        gpuErrchk(cudaMemcpy2D(params[pindex], pitch_params[pindex], hparams[pindex], param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][1] * sizeof(float), param_sizes[pindex][0], cudaMemcpyHostToDevice));
    }

}

void initialize_stages() {

    int sindex;

    for (sindex = 0; sindex < NUM_STAGES; sindex++) {
        gpuErrchk(cudaMallocPitch(&stages[sindex], &pitch_stages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0]));
        hstages[sindex] = (float *) calloc(stage_sizes[sindex][1] * stage_sizes[sindex][0], sizeof(float));
        if (sindex + 1 >= NUM_STAGES) last_stage = (float *) calloc(stage_sizes[sindex][1] * stage_sizes[sindex][0], sizeof(float));

        gpuErrchk(cudaMemcpy2D(hstages[sindex], stage_sizes[sindex][1] * sizeof(float), stages[sindex], pitch_stages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0], cudaMemcpyDeviceToHost));
    }
    hout = (float *) calloc(stage_sizes[9][1] * stage_sizes[9][0], sizeof(float));
    cudaMalloc(&sum, sizeof(float));
    cudaMalloc(&maxf, sizeof(float));
    cudaMalloc(&loss, sizeof(float));

}

void reset_stages() {

    int sindex;

    for (sindex = 0; sindex < NUM_STAGES; sindex++) {
        cudaMemcpy2D(stages[sindex], pitch_stages[sindex], hstages[sindex], stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][1] * sizeof(float), stage_sizes[sindex][0], cudaMemcpyHostToDevice);
    }

    cudaMemset(sum, 0, sizeof(float));
    cudaMemset(maxf, 0, sizeof(float));
    cudaMemset(loss, 0, sizeof(float));

}

void initialize_grads() {

    int gindex;

    for (gindex = 0; gindex < NUM_GRADS; gindex++) {
        cudaMallocPitch(&grads[gindex], &pitch_grads[gindex], grad_sizes[gindex][1] * sizeof(float), grad_sizes[gindex][0]);
    }

    cudaMalloc(&d_param_sizes, 2 * NUM_PARAMS * sizeof(float));
    cudaMemcpy(d_param_sizes, param_sizes, 2 * NUM_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

}

__global__
void debug_kernel(float *a, size_t pitch_a, int h, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        float add = 0;
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                float *elem_a = (float *)(((char *) a) + r * pitch_a + c * sizeof(float));
                add += *elem_a;
            }
        }
        printf("%f ", add);
    }
}

void inference(int image, float *ploss, int *pacc) {

    matmul<<<16, 1>>>(params[0], stages[0], stages[1], pitch_params[0], pitch_stages[0], pitch_stages[1], LAYER_SIZE_1, INPUT_SIZE, INPUT_SIZE, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    matadd<<<16, 1>>>(stages[1], params[1], stages[2], pitch_stages[1], pitch_params[1], pitch_stages[2], LAYER_SIZE_1, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    leakyrelu<<<16, 1>>>(stages[2], stages[3], 0.2, pitch_stages[2], pitch_stages[3], LAYER_SIZE_1, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    matmul<<<16, 1>>>(params[2], stages[3], stages[4], pitch_params[2], pitch_stages[3], pitch_stages[4], LAYER_SIZE_2, LAYER_SIZE_1, LAYER_SIZE_1, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    matadd<<<16, 1>>>(stages[4], params[3], stages[5], pitch_stages[4], pitch_params[3], pitch_stages[5], LAYER_SIZE_2, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    leakyrelu<<<16, 1>>>(stages[5], stages[6], 0.2, pitch_stages[5], pitch_stages[6], LAYER_SIZE_2, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    matmul<<<10, 1>>>(params[4], stages[6], stages[7], pitch_params[4], pitch_stages[6], pitch_stages[7], LAYER_SIZE_3, LAYER_SIZE_2, LAYER_SIZE_2, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    matadd<<<10, 1>>>(stages[7], params[5], stages[8], pitch_stages[7], pitch_params[5], pitch_stages[8], LAYER_SIZE_3, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    max<<<1, 1>>>(stages[8], maxf, pitch_stages[8], LAYER_SIZE_3, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    expsum<<<10, 1>>>(stages[8], sum, maxf, pitch_stages[8], LAYER_SIZE_3, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    softmax<<<10, 1>>>(stages[8], stages[9], sum, maxf, pitch_stages[8], pitch_stages[9], LAYER_SIZE_3, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cross_entropy<<<10, 1>>>(stages[9], labels, loss, pitch_stages[9], LAYER_SIZE_3, 1, image);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    int nindex;

    gpuErrchk(cudaMemcpy2D(last_stage, stage_sizes[NUM_STAGES - 1][1] * sizeof(float), stages[NUM_STAGES - 1], pitch_stages[NUM_STAGES - 1], stage_sizes[NUM_STAGES - 1][1] * sizeof(float), stage_sizes[NUM_STAGES - 1][0], cudaMemcpyDeviceToHost));
    int max_index = 0;
    for (nindex = 1; nindex < stage_sizes[NUM_STAGES - 1][0]; nindex++) {
        if (last_stage[nindex] > last_stage[max_index]) max_index = nindex;
    }
    *pacc += max_index == hlabels[image];

    float hloss;
    cudaMemcpy(&hloss, loss, sizeof(float), cudaMemcpyDeviceToHost);
    *ploss += hloss;
}

void backprop(int image) {
    int layer;
    for (layer = NUM_LAYERS - 1; layer >= 0; layer--) {
        calc_grad_x<<<param_sizes[layer * 2][1], 1>>>(layer, labels, image, d_param_sizes, params[layer * 2], stages[layer * 3 + 3], stages[layer * 3 + 2], grads[layer * 3 + 1], grads[layer * 3 + 4], pitch_params[layer * 2], pitch_stages[layer * 3 + 3], pitch_stages[layer * 3 + 2], pitch_grads[layer * 3 + 1], pitch_grads[layer * 3 + 4]);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        calc_grad_w<<<param_sizes[layer * 2][1], param_sizes[layer * 2][0]>>>(layer, labels, image, d_param_sizes, stages[layer * 3 + 2], stages[layer * 3], grads[layer * 3], grads[layer * 3 + 1], pitch_stages[layer * 3 + 2], pitch_stages[layer * 3], pitch_grads[layer * 3], pitch_grads[layer * 3 + 1]);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        calc_grad_b<<<param_sizes[layer * 2][0], 1>>>(layer, labels, image, d_param_sizes, params[layer * 2], stages[layer * 3 + 2], grads[layer * 3 + 2], grads[layer * 3 + 1], pitch_params[layer * 2], pitch_stages[layer * 3 + 2], pitch_grads[layer * 3 + 2], pitch_grads[layer * 3 + 1]);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
}

void grad_descent() {
    int param_num;
    int grad_num = 0;
    for (param_num = 0; param_num < NUM_PARAMS; param_num++) {
        grad_step<<<param_sizes[param_num][0], param_sizes[param_num][1]>>>(param_num, d_param_sizes, params[param_num], grads[grad_num], pitch_params[param_num], pitch_grads[grad_num], LR);
        grad_num += (param_num % 2 == 0) ? 2 : 1;
    }

}

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
    initialize_grads();
    load_data();
    int i, epoch, pacc;
    float epoch_loss;
    for (epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        epoch_loss = 0;
        pacc = 0;
        for (i = 0; i < DATA_SIZE; i++) {
            reset_stages();
            gpuErrchk(cudaMemcpy2D(stages[0], pitch_stages[0], images[i], pitch_images[i], stage_sizes[0][1] * sizeof(float), stage_sizes[0][0], cudaMemcpyDeviceToDevice));
            inference(i, &epoch_loss, &pacc);
            backprop(i);
            grad_descent();
        }
        epoch_loss /= (float) DATA_SIZE;
        printf("Epoch %d   Loss : %f   Accuracy : %f\n", epoch, epoch_loss, ((float) pacc) / ((float) DATA_SIZE));
    }
    return 0;
}
