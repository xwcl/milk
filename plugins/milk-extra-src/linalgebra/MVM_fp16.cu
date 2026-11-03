/**
 * @file    MVM_fp16.cu
 * @brief   GPU MVM with a half-precision matrix and single-precision input and sums
 *
 * Halving the matrix size halves the memory traffic of a matrix-vector multiply.
 * Each mode is scaled by its largest |element| before conversion, so the result
 * does not depend on the matrix units (fp16 loses precision below ~6e-5).
 */

#include <cuda_fp16.h>

#include "MVM_fp16.h"

// one warp per mode: scale[k] = max|A[:,k]|, Ah[:,k] = A[:,k] / scale[k], zero-padded to lda
__global__ static void tohalf(const float *__restrict A, int npix, int nmodes,
                              __half *__restrict Ah, float *__restrict scale, int lda)
{
    int k    = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if(k >= nmodes)
    {
        return;
    }
    const float *a = A + (size_t) k * npix;
    float        m = 0;
    for(int i = lane; i < npix; i += 32)
    {
        m = fmaxf(m, fabsf(a[i]));
    }
    for(int o = 16; o > 0; o /= 2)
    {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    float s = (m > 0) ? m : 1;
    for(int i = lane; i < lda; i += 32)
    {
        Ah[(size_t) k * lda + i] = __float2half((i < npix) ? a[i] / s : 0);
    }
    if(lane == 0)
    {
        scale[k] = s;
    }
}

// one warp per mode, 16-byte loads of 8 matrix elements and 8 inputs
__global__ static void gemvT_h(const __half *__restrict A, const float *__restrict scale,
                               const float *__restrict x, float *__restrict y, int lda,
                               int nmodes)
{
    int k    = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if(k >= nmodes)
    {
        return;
    }
    const uint4  *a4  = reinterpret_cast<const uint4 *>(A + (size_t) k * lda);
    const float4 *x4  = reinterpret_cast<const float4 *>(x);
    float         sum = 0;
    for(int j = lane; j < lda / 8; j += 32)
    {
        uint4  av = a4[j];
        float4 x0 = x4[2 * j];
        float4 x1 = x4[2 * j + 1];
        float2 a0 = __half22float2(*reinterpret_cast<__half2 *>(&av.x));
        float2 a1 = __half22float2(*reinterpret_cast<__half2 *>(&av.y));
        float2 a2 = __half22float2(*reinterpret_cast<__half2 *>(&av.z));
        float2 a3 = __half22float2(*reinterpret_cast<__half2 *>(&av.w));
        sum += a0.x * x0.x + a0.y * x0.y + a1.x * x0.z + a1.y * x0.w + a2.x * x1.x +
               a2.y * x1.y + a3.x * x1.z + a3.y * x1.w;
    }
    for(int o = 16; o > 0; o /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, o);
    }
    if(lane == 0)
    {
        y[k] = scale[k] * sum;
    }
}

static const int MVM_FP16_THREADS = 256; // 8 modes per block

extern "C" int MVM_fp16_setup(const float *d_modes, int npix, int nmodes, void **d_modesh,
                              float **d_scale, int *lda)
{
    *lda = (npix + 7) / 8 * 8;
    if(cudaMalloc(d_modesh, sizeof(__half) * (size_t)(*lda) * nmodes) != cudaSuccess)
    {
        return -1;
    }
    if(cudaMalloc((void **) d_scale, sizeof(float) * nmodes) != cudaSuccess)
    {
        cudaFree(*d_modesh);
        *d_modesh = NULL;
        return -1;
    }
    int blocks = (nmodes + MVM_FP16_THREADS / 32 - 1) / (MVM_FP16_THREADS / 32);
    tohalf<<<blocks, MVM_FP16_THREADS>>>(d_modes, npix, nmodes, (__half *) *d_modesh,
                                         *d_scale, *lda);
    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(*d_modesh);
        cudaFree(*d_scale);
        *d_modesh = NULL;
        *d_scale  = NULL;
        return -1;
    }
    return 0;
}

extern "C" int MVM_fp16_gemvT(const void *d_modesh, const float *d_scale, const float *d_in,
                              float *d_out, int lda, int nmodes, cudaStream_t stream)
{
    int blocks = (nmodes + MVM_FP16_THREADS / 32 - 1) / (MVM_FP16_THREADS / 32);
    gemvT_h<<<blocks, MVM_FP16_THREADS, 0, stream>>>((const __half *) d_modesh, d_scale, d_in,
                                                     d_out, lda, nmodes);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
