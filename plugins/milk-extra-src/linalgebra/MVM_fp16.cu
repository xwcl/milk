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

// 8 fp16 matrix elements times 8 fp32 inputs
__device__ __forceinline__ static float dot8(uint4 av, float4 x0, float4 x1)
{
    float2 a0 = __half22float2(*reinterpret_cast<__half2 *>(&av.x));
    float2 a1 = __half22float2(*reinterpret_cast<__half2 *>(&av.y));
    float2 a2 = __half22float2(*reinterpret_cast<__half2 *>(&av.z));
    float2 a3 = __half22float2(*reinterpret_cast<__half2 *>(&av.w));
    return a0.x * x0.x + a0.y * x0.y + a1.x * x0.z + a1.y * x0.w + a2.x * x1.x + a2.y * x1.y +
           a3.x * x1.z + a3.y * x1.w;
}

static const int MVM_FP16_THREADS = 256; // threads per block
static const int MVM_FP16_TPM     = 32;  // threads per mode (64 raised loop jitter on an RTX 4090)

// MVM_FP16_TPM threads per mode, 16-byte loads, two loads in flight per thread
__global__ static void gemvT_h(const __half *__restrict A, const float *__restrict scale,
                               const float *__restrict x, float *__restrict y, int lda,
                               int nmodes)
{
    __shared__ float red[MVM_FP16_THREADS / 32];

    int t = threadIdx.x % MVM_FP16_TPM;
    int k = blockIdx.x * (MVM_FP16_THREADS / MVM_FP16_TPM) + threadIdx.x / MVM_FP16_TPM;
    const uint4  *a4 = reinterpret_cast<const uint4 *>(A + (size_t)((k < nmodes) ? k : 0) * lda);
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    int           n8 = lda / 8;
    float         s0 = 0;
    float         s1 = 0;
    int           j  = t;
    for(; j + MVM_FP16_TPM < n8; j += 2 * MVM_FP16_TPM)
    {
        uint4 p = a4[j];
        uint4 q = a4[j + MVM_FP16_TPM];
        s0 += dot8(p, x4[2 * j], x4[2 * j + 1]);
        s1 += dot8(q, x4[2 * (j + MVM_FP16_TPM)], x4[2 * (j + MVM_FP16_TPM) + 1]);
    }
    if(j < n8)
    {
        s0 += dot8(a4[j], x4[2 * j], x4[2 * j + 1]);
    }

    float sum = s0 + s1;
    for(int o = 16; o > 0; o /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, o);
    }
    int w = threadIdx.x / 32;
    if((threadIdx.x & 31) == 0)
    {
        red[w] = sum;
    }
    __syncthreads();
    if(t == 0 && k < nmodes)
    {
        float r = 0;
        for(int i = 0; i < MVM_FP16_TPM / 32; i++)
        {
            r += red[w + i];
        }
        y[k] = scale[k] * r;
    }
}

extern "C" int MVM_fp16_setup(const float *d_modes, int npix, int nmodes, void **d_modesh,
                              float **d_scale, int *lda, int *l2persist)
{
    *lda       = (npix + 7) / 8 * 8;
    *l2persist = 0;
    size_t nbytes = sizeof(__half) * (size_t)(*lda) * nmodes;
    if(cudaMalloc(d_modesh, nbytes) != cudaSuccess)
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

    // reserve L2 for the matrix if it fits, so other GPU work cannot evict it
    int dev;
    int maxpersist = 0;
    int maxwindow  = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&maxpersist, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&maxwindow, cudaDevAttrMaxAccessPolicyWindowSize, dev);
    if((size_t) maxpersist >= nbytes && (size_t) maxwindow >= nbytes &&
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, nbytes) == cudaSuccess)
    {
        *l2persist = 1;
    }
    return 0;
}

extern "C" int MVM_fp16_gemvT(const void *d_modesh, const float *d_scale, const float *d_in,
                              float *d_out, int lda, int nmodes, cudaStream_t stream,
                              int l2persist)
{
    cudaLaunchAttribute attr;
    attr.id                                        = cudaLaunchAttributeAccessPolicyWindow;
    attr.val.accessPolicyWindow.base_ptr           = (void *) d_modesh;
    attr.val.accessPolicyWindow.num_bytes          = sizeof(__half) * (size_t) lda * nmodes;
    attr.val.accessPolicyWindow.hitRatio           = 1.0f;
    attr.val.accessPolicyWindow.hitProp            = cudaAccessPropertyPersisting;
    attr.val.accessPolicyWindow.missProp           = cudaAccessPropertyStreaming;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3((nmodes + MVM_FP16_THREADS / MVM_FP16_TPM - 1) /
                        (MVM_FP16_THREADS / MVM_FP16_TPM));
    cfg.blockDim = dim3(MVM_FP16_THREADS);
    cfg.stream   = stream;
    cfg.attrs    = &attr;
    cfg.numAttrs = l2persist ? 1 : 0;

    return (cudaLaunchKernelEx(&cfg, gemvT_h, (const __half *) d_modesh, d_scale, d_in, d_out,
                               lda, nmodes) == cudaSuccess) ? 0 : -1;
}

extern "C" void MVM_fp16_l2release(void)
{
    cudaCtxResetPersistingL2Cache();
}
