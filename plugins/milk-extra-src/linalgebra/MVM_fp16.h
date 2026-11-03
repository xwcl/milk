/**
 * @file    MVM_fp16.h
 * @brief   GPU MVM with a half-precision matrix and single-precision input and sums
 */

#ifndef LINALGEBRA_MVM_FP16_H
#define LINALGEBRA_MVM_FP16_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// Convert a column-major npix x nmodes fp32 matrix on the GPU to fp16, one scale
// factor per mode (its largest |element|), with columns padded to *lda (multiple of 8).
// Allocates *d_modesh and *d_scale. Sets *l2persist when L2 has been reserved for
// the fp16 matrix. Returns 0 on success.
int MVM_fp16_setup(const float *d_modes, int npix, int nmodes, void **d_modesh,
                   float **d_scale, int *lda, int *l2persist);

// d_out[k] = d_scale[k] * sum_i modesh[k*lda + i] * d_in[i], summed in fp32.
// d_in must hold lda floats, zero beyond npix. With l2persist the matrix is kept
// in the L2 reserved by MVM_fp16_setup.
int MVM_fp16_gemvT(const void *d_modesh, const float *d_scale, const float *d_in,
                   float *d_out, int lda, int nmodes, cudaStream_t stream, int l2persist);

// Return L2 lines held for the fp16 matrix to normal use (when switching back to fp32)
void MVM_fp16_l2release(void);

#ifdef __cplusplus
}
#endif

#endif
