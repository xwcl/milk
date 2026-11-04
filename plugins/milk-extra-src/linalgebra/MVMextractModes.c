#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#endif


#include <pthread.h>


// Use MKL if available
// Otherwise use openBLAS
//
#ifdef HAVE_MKL
#include "mkl.h"
#define BLASLIB "IntelMKL"
#else
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#define BLASLIB "OpenBLAS"
#endif
#endif



#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "MVM_CPU.h"

#ifdef HAVE_CUDA
#include "MVM_fp16.h"
#endif



// Local variables pointers
static int32_t *GPUindex;
long fpi_GPUindex;


static uint32_t *mmax;
long fpi_mmax;

static uint32_t *nmax;
long fpi_nmax;



static char *insname;
long fpi_insname;

static char *inmasksname;
long fpi_inmasksname;

static char *immodes;
long fpi_immodes;

static char *outcoeff;
long fpi_outcoeff;

static int64_t *outinit;
long fpi_outinit;

static uint32_t *axmode;
long fpi_axmode;

static int64_t *PROCESS;
long fpi_PROCESS;

static int64_t *TRACEMODE;
long fpi_TRACEMODE;

static int64_t *MODENORM;
long fpi_MODENORM;

static char *intot_stream;
long fpi_intot_stream;

static char *inrefsname;
long fpi_inrefsname;

static char *outrefsname;
long fpi_outrefsname;

static uint64_t *twait;
long fpi_twait;

static int64_t *GPUtiming;
long fpi_GPUtiming;

static int64_t *GPUpinned;
long fpi_GPUpinned;

static int64_t *GPUgraph;
long fpi_GPUgraph;

static int32_t *GPUsplit;
long fpi_GPUsplit;

static int64_t *GPUfp16;
long fpi_GPUfp16;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_INT32,
        ".GPUindex",
        "GPU index, 99 for CPU",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &GPUindex,
        &fpi_GPUindex
    },
    {
        CLIARG_STREAM,
        ".insname",
        "input stream name",
        "inV",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &insname,
        &fpi_insname
    },
    {
        CLIARG_STREAM,
        ".inmasksname",
        "nput mask stream name",
        "inV",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmasksname,
        &fpi_inmasksname
    },
    {
        CLIARG_STREAM,
        ".immodes",
        "modes stream name",
        "mat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &immodes,
        &fpi_immodes
    },
    {
        CLIARG_STREAM,
        ".outcoeff",
        "output coefficients",
        "outV",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outcoeff,
        &fpi_outcoeff
    },
    {
        CLIARG_ONOFF,
        ".outinit",
        "output init mode",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &outinit,
        &fpi_outinit
    },
    {
        CLIARG_UINT32,
        ".option.axmode",
        "0 for normal mode extraction, 1 for expansion",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &axmode,
        &fpi_axmode
    },
    {
        CLIARG_ONOFF,
        ".option.PROCESS",
        "processing flag",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &PROCESS,
        &fpi_PROCESS
    },
    {
        CLIARG_ONOFF,
        ".option.TRACEMODE",
        "writing trace",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &TRACEMODE,
        &fpi_TRACEMODE
    },
    {
        CLIARG_ONOFF,
        ".option.MODENORM",
        "input modes normalization",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &MODENORM,
        &fpi_MODENORM
    },
    {
        CLIARG_STREAM,
        ".option.sname_intot",
        "optional input normalization stream",
        "null",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &intot_stream,
        &fpi_intot_stream
    },
    {
        CLIARG_STREAM,
        ".option.sname_refin",
        "optional input reference to be subtracted stream",
        "null",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inrefsname,
        &fpi_inrefsname
    },
    {
        CLIARG_STREAM,
        ".option.sname_refout",
        "optional output reference to be subtracted stream",
        "null",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outrefsname,
        &fpi_outrefsname
    },
    {
        CLIARG_UINT64,
        ".option.twait",
        "delay output publication [us] (GPU mode)",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &twait,
        &fpi_twait
    },
    {
        CLIARG_UINT32,
        ".option.mmax",
        "partial computation: max m index value",
        "100000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &mmax,
        &fpi_mmax
    },
    {
        CLIARG_UINT32,
        ".option.nmax",
        "partial computation: max n index value",
        "100000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &nmax,
        &fpi_nmax
    },
    {
        CLIARG_ONOFF,
        ".option.GPUtiming",
        "time the GPU MVM kernel with CUDA events",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUtiming,
        &fpi_GPUtiming
    },
    {
        CLIARG_ONOFF,
        ".option.GPUpinned",
        "stage GPU transfers through pinned host buffers",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUpinned,
        &fpi_GPUpinned
    },
    {
        CLIARG_ONOFF,
        ".option.GPUgraph",
        "launch the per-frame GPU copies and MVM as one CUDA graph",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUgraph,
        &fpi_GPUgraph
    },
    {
        CLIARG_INT32,
        ".option.GPUsplit",
        "second GPU computing half the modes, -1 for none (uses the CUDA graph)",
        "-1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUsplit,
        &fpi_GPUsplit
    },
    {
        CLIARG_ONOFF,
        ".option.GPUfp16",
        "GPU MVM with a half-precision matrix, single-precision sums (not with GPUsplit)",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUfp16,
        &fpi_GPUfp16
    }
};

// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        data.fpsptr->parray[fpi_insname].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
        data.fpsptr->parray[fpi_immodes].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
        data.fpsptr->parray[fpi_GPUtiming].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_GPUpinned].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_GPUgraph].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_GPUsplit].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_GPUfp16].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_twait].fpflag |= FPFLAG_WRITERUN;
    }

    return RETURN_SUCCESS;
}

// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {
    }

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "MVMmextrmodes", "extract modes by MVM", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



#ifdef HAVE_CUDA
// Capture the per-frame GPU work (input copy, MVM, result copy) as one CUDA graph,
// replacing three launches and their synchronizations with one of each.
// The MVM uses the fp16 matrix d_modesh when given, otherwise cuBLAS on d_modes.
// Returns NULL if capture fails.
static cudaGraphExec_t capture_mvm_graph(
    cudaStream_t stream,
    cublasHandle_t cublasH,
    float *d_in,
    const float *h_in,
    int npix,
    const float *d_modes,
    int NBmodes,
    float alpha,
    float beta,
    float *d_modeval,
    float *h_modeval,
    const void *d_modesh,
    const float *d_mscale,
    int ldah,
    int l2persist
)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphexec = NULL;

    cublasSetStream(cublasH, stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    cudaMemcpyAsync(d_in, h_in, sizeof(float) * npix, cudaMemcpyHostToDevice, stream);
    if(d_modesh != NULL)
    {
        MVM_fp16_gemvT(d_modesh, d_mscale, d_in, d_modeval, ldah, NBmodes, stream, l2persist);
    }
    else
    {
        cublasSgemv(cublasH, CUBLAS_OP_T, npix, NBmodes, &alpha, d_modes, npix, d_in, 1,
                    &beta, d_modeval, 1);
    }
    cudaMemcpyAsync(h_modeval, d_modeval, sizeof(float) * NBmodes, cudaMemcpyDeviceToHost,
                    stream);
    if(cudaStreamEndCapture(stream, &graph) == cudaSuccess)
    {
        if(cudaGraphInstantiateWithFlags(&graphexec, graph, 0) != cudaSuccess)
        {
            graphexec = NULL;
        }
        cudaGraphDestroy(graph);
    }
    cublasSetStream(cublasH, 0);

    return graphexec;
}



// Second-GPU state for .option.GPUsplit
typedef struct
{
    int             dev;      // second GPU index, -1 until set up
    int             n0;       // modes [0, n0) run on the first GPU, the rest on the second
    cudaGraphExec_t graph0;   // first GPU's graph over modes [0, n0)
    cudaGraphExec_t graph1;   // second GPU's graph over modes [n0, NBmodes)
    cudaStream_t    stream1;
    cublasHandle_t  cublasH1;
    float          *d_modes1;
    float          *d_in1;
    float          *d_modeval1;
} MVMSPLIT;

// Copy the second half of the modes to dev1 and capture a graph on each GPU.
// The modes are matrix columns, so each half is contiguous in d_modes.
// Both graphs read h_in and write their half of h_modeval.
// Returns 0 on success, -1 on failure.
static int setup_mvm_split(
    MVMSPLIT *sp,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cublasHandle_t cublasH0,
    float *d_in0,
    float *d_modeval0,
    const float *d_modes,
    const float *h_in,
    int npix,
    int NBmodes,
    float alpha,
    float beta,
    float *h_modeval
)
{
    sp->n0 = NBmodes / 2;
    int n1 = NBmodes - sp->n0;

    sp->graph0 = capture_mvm_graph(stream0, cublasH0, d_in0, h_in, npix, d_modes, sp->n0,
                                   alpha, beta, d_modeval0, h_modeval, NULL, NULL, 0, 0);
    if(sp->graph0 == NULL)
    {
        return -1;
    }

    // first use of dev1 creates its context, which takes a while
    int ok = (cudaSetDevice(dev1) == cudaSuccess)
             && (cudaMalloc((void **)&sp->d_modes1, sizeof(float) * npix * n1) == cudaSuccess)
             && (cudaMalloc((void **)&sp->d_in1, sizeof(float) * npix) == cudaSuccess)
             && (cudaMalloc((void **)&sp->d_modeval1, sizeof(float) * n1) == cudaSuccess)
             && (cudaMemcpyPeer(sp->d_modes1, dev1, d_modes + (size_t) npix * sp->n0, dev0,
                                sizeof(float) * npix * n1) == cudaSuccess)
             && (cudaStreamCreateWithFlags(&sp->stream1, cudaStreamNonBlocking) == cudaSuccess)
             && (cublasCreate(&sp->cublasH1) == CUBLAS_STATUS_SUCCESS);
    if(ok)
    {
        sp->graph1 = capture_mvm_graph(sp->stream1, sp->cublasH1, sp->d_in1, h_in, npix,
                                       sp->d_modes1, n1, alpha, beta, sp->d_modeval1,
                                       h_modeval + sp->n0, NULL, NULL, 0, 0);
        ok = (sp->graph1 != NULL);
    }
    cudaSetDevice(dev0);

    if(!ok)
    {
        return -1;
    }
    sp->dev = dev1;
    return 0;
}
#endif














static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    int MODEVALCOMPUTE = 1; // 1 if compute, 0 if import


#ifdef HAVE_CUDA
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    struct cudaDeviceProp deviceProp;
    cudaEvent_t evgemv0, evgemv1; // bracket the GPU MVM kernel for .option.GPUtiming
    cudaStream_t mvmstream = NULL;    // stream for .option.GPUgraph
    cudaGraphExec_t mvmgraph = NULL;  // captured on first use of .option.GPUgraph
    MVMSPLIT mvmsplit = {.dev = -1};  // set up on first use of .option.GPUsplit
    void *d_modesh = NULL;            // fp16 matrix, converted on first use of .option.GPUfp16
    float *d_mscale = NULL;           // per-mode scale of d_modesh
    int ldah = 0;                     // padded column length of d_modesh
    int l2fp16 = 0;                   // L2 is reserved for d_modesh
    int fp16prev = 0;                 // previous frame used d_modesh
    cudaGraphExec_t mvmgraph16 = NULL; // .option.GPUgraph with .option.GPUfp16
#endif

    float *d_modes = NULL; // linear memory of GPU
    float *d_in = NULL;
    float *d_modeval = NULL;
    float *h_in = NULL;      // pinned host staging for d_in, used with .option.GPUpinned
    float *h_modeval = NULL; // pinned host staging for d_modeval, used with .option.GPUpinned




    // each step is 2x longer average than previous step
    uint32_t NBaveSTEP = 10;

    int initref = 0; // 1 when reference has been processed

    // CONNECT TO INPUT STREAM
    IMGID imgin = mkIMGID_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_ABORT);
    printf("Input stream size : %u %u\n", imgin.md->size[0], imgin.md->size[1]);
    long m = imgin.md->size[0] * imgin.md->size[1];

    // CONNECT TO MASK STREAM
    int use_mask = 0; //flag indicating that the mask is being used
    uint32_t mask_npix = 0; //The number of 1 pixels in the mask
    uint32_t *mask_idx = NULL;  //Array holding the indices of the 1 pixels
    float *masked_pix = NULL;  //Array to hold the pixel values

    IMGID imgmask = mkIMGID_from_name(inmasksname);
    if(resolveIMGID(&imgmask, ERRMODE_WARN) != -1)
    {
        printf("Mask stream size : %u %u\n", imgmask.md->size[0], imgmask.md->size[1]);
        if(imgmask.md->size[0] == imgin.md->size[0]
                && imgmask.md->size[1] == imgin.md->size[1])
        {
            use_mask = 1;
        }
    }

    printf("USE MASK = %d\n", use_mask);

    //use_mask = 0; //for testing

    //setup the mask
    //
    if(use_mask)
    {
        for(long n = 0; n < imgmask.md->size[0]*imgmask.md->size[1]; ++n)
        {
            if(imgmask.im->array.F[n] == 1)
            {
                ++mask_npix;
            }
        }

        mask_idx = (uint32_t *) malloc(mask_npix * sizeof(long));
        masked_pix = (float *) malloc(mask_npix * sizeof(float));
        long nn = 0;
        for(long n = 0; n < imgmask.md->size[0]*imgmask.md->size[1]; ++n)
        {
            if(imgmask.im->array.F[n] == 1)
            {
                mask_idx[nn] = n;
                ++nn;
            }
        }

        printf("Mask has : %u pixels (%f%%)\n", mask_npix,
               (100.0 * mask_npix) / (imgmask.md->size[0]*imgmask.md->size[1]));
    }
    else
    {
        //Just use full image
        mask_npix = imgin.md->size[0] * imgin.md->size[1];
        printf("No mask using : %u pixels (%f%%)\n", mask_npix,
               (100.0 * mask_npix) / (imgin.md->size[0]*imgin.md->size[1]));
    }



    /* // This was probaly never implemented at all.
    // NORMALIZATION
    // CONNECT TO TOTAL FLUX STREAM
    imageID IDintot;
    IDintot = image_ID(intot_stream);
    int INNORMMODE = 0; // 1 if input normalized

    if(IDintot == -1)
    {
        INNORMMODE = 0;
        create_2Dimage_ID("intot_tmp", 1, 1, &IDintot);
        data.image[IDintot].array.F[0] = 1.0;
    }
    else
    {
        INNORMMODE = 1;
    }
    */



    // CONNECT TO OPTIONAL INPUT REFERENCE STREAM
    imageID IDinref = -1;
    IMGID imginref = mkIMGID_from_name(inrefsname);
    resolveIMGID(&imginref, ERRMODE_WARN);
    if(imginref.ID == -1)
    {
        create_2Dimage_ID("_tmprefin",
                          imgin.md->size[0],
                          imgin.md->size[1],
                          &IDinref);
        for(uint64_t ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
        {
            data.image[IDinref].array.F[ii] = 0.0;
        }
    }
    else
    {
        IDinref = imginref.ID;
    }


    // CONNECT TO OPTIONAL OUTPUT REFERENCE STREAM
    IMGID imgoutref = mkIMGID_from_name(outrefsname);
    resolveIMGID(&imgoutref, ERRMODE_WARN);


    // CONNECT TO MODES STREAM
    IMGID imgmodes = mkIMGID_from_name(immodes);
    resolveIMGID(&imgmodes, ERRMODE_ABORT);

    // Could this be IMGIDcompare?
    if(imgmodes.md->datatype != _DATATYPE_FLOAT)
    {
        PRINT_ERROR("Cannot operate with modes other than FP32!!!s");
        abort();
    }

    printf("Modes stream size : %u %u\n",
           imgmodes.md->size[0],
           imgmodes.md->size[1]);


    long n;
    long NBmodes = 1;
    imageID IDmodes = -1;


    if((*axmode) == 0)
    {
        //
        // Extract modes.
        // This is the default geometry, no need to remap
        //
        n = imgmodes.md->size[2];
        IDmodes = imgmodes.ID;
        NBmodes = n;
        printf("NBmodes = %ld\n", NBmodes);
        fflush(stdout);


        // make col-major storage
    }
    else
    {
        //
        // Expand
        // Remap to new matrix tmpmodes
        //

        NBmodes = imgmodes.md->size[0] * imgmodes.md->size[1];
        n = NBmodes;
        printf("NBmodes = %ld\n", NBmodes);
        fflush(stdout);

        printf("creating _tmpmodes  %ld %ld %ld\n",
               (long)imgin.md->size[0],
               (long)imgin.md->size[1],
               NBmodes);
        fflush(stdout);

        create_3Dimage_ID("_tmpmodes",
                          imgin.md->size[0],
                          imgin.md->size[1],
                          NBmodes,
                          &IDmodes);

        for(uint32_t ii = 0; ii < imgin.md->size[0]; ii++)
            for(uint32_t jj = 0; jj < imgin.md->size[1]; jj++)
            {
                for(long kk = 0; kk < NBmodes; kk++)
                {
                    data.image[IDmodes]
                    .array.F[kk * imgin.md->size[0] * imgin.md->size[1] +
                                jj * imgin.md->size[0] + ii] =
                                 imgmodes.im->array
                                 .F[NBmodes * (jj * imgin.md->size[0] + ii) + kk];
                }
            }

        // save_fits("_tmpmodes", "_test_tmpmodes.fits");
    }

    float *normcoeff = (float *)malloc(sizeof(float) * NBmodes);

    if((*MODENORM) == 1)
    {
        // In this mode, the input modes are normalized to unity (vector 2-norm)
        // norm is computed here


        // compute normalization coeffs
        for(long k = 0; k < NBmodes; k++)
        {
            normcoeff[k] = 0.0;
            for(long ii = 0; ii < m; ii++)
            {
                normcoeff[k] += imgmodes.im->array.F[k * m + ii] *
                                imgmodes.im->array.F[k * m + ii];
            }
        }
    }
    else
    {
        // or set them to 1
        for(long k = 0; k < NBmodes; k++)
        {
            normcoeff[k] = 1.0;
        }
    }

    float *modevalarray = (float *)malloc(sizeof(float) * n);
    float *modevalarrayref = (float *)malloc(sizeof(float) * n);

    uint32_t *arraytmp = (uint32_t *)malloc(sizeof(uint32_t) * 2);

    // IDrefout = image_ID(IDrefout_name);
    imageID IDrefout = -1; // TODO handle this
    if(IDrefout == -1)
    {
        if((*axmode) == 0)
        {
            arraytmp[0] = NBmodes;
            arraytmp[1] = 1;
        }
        else
        {
            arraytmp[0] = imgmodes.md->size[0];
            arraytmp[1] = imgmodes.md->size[1];
        }
    }
    else
    {
        arraytmp[0] = data.image[IDrefout].md->size[0];
        arraytmp[1] = data.image[IDrefout].md->size[1];
    }



    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgout = stream_connect_create_2Df32(outcoeff, arraytmp[0], arraytmp[1]);

    // Local working copy of output
    float *outarray = (float *) malloc(sizeof(float) * arraytmp[0] * arraytmp[1]);



    MODEVALCOMPUTE = 1;

    free(arraytmp);



    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;



    if(MODEVALCOMPUTE == 1)
    {

        if(((*GPUindex) >= 0) && ((*GPUindex) != 99))
        {

#ifdef HAVE_CUDA
            int deviceCount;
            int devicecntMax = 100;

            cudaGetDeviceCount(&deviceCount);
            printf("%d devices found\n", deviceCount);
            fflush(stdout);

            processinfo_WriteMessage_fmt(processinfo, "CUDA : %d devices", deviceCount);

            if(deviceCount > devicecntMax)
            {
                deviceCount = 0;
            }
            if(deviceCount < 0)
            {
                deviceCount = 0;
            }

            printf("\n");

            for(int k = 0; k < deviceCount; k++)
            {
                cudaGetDeviceProperties(&deviceProp, k);

                int clockRate;
                cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, k);

                printf("Device %d / %d [ %20s ]  has compute capability %d.%d.\n",
                       k + 1,
                       deviceCount,
                       deviceProp.name,
                       deviceProp.major,
                       deviceProp.minor);
                printf(
                    "  Total amount of global memory:                 %.0f MBytes "
                    "(%llu bytes)\n",
                    (float)deviceProp.totalGlobalMem / 1048576.0f,
                    (unsigned long long)deviceProp.totalGlobalMem);
                printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
                printf(
                    "  GPU Clock rate:                                %.0f MHz "
                    "(%0.2f GHz)\n",
                    clockRate * 1e-3f,
                    clockRate * 1e-6f);
                printf("\n");
            }

            if((*GPUindex) < deviceCount)
            {
                cudaSetDevice(*GPUindex);
            }
            else
            {
                printf("Invalid Device : %d / %d\n", *GPUindex, deviceCount);
                processinfo_WriteMessage_fmt(processinfo, "Invalid GPU device %d", *GPUindex);
                exit(0);
            }

            printf("Create cublas handle ...");
            fflush(stdout);
            cublas_status = cublasCreate(&cublasH);
            if(cublas_status != CUBLAS_STATUS_SUCCESS)
            {
                printf("CUBLAS initialization failed\n");
                return EXIT_FAILURE;
            }
            printf(" done\n");
            fflush(stdout);

            long matsz;
            float *modesmat;

            if(use_mask)
            {
                //reformat the matrix using the mask
                matsz = mask_npix * NBmodes;
                modesmat = (float *) malloc(sizeof(float) * mask_npix *
                                            data.image[IDmodes].md->size[2]);

                uint32_t nrows = data.image[IDmodes].md->size[2];
                uint32_t ncols = data.image[IDmodes].md->size[0] *
                                 data.image[IDmodes].md->size[1];

                for(uint32_t rr = 0; rr < nrows; ++rr)
                {
                    for(uint32_t cc = 0; cc < mask_npix; ++cc)
                    {
                        modesmat[rr * mask_npix + cc] = data.image[IDmodes].array.F[rr * ncols +
                                                        mask_idx[cc]];
                    }
                }
            }
            else
            {
                matsz = m * NBmodes;
                modesmat = data.image[IDmodes].array.F;
            }

            // load modes to GPU
            cudaStat = cudaMalloc((void **)&d_modes, sizeof(float) * matsz);
            if(cudaStat != cudaSuccess)
            {
                printf("cudaMalloc d_modes returned error code %d, line %d\n",
                       cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }

            cudaStat = cudaMemcpy(d_modes,
                                  modesmat,
                                  sizeof(float) * matsz,
                                  cudaMemcpyHostToDevice);
            // cudaStat = cudaMemcpy(d_modes, imgmodes.im->array.F, sizeof(float) * m * NBmodes, cudaMemcpyHostToDevice);

            if(use_mask)
            {
                free(modesmat);
            }

            if(cudaStat != cudaSuccess)
            {
                printf("cudaMemcpy returned error code %d, line %d\n",
                       cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }


            // create d_in, zero-padded to a multiple of 8 for the fp16 MVM
            cudaStat = cudaMalloc((void **)&d_in, sizeof(float) * ((m + 7) / 8 * 8));
            if(cudaStat == cudaSuccess)
            {
                cudaStat = cudaMemset(d_in, 0, sizeof(float) * ((m + 7) / 8 * 8));
            }
            if(cudaStat != cudaSuccess)
            {
                printf("cudaMalloc d_in returned error code %d, line %d\n",
                       cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }

            // create d_modeval
            cudaStat = cudaMalloc((void **)&d_modeval, sizeof(float) * NBmodes);
            if(cudaStat != cudaSuccess)
            {
                printf("cudaMalloc d_modeval returned error code %d, line %d\n",
                       cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }

            cudaEventCreate(&evgemv0);
            cudaEventCreate(&evgemv1);
            cudaStreamCreateWithFlags(&mvmstream, cudaStreamNonBlocking);

            // pinned buffers let the DMA engine copy directly, skipping the driver's staging copy
            if(cudaMallocHost((void **)&h_in, sizeof(float) * mask_npix) != cudaSuccess ||
                    cudaMallocHost((void **)&h_modeval, sizeof(float) * NBmodes) != cudaSuccess)
            {
                printf("cudaMallocHost returned error, line %d\n", __LINE__);
                exit(EXIT_FAILURE);
            }
#else
            processinfo_WriteMessage(processinfo, "NO CUDA - CPU fallback");
            *GPUindex = 99;
#endif
        }
    }

    if((*TRACEMODE) == 1)
    {
        char traceim_name[STRINGMAXLEN_IMGNAME];
        long TRACEsize = 2000;
        imageID IDtrace;

        uint32_t *sizearraytmp = (uint32_t *)malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(traceim_name,
                                STRINGMAXLEN_IMGNAME,
                                "%s_trace",
                                outcoeff);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0] = TRACEsize;
        sizearraytmp[1] = NBmodes;
        IDtrace = image_ID(traceim_name);
        int imOK = 1;
        if(IDtrace == -1)
        {
            imOK = 0;
        }
        else
        {
            if((data.image[IDtrace].md[0].size[0] != TRACEsize) ||
                    (data.image[IDtrace].md[0].size[1] != NBmodes))
            {
                imOK = 0;
                delete_image_ID(traceim_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if(imOK == 0)
        {
            create_image_ID(traceim_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDtrace);
        }
        free(sizearraytmp);
    }

    if((*PROCESS) == 1)
    {
        char process_ave_name[STRINGMAXLEN_IMGNAME];
        char process_rms_name[STRINGMAXLEN_IMGNAME];
        imageID IDprocave;
        imageID IDprocrms;

        uint32_t *sizearraytmp = (uint32_t *)malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(process_ave_name,
                                STRINGMAXLEN_IMGNAME,
                                "%s_ave",
                                outcoeff);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0] = NBmodes;
        sizearraytmp[1] = NBaveSTEP;
        IDprocave       = image_ID(process_ave_name);
        int imOK = 1;
        if(IDprocave == -1)
        {
            imOK = 0;
        }
        else
        {
            if((data.image[IDprocave].md[0].size[0] != NBmodes) ||
                    (data.image[IDprocave].md[0].size[1] != NBaveSTEP))
            {
                imOK = 0;
                delete_image_ID(process_ave_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if(imOK == 0)
        {
            create_image_ID(process_ave_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDprocave);
        }
        free(sizearraytmp);

        sizearraytmp = (uint32_t *)malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(process_rms_name,
                                STRINGMAXLEN_IMGNAME,
                                "%s_rms",
                                outcoeff);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0] = NBmodes;
        sizearraytmp[1] = NBaveSTEP;
        IDprocrms = image_ID(process_rms_name);
        imOK = 1;
        if(IDprocrms == -1)
        {
            imOK = 0;
        }
        else
        {
            if((data.image[IDprocrms].md->size[0] != NBmodes) ||
                    (data.image[IDprocrms].md->size[1] != NBaveSTEP))
            {
                imOK = 0;
                delete_image_ID(process_rms_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if(imOK == 0)
        {
            create_image_ID(process_rms_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDprocrms);
        }
        free(sizearraytmp);
    }

    initref = 0; // 1 when reference has been processed

    printf("LOOP START   MODEVALCOMPUTE = %d\n", MODEVALCOMPUTE);
    fflush(stdout);

    if(MODEVALCOMPUTE == 0)
    {
        printf("\n");
        printf("This function is NOT computing mode values\n");
        printf("Pre-existing stream %s was detected\n", outcoeff);
        printf("\n");
    }
    else
    {
        char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

        {
            int slen = snprintf(msgstring,
                                STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                "Running on GPU %d",
                                (*GPUindex));
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
    }

    printf(" m       = %u\n", mask_npix);
    printf(" n       = %ld\n", n);
    printf(" NBmodes = %ld\n", NBmodes);

    float alpha = 1.0;
    float beta = 0.0;
    uint64_t refindex = 0;

#ifdef HAVE_OPENBLAS
    printf("OpenBLASS  YES\n");
#else
    printf("OpenBLASS  NO\n");
#endif

#ifdef HAVE_MKL
    printf("MKL        YES\n");
#else
    printf("MKL        NO\n");
#endif


#ifdef HAVE_CUDA
    printf("CUDA       YES\n");
#else
    printf("CUDA       NO\n");
#endif




    float *ColMajorMatrix = (float *) malloc(sizeof(float) * m * n);
    if(*axmode == 0)
    {
        for(int ii = 0; ii < m; ii++)
        {
            for(int jj = 0; jj < n; jj++)
            {
                ColMajorMatrix[ii * n + jj] = imgmodes.im->array.F[jj * m + ii];
            }
        }
    }
    else
    {
        memcpy(ColMajorMatrix, imgmodes.im->array.F, sizeof(float)*m * n);
    }


    float *imginfloatptr = NULL;


    if(imgin.md->datatype == _DATATYPE_FLOAT)
    {
        imginfloatptr = imgin.im->array.F;
        printf("INPUT type = FLOAT  - no type conversion required\n");
    }
    else
    {
        imginfloatptr = (float *) malloc(sizeof(float) * imgin.md->size[0] *
                                         imgin.md->size[1]);
        printf("INPUT not float -> type conversion to float enabled\n");
    }


    printf(">>> START MVM loop\n");

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        // Are we computing a new reference ?
        // if yes, set initref to 0 (reference is NOT initialized)
        //
        if(refindex != data.image[IDinref].md->cnt0)
        {
            initref = 0;
            refindex = data.image[IDinref].md->cnt0;
        }


        if(((*GPUindex) < 0) || (*GPUindex == 99))
        {
            // using CPU

#ifdef BLASLIB
            struct timespec t0, t1;
            clock_gettime(CLOCK_MILK, &t0);
            processinfo_WriteMessage_fmt(processinfo, "imgout %s ID %d", imgout.md->name,
                                         imgout.ID);
            {
                float beta = 0.0;

                if(imgoutref.ID != -1)
                {
                    beta = 1.0;
                    memcpy(outarray, imgoutref.im->array.F, sizeof(float)*n);
                }



                if(imgin.md->datatype != _DATATYPE_FLOAT)
                {
                    imginfloatptr = imgin.im->array.F;


                    // type conversion (if needed)
                    switch(imgin.md->datatype)
                    {
                    case _DATATYPE_DOUBLE:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.D[ii];
                        }
                        break;

                    case _DATATYPE_UINT8:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.UI8[ii];
                        }
                        break;

                    case _DATATYPE_INT8:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.SI8[ii];
                        }
                        break;

                    case _DATATYPE_UINT16:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.UI16[ii];
                        }
                        break;

                    case _DATATYPE_INT16:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.SI16[ii];
                        }
                        break;

                    case _DATATYPE_UINT32:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.UI32[ii];
                        }
                        break;

                    case _DATATYPE_INT32:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.SI32[ii];
                        }
                        break;

                    case _DATATYPE_UINT64:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.UI64[ii];
                        }
                        break;

                    case _DATATYPE_INT64:
                        for(int ii = 0; ii < imgin.md->size[0] * imgin.md->size[1]; ii++)
                        {
                            imginfloatptr[ii] = (float) imgin.im->array.SI64[ii];
                        }
                        break;
                    }
                }



                if(*axmode == 1)
                {
                    cblas_sgemv(CblasColMajor,
                                CblasNoTrans, (int) n, (int) m,
                                1.0, ColMajorMatrix, (int) n,
                                imginfloatptr, 1, beta,
                                outarray, 1);
                }
                else
                {
                    cblas_sgemv(CblasColMajor,
                                CblasNoTrans, (int) n, (int) m,
                                1.0, ColMajorMatrix, (int) n,
                                imginfloatptr, 1, beta,
                                outarray, 1);
                }

                clock_gettime(CLOCK_MILK, &t1);
                struct timespec tdiff;
                tdiff = timespec_diff(t0, t1);
                double t01d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                processinfo_WriteMessage_fmt(processinfo, "%s %dx%d MVM %.3f us", BLASLIB, n, m,
                                             t01d * 1e6);
            }
#else
            // Run on CPU without lib
            int mmax1 = (*mmax);
            if(mmax1 > m)
            {
                mmax1 = m;
            }

            int nmax1 = (*nmax);
            if(nmax1 > n)
            {
                nmax1 = n;
            }

            for(int jj = 0; jj < n; jj++)
            {
                outarray[jj] = 0.0;
            }


            for(int ii = 0; ii < m; ii++)
            {
                for(int jj = 0; jj < n; jj++)
                {
                    int index = ii * n + jj;
                    outarray[jj] += imgmodes.im->array.F[index] * imginfloatptr[ii];
                }
            }

#endif

            // update output
            data.image[imgout.ID].md->write = 1;
            for(int jj = 0; jj < n; jj++)
            {
                imgout.im->array.F[jj] = outarray[jj] / normcoeff[jj];
            }
//            memcpy(imgout.im->array.F, outarray, sizeof(float)*n);
            processinfo_update_output_stream(processinfo, imgout.ID);
        }
        else
        {
            // running on GPU
#ifdef HAVE_CUDA

            // stage times: t0 start, t1 gathered, t2 on GPU, t3 result back, t4 output written
            struct timespec t0, t1, t2, t3, t4;
            clock_gettime(CLOCK_MILK, &t0);
            int gputiming = (*GPUtiming == 1); // read once so both events are recorded
            int gpufp16 = (*GPUfp16 == 1);
            if(gpufp16 && d_modesh == NULL)
            {
                if(MVM_fp16_setup(d_modes, mask_npix, NBmodes, &d_modesh, &d_mscale, &ldah,
                                  &l2fp16) != 0)
                {
                    processinfo_WriteMessage(processinfo, "GPU fp16 setup failed");
                    data.fpsptr->parray[fpi_GPUfp16].fpflag &= ~FPFLAG_ONOFF;
                    *GPUfp16 = 0;
                    gpufp16 = 0;
                }
                else
                {
                    printf("fp16 matrix ready, L2 %s\n", l2fp16 ? "reserved" : "reservation unavailable");
                }
            }
            else if(gpufp16 && !fp16prev && l2fp16)
            {
                // reserve the L2 again after switching back from fp32
                MVM_fp16_l2reserve(ldah, NBmodes);
            }
            if(fp16prev && !gpufp16 && l2fp16)
            {
                // let the fp32 matrix use all of the L2 again
                MVM_fp16_l2release();
            }
            fp16prev = gpufp16;
            // the reference frame always takes the step-by-step path; the split is fp32 only
            int gpusplit = (*GPUsplit >= 0) && (*GPUsplit != *GPUindex) && (initref == 1) && !gpufp16;
            if(gpusplit && mvmsplit.dev < 0)
            {
                if(setup_mvm_split(&mvmsplit, *GPUindex, *GPUsplit, mvmstream, cublasH, d_in,
                                   d_modeval, d_modes, h_in, mask_npix, NBmodes, alpha, beta,
                                   h_modeval) != 0)
                {
                    processinfo_WriteMessage(processinfo, "GPU split setup failed");
                    *GPUsplit = -1;
                    gpusplit = 0;
                }
            }
            int gpugraph = ((*GPUgraph == 1) && (initref == 1)) || gpusplit;
            cudaGraphExec_t *graphp = gpufp16 ? &mvmgraph16 : &mvmgraph;
            if(gpugraph && !gpusplit && *graphp == NULL)
            {
                *graphp = capture_mvm_graph(mvmstream, cublasH, d_in, h_in, mask_npix, d_modes,
                                            NBmodes, alpha, beta, d_modeval, h_modeval,
                                            gpufp16 ? d_modesh : NULL, d_mscale, ldah, l2fp16);
                if(*graphp == NULL)
                {
                    processinfo_WriteMessage(processinfo, "GPU graph capture failed");
                    data.fpsptr->parray[fpi_GPUgraph].fpflag &= ~FPFLAG_ONOFF;
                    *GPUgraph = 0;
                    gpugraph = 0;
                }
            }
            int gpupinned = (*GPUpinned == 1) || gpugraph; // the graph copies through the pinned buffers

            // load in_stream to GPU (the reference when one is being computed)
            float *srcpix = (initref == 0) ? data.image[IDinref].array.F : imginfloatptr;
            float *hin = gpupinned ? h_in : masked_pix;
            if(use_mask == 1)
            {
                for(uint32_t cc = 0; cc < mask_npix; ++cc)
                {
                    hin[cc] = srcpix[mask_idx[cc]];
                }
            }
            else if(gpupinned)
            {
                memcpy(hin, srcpix, sizeof(float) * mask_npix);
            }
            else
            {
                hin = srcpix;
            }
            clock_gettime(CLOCK_MILK, &t1);

            if(gpugraph)
            {
                // with the graph, kernel timing covers both copies and the MVM
                // (on the first GPU only when split)
                if(gputiming)
                {
                    cudaEventRecord(evgemv0, mvmstream);
                }
                cudaStat = cudaGraphLaunch(gpusplit ? mvmsplit.graph0 : *graphp, mvmstream);
                if(gputiming)
                {
                    cudaEventRecord(evgemv1, mvmstream);
                }
                if(gpusplit && cudaStat == cudaSuccess)
                {
                    cudaSetDevice(mvmsplit.dev);
                    cudaStat = cudaGraphLaunch(mvmsplit.graph1, mvmsplit.stream1);
                    cudaSetDevice(*GPUindex);
                }
            }
            else
            {
                cudaStat = cudaMemcpy(d_in,
                                      hin,
                                      sizeof(float) * mask_npix,
                                      cudaMemcpyHostToDevice);
            }
            clock_gettime(CLOCK_MILK, &t2);

            if(cudaStat != cudaSuccess)
            {
                printf("initref = %d    %ld  %ld\n", initref, IDinref, imgin.ID);
                printf("cudaMemcpy returned error code %d, line %d\n",
                       cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }

            // compute
            if(gputiming && !gpugraph)
            {
                cudaEventRecord(evgemv0, 0);
            }
            if(!gpugraph && gpufp16)
            {
                if(MVM_fp16_gemvT(d_modesh, d_mscale, d_in, d_modeval, ldah, NBmodes, 0, l2fp16) != 0)
                {
                    cublas_status = CUBLAS_STATUS_EXECUTION_FAILED;
                }
            }
            else if(!gpugraph)
            {
                cublas_status = cublasSgemv(cublasH,
                                            CUBLAS_OP_T,
                                            mask_npix,
                                            NBmodes,
                                            &alpha,
                                            d_modes,
                                            mask_npix,
                                            d_in,
                                            1,
                                            &beta,
                                            d_modeval,
                                            1);
            }
            if(cublas_status != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemv returned error code %d, line(%d)\n",
                       cublas_status,
                       __LINE__);
                fflush(stdout);
                if(cublas_status == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if(cublas_status == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if(cublas_status == CUBLAS_STATUS_ARCH_MISMATCH)
                {
                    printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                }
                if(cublas_status == CUBLAS_STATUS_EXECUTION_FAILED)
                {
                    printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                }

                printf("GPU index                           = %d\n", (*GPUindex));

                printf("CUBLAS_OP                           = %d\n", CUBLAS_OP_T);
                printf("alpha                               = %f\n", alpha);
                printf("alpha                               = %f\n", beta);
                printf("m                                   = %d\n", (int)m);
                printf("NBmodes                             = %d\n", (int)NBmodes);
                fflush(stdout);
                exit(EXIT_FAILURE);
            }
            if(gputiming && !gpugraph)
            {
                cudaEventRecord(evgemv1, 0);
            }

            // copy result
            imgout.md->write = 1;

            if(initref == 0)
            {
                // construct reference to be subtracted
                printf("... reference compute\n");
                cudaStat = cudaMemcpy(modevalarrayref,
                                      d_modeval,
                                      sizeof(float) * NBmodes,
                                      cudaMemcpyDeviceToHost);

                IDrefout = image_ID(outrefsname);
                if(IDrefout != -1)
                    for(long k = 0; k < NBmodes; k++)
                    {
                        modevalarrayref[k] -= data.image[IDrefout].array.F[k];
                    }

            }
            else
            {
                float *hmodeval = gpupinned ? h_modeval : modevalarray;
                if(gpugraph)
                {
                    cudaStat = cudaStreamSynchronize(mvmstream);
                    if(gpusplit && cudaStat == cudaSuccess)
                    {
                        cudaStat = cudaStreamSynchronize(mvmsplit.stream1);
                    }
                }
                else
                {
                    cudaStat = cudaMemcpy(hmodeval,
                                          d_modeval,
                                          sizeof(float) * NBmodes,
                                          cudaMemcpyDeviceToHost);
                }
                clock_gettime(CLOCK_MILK, &t3);

                for(long k = 0; k < NBmodes; k++)
                {
                    imgout.im->array.F[k] =
                        (hmodeval[k] - modevalarrayref[k]) / normcoeff[k];
                    // Renorm was never implemented
                    // (modevalarray[k] / data.image[IDintot].array.F[0] - modevalarrayref[k]) / normcoeff[k];
                }




                clock_gettime(CLOCK_MILK, &t4);

                if(gputiming)
                {
                    // compact to fit procCTRL's 35-char message column, all in us:
                    // total, gather, H2D, wait for gemv + D2H, output, GPU-side kernel time
                    // (G0h: fp16 matrix)
                    float kernelms = 0;
                    cudaEventElapsedTime(&kernelms, evgemv0, evgemv1);
                    processinfo_WriteMessage_fmt(processinfo, "G%d%s %.1f:g%.1f h%.1f w%.1f o%.1f k%.1f",
                                                 *GPUindex, gpufp16 ? "h" : "",
                                                 timespec_diff_double(t0, t4) * 1e6,
                                                 timespec_diff_double(t0, t1) * 1e6,
                                                 timespec_diff_double(t1, t2) * 1e6,
                                                 timespec_diff_double(t2, t3) * 1e6,
                                                 timespec_diff_double(t3, t4) * 1e6,
                                                 kernelms * 1e3);
                }
                else
                {
                    processinfo_WriteMessage_fmt(processinfo, "GPU%d%s %dx%d MVM %.3f us", *GPUindex,
                                                 gpufp16 ? " fp16" : "", n, m,
                                                 timespec_diff_double(t0, t4) * 1e6);
                }

                if(*twait > 0)
                {
                    // delay publication by twait us (latency and phase tests)
                    struct timespec tw;
                    do
                    {
                        clock_gettime(CLOCK_MILK, &tw);
                    } while(timespec_diff_double(t4, tw) * 1e6 < *twait);
                }

                processinfo_update_output_stream(processinfo, imgout.ID);
            }
#endif
        }

        initref = 1;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(outarray);

    free(ColMajorMatrix);

    free(normcoeff);
    free(modevalarray);
    free(modevalarrayref);

#ifdef HAVE_CUDA
    if(h_in != NULL)
    {
        cudaFreeHost(h_in);
        cudaFreeHost(h_modeval);
    }
    if(mvmgraph != NULL)
    {
        cudaGraphExecDestroy(mvmgraph);
    }
    if(mvmgraph16 != NULL)
    {
        cudaGraphExecDestroy(mvmgraph16);
    }
    if(d_modesh != NULL)
    {
        cudaFree(d_modesh);
        cudaFree(d_mscale);
    }
    if(mvmstream != NULL)
    {
        cudaStreamDestroy(mvmstream);
    }
    if(mvmsplit.dev >= 0)
    {
        cudaGraphExecDestroy(mvmsplit.graph0);
        cudaSetDevice(mvmsplit.dev);
        cudaGraphExecDestroy(mvmsplit.graph1);
        cudaStreamDestroy(mvmsplit.stream1);
        cublasDestroy(mvmsplit.cublasH1);
        cudaFree(mvmsplit.d_modes1);
        cudaFree(mvmsplit.d_in1);
        cudaFree(mvmsplit.d_modeval1);
    }
#endif


    if(imgin.md->datatype != _DATATYPE_FLOAT)
    {
        free(imginfloatptr);
    }



    if(use_mask)
    {
        free(masked_pix);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__MVMextractModes()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
