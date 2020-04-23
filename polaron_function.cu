//CUDA version of polaron code from Dr. Zoom
// this code require CUDA-aware MPI technology from Nvidia

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <time.h>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <iomanip>
#include <unistd.h>
#include <algorithm>    // std::min
#include <sys/stat.h>
#include <dirent.h>
#include <chrono> 
#include <cuda_runtime.h> 

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"

using namespace std;
using namespace std::chrono; 


extern "C"  __global__
void abs_osc( int sys_kount, double d_sys_h[], 
    double d_ux[], double d_uy[], 
    double d_ab_osc_x[], double d_ab_osc_y[] )
{

    int state, hx, istart, istride; 

    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    // Loop over the states to the current thread
    for ( state = istart ; state < sys_kount; state += istride ){
        d_ab_osc_x[state] = 0.0; 
        d_ab_osc_y[state] = 0.0;
        // ab_osc_z[state] = 0.0; 
        
        if(state == 0 ) continue; 

        for ( hx = 0; hx < sys_kount; hx += 1 ) {
            d_ab_osc_x[state] += d_ux[hx] * d_sys_h[hx+sys_kount*state] * d_sys_h[hx];
            d_ab_osc_y[state] += d_uy[hx] * d_sys_h[hx+sys_kount*state] * d_sys_h[hx];
            // ab_osc_z[state] += uz[hx] * d_sys_h[hx+sys_kount*state] * d_sys_h[hx];
        }
        d_ab_osc_x[state] = pow(d_ab_osc_x[state],2);
        d_ab_osc_y[state] = pow(d_ab_osc_y[state],2);
        // ab_osc_z[state] = pow(ab_osc_z[state],2);
    }
}

extern "C"  __global__
void abs_spectra( int sys_kount, double d_ab_osc_x[], double d_ab_osc_y[], double d_sys_eval[],
    double d_ab_x[], double d_ab_y[], 
    int spec_step, double spec_start_ab, double spec_end_ab, double abs_lw )
{
    
    int spec_point;
    double energy, tran_e, lineshape; 
    const double pi = atan(1.00) * 4;

    int state; 
    
    //absorption spectrum  
    // printf(" %lf %lf %d %lf",spec_start_ab, spec_end_ab, spec_step, abs_lw  ); 
    
    // for (state = 0 ; state < sys_kount; state += 1){
        // tran_e = d_sys_eval[state]-d_sys_eval[0];
        // printf(" %lf ", tran_e ); 
    // }
    // memset(d_ab_x, 0.0, sizeof(d_ab_x));
    // memset(d_ab_y, 0.0, sizeof(d_ab_y));
    
    for (spec_point = 0; spec_point < spec_step; spec_point++){
        d_ab_x[spec_point] = 0.0; 
        d_ab_y[spec_point] = 0.0;
        // printf(" %lf",d_ab_x[spec_point]); 
        // ab_z[spec_point] = 0.0; 
    }
    
    for (spec_point = 0; spec_point < spec_step; spec_point++){
        d_ab_x[spec_point] = 0.0; 
        d_ab_y[spec_point] = 0.0;
        // ab_z[spec_point] = 0.0; 
        // printf(" %lf",d_ab_x[spec_point]); 
        energy = spec_start_ab + (spec_end_ab - spec_start_ab)/spec_step*(spec_point+1); 
        for(state = 1; state < sys_kount; state++ ){
            
            tran_e = d_sys_eval[state] - d_sys_eval[0];
            lineshape = abs_lw/(pow((energy-tran_e),2)+pow(abs_lw,2))/pi;
            
            d_ab_x[spec_point] += lineshape * d_ab_osc_x[state] * tran_e;
            d_ab_y[spec_point] += lineshape * d_ab_osc_y[state] * tran_e;

        }
        // printf(" %lf \n", d_ab_x[spec_point]); 
    }

}	

void magma_diagonalization( int sys_kount, double sys_h[], 
    double ux[], double uy[], double ab_x[], double ab_y[], 
    int spec_step, double spec_start_ab, double spec_end_ab, double abs_lw)
{

// ***********************************************
    // copy Hamiltonian to device
// ***********************************************
    int blockSize = 128; 
    magma_init(); // magma_queue_create( 0, &queue ); 

    #define MALLOC_ERR  { printf(">>> ERROR on CPU: out of memory.\n"); exit(EXIT_FAILURE);}
    #define CHK_MERR    if (Merr != MAGMA_SUCCESS ) { printf(">>> ERROR on MAGMA: %s.\n", magma_strerror(Merr)); exit(EXIT_FAILURE);}
    cudaError_t         Cuerr;
    #define CHK_ERR     if (Cuerr != cudaSuccess ) { printf(">>> ERROR on CUDA: %s.\n", cudaGetErrorString(Cuerr)); exit(EXIT_FAILURE);}
    int                 Merr;
    
    
    int state; 
    // magma variables for dsyevd
    magma_queue_t       queue;
    
    
    double         *d_sys_h;                                                       // the hamiltonian on the GPU
    double         aux_work[1];                                                    // To get optimal size of lwork
    magma_int_t         aux_iwork[1], info;                                             // To get optimal liwork, and return info
    magma_int_t         lwork, liwork;                                                  // Leading dim of kappa, sizes of work arrays
    magma_int_t         *iwork;                                                         // Work array
    double         *work;                                                          // Work array
    double         *d_sys_eval   ;                                                          // Eigenvalues
    double         *wA  ;                                                          // Work array
    const int numBlocks = (sys_kount+blockSize-1)/blockSize;
    
    double *d_ux;
    double *d_uy; 
    double *d_ab_x;
    double *d_ab_y;
    // double *d_sys_eval; 

    double *d_ab_osc_x; 
    double *d_ab_osc_y;
    //double         *d_w;                                                           // Eigenvalues on the GPU

    // for (state = 0; state < sys_kount; state++){
        // // printf(" %d %lf \n", state, *(w+state)); 
        // printf(" %lf \n", ux[state]); 
        // // d_sys_eval[state] = *(w + state) - *(w);
        // // printf(" %lf \n", d_sys_eval[state]);
    // }
    
    // cout << sys_kount << endl; 
    magma_queue_create( 0, &queue ); 
    int SSYEVD_ALLOC_FLAG = 1;     // flag whether to allocate ssyevr arrays -- it is turned off after they are allocated
    
    magma_int_t sys_kount2 = (magma_int_t) sys_kount*sys_kount; 
    //allocate memory in GPU
    Cuerr = cudaMalloc (&d_sys_h, sys_kount2*sizeof(double)); CHK_ERR;
    Cuerr = cudaMalloc (&d_ux, sys_kount*sizeof(double)); CHK_ERR;
    Cuerr = cudaMalloc (&d_uy, sys_kount*sizeof(double)); CHK_ERR;
    Cuerr = cudaMalloc (&d_ab_x, spec_step*sizeof(double)); CHK_ERR;
    Cuerr = cudaMalloc (&d_ab_y, spec_step*sizeof(double)); CHK_ERR;
    // Cuerr = cudaMalloc (&d_sys_eval, sys_kount*sizeof(double)); CHK_ERR;
    // Cuerr = cudaMalloc (&d_sys_eval, sys_kount*sizeof(double)); CHK_ERR;
    Cuerr = cudaMalloc (&d_ab_osc_x, sys_kount*sizeof(double)); CHK_ERR;
    Cuerr = cudaMalloc (&d_ab_osc_y, sys_kount*sizeof(double)); CHK_ERR;
    
// just in master thread
//if(myid == MASTER){
    cudaMemcpy( d_sys_h, sys_h, sys_kount2* sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( d_ux, ux, sys_kount * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( d_uy, uy, sys_kount * sizeof(double), cudaMemcpyHostToDevice );
    // cudaMemcpy( d_sys_eval, sys_eval, sys_kount * sizeof(double), cudaMemcpyHostToDevice );
    
    // for (state = 0; state < sys_kount; state++){
        // // printf(" %d %lf \n", state, *(w+state)); 
        // printf(" %lf \n", d_sys_h[state]); 
        // // d_sys_eval[state] = *(w + state) - *(w);
        // // printf(" %lf \n", d_sys_eval[state]);
    // }
//}
// broadcast to salve threads
//MPI_Bcast( d_sys_h, sys_kount2, MPI_DOUBLE_PRECISION, MASTER, MPI_COMM_WORLD); 

// ***********************************************
    // add disorder into each Hamiltonian
// ***********************************************
    // test
    // magma_dprint_gpu((magma_int_t) sys_kount,(magma_int_t) sys_kount,
    // d_sys_h,(magma_int_t) sys_kount,queue); 	


// ***********************************************
    // diagonalize in parallel
// ***********************************************

    // if the first time, query for optimal workspace dimensions
    if ( SSYEVD_ALLOC_FLAG )
    {   
        magma_dsyevd_gpu( MagmaVec, MagmaUpper, (magma_int_t) sys_kount, NULL, (magma_int_t) sys_kount, 
            NULL, NULL, (magma_int_t) sys_kount, aux_work, -1, aux_iwork, -1, &info );
        
        lwork  = (magma_int_t) MAGMA_D_REAL( aux_work[0] );
        liwork  = aux_iwork[0];

        // allocate work arrays, eigenvalues and other stuff
        
        Merr = magma_imalloc_cpu   ( &iwork, liwork ); CHK_MERR; 
        Merr = magma_dmalloc_pinned( &wA , sys_kount2 ) ; CHK_MERR;
        Merr = magma_dmalloc_pinned( &d_sys_eval,     sys_kount ); CHK_MERR; 
        Merr = magma_dmalloc_pinned( &work , lwork  ); CHK_MERR;

        SSYEVD_ALLOC_FLAG = 0;      // is allocated here, so we won't need to do it again
        //cout<< "Hamiltonian" <<endl;

        // get info about space needed for diagonalization
        size_t freem, total;
        cudaMemGetInfo( &freem, &total );
        printf("\n>>> cudaMemGetInfo returned\n"
               "\tfree:  %g gb\n"
               "\ttotal: %g gb\n", (double) freem/(1E9), (double) total/(1E9));
        printf(">>> %g gb needed by diagonalization routine.\n", (double) (lwork * (double) sizeof(double)/(1E9)));
    }

    magma_dsyevd_gpu( MagmaVec, MagmaUpper, (magma_int_t) sys_kount, d_sys_h, (magma_int_t) sys_kount,
        d_sys_eval, wA, (magma_int_t) sys_kount, work, lwork, iwork, liwork, &info );

    // for (state = 0; state < sys_kount; state++){
        // printf(" %d %lf \n", state, *(w+state)); 
        // printf(" %lf \n", d_ux[state]); 
        // d_sys_eval[state] = *(w + state) - *(w);
        // printf(" %lf \n", d_sys_eval[state]);
    // }
    
    if ( info != 0 ){ printf("ERROR: magma_dsyevd_gpu returned info %lld.\n", info ); exit(EXIT_FAILURE);}
                
    // copy eigenvalues to device memory
//    cudaMemcpy( sys_h, d_sys_h, sys_kount2* sizeof(double), cudaMemcpyDeviceToHost );
    
    // test
    // magma_dprint_gpu((magma_int_t) sys_kount,(magma_int_t) sys_kount,
    // d_sys_h,(magma_int_t) sys_kount,queue); 	
    


    // for (state = 0; state < sys_kount; state++){
        // // printf(" %d %lf \n", state, *(w+state)); 
        // printf(" %lf \n", d_ux[state]); 
        // // d_sys_eval[state] = *(w + state) - *(w);
        // // printf(" %lf \n", d_sys_eval[state]);
    // }

    // for (state = 0; state < sys_kount; state++){
        // printf(" %lf \n", *(w + state) - *(w));
        // d_sys_eval[state] = *(w + state) - *(w); 
        // printf(" %lf \n", d_sys_eval[state]);
    // }
    

// ***********************************************
    // absorption oscillator strength & spectra
// ***********************************************
abs_osc<<<numBlocks, blockSize>>> ( sys_kount, d_sys_h, 
    d_ux, d_uy, d_ab_osc_x, d_ab_osc_y); 
    
// magma_dprint_gpu((magma_int_t) sys_kount,1,
    // d_ab_osc_x,(magma_int_t) sys_kount,queue); 	
// magma_dprint_gpu((magma_int_t) sys_kount,1,
    // d_ab_osc_y,(magma_int_t) sys_kount,queue); 	
// magma_dprint_gpu((magma_int_t) sys_kount,1,
    // d_sys_eval,(magma_int_t) sys_kount,queue); 	
// for (state = 0; state < sys_kount; state++){
    // printf(" %lf \n", d_ux[state]); 
    // // d_sys_eval[state] = *(w + state) - *(w);
// }

abs_spectra<<<1, 1>>>(sys_kount, d_ab_osc_x, d_ab_osc_y, d_sys_eval , 
    d_ab_x, d_ab_y, 
    spec_step, spec_start_ab, spec_end_ab, abs_lw ); 
// printf(" %lf \n",abs_lw);  
// magma_dprint_gpu((magma_int_t) sys_kount,1,
    // d_ab_osc_x,(magma_int_t) sys_kount,queue); 	 	
// magma_dprint_gpu((magma_int_t) sys_kount,1,
//     d_sys_eval,(magma_int_t) sys_kount,queue); 

// magma_dprint_gpu((magma_int_t) spec_step,1,
    // d_ab_x,(magma_int_t) spec_step,queue);

cudaMemcpy( ab_x, d_ab_x, spec_step* sizeof(double), cudaMemcpyDeviceToHost );
cudaMemcpy( ab_y, d_ab_y, spec_step* sizeof(double), cudaMemcpyDeviceToHost );
    // // ***********************************************
        // // free memory on the GPU 
    // // ***********************************************
    magma_queue_destroy( queue );
    // free memory on the CPU and GPU and finalize magma library
    cudaFree(d_sys_h);
    cudaFree(d_ux);
    cudaFree(d_uy);

    cudaFree(d_ab_x);
    cudaFree(d_ab_y);
    cudaFree(d_sys_eval);

    cudaFree(d_ab_osc_x);
    cudaFree(d_ab_osc_y);

    if ( SSYEVD_ALLOC_FLAG == 0 )
    {
        cudaFree(d_sys_eval);
        free(iwork);
        magma_free_pinned( work );
        magma_free_pinned( wA );
    }
    // final call to finalize magma math library
    magma_finalize();
    
}

