/*==========================================================
 * Multrnd_Matrix_mex.c - 
 *
 *
 * The calling syntax is:
 *
 *		[ZSDS,WSZS] = Multrnd_Matrix_mex_fast(Xtrain,Phi,Theta);
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2012 Mingyuan Zhou
 *
 *========================================================*/
/* $Revision: 0.1 $ */

#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include "matrix.h"*/
//#include "cokus.c"
//#define RAND_MAX_32 4294967295.0

/* //  The computational routine 
//void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, mwIndex *ir, mwIndex *jc, double *pr, mwSize Vsize, mwSize Nsize, mwSize Ksize, double *RND, double *prob_cumsum) //, mxArray **lhsPtr, mxArray **rhsPtr)*/


mwIndex BinarySearch(double probrnd, double *prob_cumsum, mwSize Ksize) {
    mwIndex k, kstart, kend;
    if (probrnd <=prob_cumsum[0])
        return(0);
    else {
        for (kstart=1, kend=Ksize-1; ; ) {
            if (kstart >= kend) {
                /*//k = kend;*/
                return(kend);
            }
            else {
                k = kstart+ (kend-kstart)/2;
                if (prob_cumsum[k-1]>probrnd && prob_cumsum[k]>probrnd)
                    kend = k-1;
                else if (prob_cumsum[k-1]<probrnd && prob_cumsum[k]<probrnd)
                    kstart = k+1;
                else
                    return(k);
            }
        }
    }
    return(k);
}

void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, double *pr, mwSize Vsize, mwSize Nsize, mwSize Ksize,  double *prob_cumsum) 
/*//, mxArray **lhsPtr, mxArray **rhsPtr)*/
{     
    double cum_sum, probrnd;
    mwIndex k, i, j, token; 
	/*//, ksave;*/
    mwIndex starting_row_index, stopping_row_index, current_row_index;
      
    for (j=0;j<Nsize;j++) {
        for(i = 0; i < Vsize; ++i){
            if(pr[i + j * Vsize] < 0.5)
                continue;
            else{
                cum_sum = 0;
                for(k = 0; k < Ksize; ++k){
                    cum_sum += Phi[i+ k*Vsize]*Theta[k + Ksize*j];
                    prob_cumsum[k] = cum_sum;
                }
                for (token=0;token< pr[i + j * Vsize];token++) {
                    /*//probrnd = RND[ji]*cum_sum;*/
                    probrnd = (double) rand()/RAND_MAX*cum_sum;      
                    k = BinarySearch(probrnd, prob_cumsum, Ksize);   
                    
                    ZSDS[k+Ksize*j]++;
                    WSZS[i+k*Vsize]++;
                }
            }
        }
    }
   /*// mexPrintf("total=%d, Ji = %d",total,ji);*/
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *ZSDS, *WSZS, *Phi, *Theta, *RND;
    double  *pr, *prob_cumsum;
    //mwIndex *ir, *jc;
    mwIndex Vsize, Nsize, Ksize;
    
     pr = mxGetPr(prhs[0]);
     //ir = mxGetIr(prhs[0]);
     //jc = mxGetJc(prhs[0]);        
     Vsize = mxGetM(prhs[0]);
     Nsize = mxGetN(prhs[0]);
     Ksize = mxGetN(prhs[1]);
     Phi = mxGetPr(prhs[1]);
     Theta = mxGetPr(prhs[2]);
    /*// RND = mxGetPr(prhs[3]);*/
    
    plhs[0] = mxCreateDoubleMatrix(Ksize,Nsize,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Vsize,Ksize,mxREAL);
    ZSDS = mxGetPr(plhs[0]);
    WSZS = mxGetPr(plhs[1]);
    
    prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));

   //Multrnd_Matrix(ZSDS, WSZS, Phi, Theta, ir, jc, pr,  Vsize, Nsize, Ksize,  prob_cumsum); 
    Multrnd_Matrix(ZSDS, WSZS, Phi, Theta, pr,  Vsize, Nsize, Ksize,  prob_cumsum); 
   /*//, &lhs, &rhs); */
}