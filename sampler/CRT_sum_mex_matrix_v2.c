
#include <mex.h>
#define MAX(a,b) ((a) > (b) ? a : b)
/* L = CRT_sum_mex_matrix(X,r);   
 X is a K*N matrix, r is a 1*N vector, L is a 1*N vector */
        

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    mwSize Lenx;
    mwIndex i, j, k;
    double *ZSDS, *RND, *Lsum, *prob;
    double maxx, *r_i;    
    
    double  *pr;
    //mwIndex *ir, *jc;
    mwIndex Vsize, Nsize, Ksize;
    //mwIndex starting_row_index, stopping_row_index, current_row_index;

    pr = mxGetPr(prhs[0]);
    //ir = mxGetIr(prhs[0]);
    //jc = mxGetJc(prhs[0]); 
    Ksize = mxGetM(prhs[0]);
    Nsize = mxGetN(prhs[0]);
    r_i = mxGetPr(prhs[1]);
    
    plhs[0] = mxCreateDoubleMatrix(1, Nsize, mxREAL);
    Lsum = mxGetPr(plhs[0]);
    
    for (j=0;j<Nsize;j++) {
        for(k = 0; k < Ksize; ++k){
            if(maxx < pr[k + j * Ksize])
                maxx = pr[k + j * Ksize];
        }
        prob = (double *) mxCalloc(maxx, sizeof(double));
        for(i=0;i<maxx;i++)
            prob[i] = r_i[j]/(r_i[j]+i);
        Lsum[j] = 0;
        for(k = 0; k < Ksize; ++k){
            if(pr[k + j * Ksize] < 0.5)
                continue;
            else{
                  for(i=0;i<pr[k + j * Ksize];i++) {
                    if  ((double) rand() <= prob[i]*RAND_MAX)     
                        Lsum[j]++;
                  }
            }
        }
    }
}
