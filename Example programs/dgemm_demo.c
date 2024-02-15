#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"
//////#include <gsl/gsl_blas.h>

/* 
Matrix Matrix Multiplication

C <-- alpha * op(A)op(B) + beta * C

*/

void random_matrix(double* aa, int nn);

int main()
{
  int n;
  int z;
  double alpha=2.0, beta = 0.5;
  double    *a, *b, *c1, *c2;
  clock_t   start, finish;
  double    elapsed, sum, diff, max_diff;
  int       i, j, k;


  printf("Type in the row of the matrix:  ");
  scanf("%d", &n);
  z=n*n;

  a=(double *)calloc(z,sizeof(double));
  b=(double *)calloc(z,sizeof(double));
  c1=(double *)calloc(z,sizeof(double));
  c2=(double *)calloc(z,sizeof(double));

  printf("Generate random matrix 1 (%d x %d)\n",n,n);
  random_matrix(&a[0], z);
  printf("Generate random matrix 2 (%d x %d)\n",n,n);
  random_matrix(&b[0], z);
  for (i=0; i<z; i++) {
    c1[i] = 0.0;
    c2[i] = 0.0;
  }

  start = clock();
  printf("call cblas_dgemm\n");
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, &a[0], n, &b[0], n, beta, &c1[0], n);
  
  printf("leave cblas_dgemm\n");
  finish = clock();
  elapsed = (double)( finish - start ) / CLOCKS_PER_SEC;
  printf("CPU time used by dgemm = %f seconds\n",elapsed);

  start = clock();
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      sum = 0.0;
      for(k=0; k<n; k++)
	sum = sum + a[i*n+k] * b[k*n+j];
      c2[i*n+j] = alpha * sum + beta * c2[i*n+j];
    }
  }
  finish = clock();
  elapsed = (double)( finish - start ) / CLOCKS_PER_SEC;
  printf("CPU time used by C loops = %f seconds\n",elapsed);
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      diff = fabs( c1[i*n+j] - c2[i*n+j] );
      max_diff = ( diff > max_diff ) ? diff : max_diff;
    }
  }
  printf ("max element difference = %f\n",max_diff);

  free(a);
  free(b);
  free(c1);
  free(c2);

  return 0;
}

void random_matrix(double* aa, int nn)
{
  int i;
  for (i=0; i<nn; i++) 
    aa[i] = (double)( rand() ) / RAND_MAX;
}
