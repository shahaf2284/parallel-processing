#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

#define AA(i,j) AA[(i)*M+(j)]


int main(int argc, char **argv) {
   int i, j, k;
/************  MPI ***************************/
   int myrank_mpi, nprocs_mpi;
   MPI_Init( &argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
/************  BLACS ***************************/
   int ictxt, nprow, npcol, myrow, mycol,nb;
   int info,itemp;
   int ZERO=0,ONE=1;
   nprow = 2; npcol = 2; nb =2;
   Cblacs_pinfo( &myrank_mpi, &nprocs_mpi ) ;
   Cblacs_get( -1, 0, &ictxt );
   Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
   Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
   int M=5;
   double *AA = (double*) malloc(M*M*sizeof(double));
   for(i=0;i<M;i++ )
     for(j=0;j<M;j++)
        AA[i*M+j]=(2*i+3*j+1);

   double *X = (double*) malloc(M*sizeof(double));
   X[0]=1;X[1]=1;X[2]=0;X[3]=0;X[4]=1;

   int descA[9],descx[9],descy[9];
   /*  Guy , Numroc:
   *  Purpose
*  =======
*
*  NUMROC computes the NUMber of Rows Or Columns of a distributed
*  matrix owned by the process indicated by IPROC.
*
*  Arguments
*  =========
*
*  N         (global input) INTEGER
*            The number of rows/columns in distributed matrix.
*
*  NB        (global input) INTEGER
*            Block size, size of the blocks the distributed matrix is
*            split into.
*
*  IPROC     (local input) INTEGER
*            The coordinate of the process whose local array row or
*            column is to be determined.
*
*  ISRCPROC  (global input) INTEGER
*            The coordinate of the process that possesses the first
*            row or column of the distributed matrix.
*
*  NPROCS    (global input) INTEGER
*            The total number processes over which the matrix is
*            distributed.
*
*  =====================================================================
*/
   int mA = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int nA = numroc_( &M, &nb, &mycol, &ZERO, &npcol );
   int nx = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   int my = numroc_( &M, &nb, &myrow, &ZERO, &nprow );
   /* Guy: DESCINIT initializes the descriptor vector   */
   descinit_(descA, &M,   &M,   &nb,  &nb,  &ZERO, &ZERO, &ictxt, &mA,  &info);
   descinit_(descx, &M, &ONE,   &nb, &ONE,  &ZERO, &ZERO, &ictxt, &nx, &info);

   descinit_(descy, &M, &ONE,   &nb, &ONE,  &ZERO, &ZERO, &ictxt, &my, &info);
   double *x = (double*) malloc(nx*sizeof(double));
   double *y = (double*) calloc(my,sizeof(double));
   double *A = (double*) malloc(mA*nA*sizeof(double));
   int sat,sut;
   for(i=0;i<mA;i++)

   for(j=0;j<nA;j++){
                sat= (myrow*nb)+i+(i/nb)*nb;
                sut= (mycol*nb)+j+(j/nb)*nb;
                A[j*mA+i]=AA(sat,sut);
        }


   for(i=0;i<nx;i++){
                sut= (myrow*nb)+i+(i/nb)*nb;
                x[i]=X[sut];
        }

   double alpha = 1.0; double beta = 0.0;
   pdgemv_("N",&M,&M,&alpha,A,&ONE,&ONE,descA,x,&ONE,&ONE,descx,&ONE,&beta,y,&ONE,&ONE,descy,&ONE);

   Cblacs_barrier(ictxt,"A");
   for(i=0;i<my;i++)
   printf("rank=%d %.2f \n", myrank_mpi,y[i]);
   Cblacs_gridexit( 0 );
   MPI_Finalize();
   return 0;
}
