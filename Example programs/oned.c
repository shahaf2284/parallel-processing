/*******************************************************************    
!   oned.f - a solution to the Poisson equation using Jacobi             
!   iteration on a 1D decomposition (the y-dimension is split)
!                                                                       
!   The size of the computational domain is read by the master and 
!   broadcast to all other processors.  The Jacobi iteration is run
!   until the change in successive elements is below the tolerance 
!   The difference is printed out every 100th or 1000th step. 
!
!   The Poisson equation in 2D: u(x,y)_xx+u(x,y)_yy=f(x,y)
!                               u(x,y)=g(x,y) on the boundary (see onedinit)
! 
!   C-version by Michael Hanke 2007-06-21
!   Polished by Ulf Andersson 2002-08-30
!   Original version written by anonymous.
!
!   usage: oned [nx [ny]]
!          nx - number of grid points in x-direction
!          ny - number of grid points in y-direction
!          If only nx is given, ny = nx.
!          If no argument is given, nx will be taken from stdin.
!
!*******************************************************************    
*/

/* system headers */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

/* Use MPI */
#include "mpi.h"

/* some constants */
#define NDIM      1   /* space dimension */
#define master_id 0   /* master process */
#define stag      0   /* some tags */
#define etag      1
#define ntag      2
#define tolerance 1e-6 /* accuracy for the iteration */
#define ITMAX     10000 /* maximal number of iterations */
#define TRUE      1
#define FALSE     0

/* a macro */
#define CK (clock()/((double) CLOCKS_PER_SEC))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/*! This is a routine for producing a decomposition of n on nprocs processes.
! The values returned assume a "global" domain in [1:n]
! Example of decomposition:
! n=30, nprocs=4, gives nlocal = 8, 8, 7, 7
!                            s = 1, 9,17,24
!                            e = 8,16,23,30 
! Note that since no work is done in the ghostcells, they are not considered 
! when doing the decomposition.
*/

static void mpe_decomp1d(int n, int nproc, int coord, int *s, int *e)
{
    int deficit, nlocal;

    nlocal = n/nproc;
    *s = coord*nlocal+1;
    deficit = n%nproc;
    *s = *s+MIN(coord, deficit);
    if (coord < deficit) nlocal = nlocal+1;
    *e = *s+nlocal-1;
}

static double diff(double *a, double *b, int nx, int sy, int ey)
{
    double sum;
    int i, jj;

    sum = 0.0;
    for (jj = 1; jj <= ey-sy+1; jj++)
	for (i = 1; i <= nx; i++)
	    sum += (a[i+jj*(nx+2)]-b[i+jj*(nx+2)])*(a[i+jj*(nx+2)]-b[i+jj*(nx+2)]);
    return sum;
}

static void exchng1(double *ab, int nx, int sy, int ey, MPI_Comm comm1d,
		    int belowY, int aboveY)
{
    int ierr;
    MPI_Status status;

/* Send nx values ab(1:nx,ey) upwards. Note that these nx elements are stored
   contiguously in memory. */
    ierr = MPI_Sendrecv(ab+(1+(ey-sy+1)*(nx+2)), nx, MPI_DOUBLE, aboveY, 0,
			ab+1, nx, MPI_DOUBLE, belowY, 0,
			comm1d, &status);
/* Send nx values ab(1:nx,sy) downwards. Note that these nx elements are stored
   contiguously in memory. */
	ierr = MPI_Sendrecv(ab+(1+nx+2), nx, MPI_DOUBLE, belowY, 1,
			    ab+(1+(ey-sy+2)*(nx+2)), nx, MPI_DOUBLE, aboveY, 1,
			    comm1d, &status);
}

/*
! Initialization routine. Also sets the constant boundary conditions.
!
! If we use f===0, we are in fact solving the Laplace equation.
! Our boundary conditions u(0,y)=1
!                         u(1,y)=0
!                         u(x,0)=1
!                         u(x,1)=0
*/

static void onedinit(double * a, double *b, double *f, int nx, int sy, int ey)
{
    int i, arrsize;

    arrsize = (ey-sy+3)*(nx+2);
    for (i = 0; i < arrsize; i++) {
	a[i] = 0.0;
	b[i] = 0.0;
	f[i] = 0.0;
    }

/* handle boundary conditions */
    for (i = 0; i < arrsize-(nx+2); i += nx+2) {
	a[i] = 1.0;
	b[i] = 1.0;
    }
    if (sy == 1)
	for (i = 1; i <= nx; i++) {
	    a[i] = 1.0;
	    b[i] = 1.0;
	}
}

/* Perform a Jacobi sweep for a 1D decomposition.                       
   Sweep from old into new */
static void sweep1d(double *old, double *f, int nx, int sy, int ey, double *new)
{
    double h;
    int i, j;

    h = 1.0/(nx+1);
    for (j = 1; j <= ey-sy+1; j++ )
	for (i = 1; i <= nx; i++)
	    new[i+j*(nx+2)] = 0.25*(old[i-1+j*(nx+2)]+old[i+1+j*(nx+2)]+
				    old[i+(j-1)*(nx+2)]+old[i+(j+1)*(nx+2)]-
				    h*h*f[i+j*(nx+2)]);
}

int main(int argc, char *argv[])
{
    double diffnorm = 1.0, dwork, t1, t2, c1, c2, rtmax, ctmax;
    int ey, ierr, it, my_id, belowY, aboveY, numprocs, nx = 0, ny = 0,
	sy, py, dist, dir, i, jj, sender_id, arrsize;
    char reorder;
    double *a, *b, *f;
    int dims[NDIM], coords[NDIM];
    int periodic[NDIM];
    MPI_Comm comm1d;
    MPI_Status status;
    FILE *fid;


/* initialize MPI */
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

/* user argument processing */
    if (argc > 1) {
	nx = atoi(argv[1]);
	if (nx < 0) nx = 0;
	if (argc > 2) {
	    ny = atoi(argv[2]);
	    if (ny <= 0) ny = nx;
	}
	else ny = nx;
    }

/* user input */
    if (nx == 0) {
	if (my_id == master_id) {
	    printf("Enter nx: ");
	    scanf("%d", &nx);
	}
	ierr = MPI_Bcast(&nx, 1, MPI_INT, master_id, MPI_COMM_WORLD);
	if (nx <= 0) {
	    MPI_Finalize();
	    if (my_id == master_id) {
		printf("nx out of range...\n");
		return(1);
	    }
	    return(0);
	}
	ny = nx;  /* a square domain */
    }

/* get a new communicator for a decomposition of the domain */
    periodic[0] = FALSE;
    reorder = TRUE;
    dims[0] = numprocs;
    ierr = MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, periodic, reorder,
			   &comm1d);

/* get my position in this communicator, and my neighbors */
    ierr = MPI_Comm_rank(comm1d, &my_id);
    ierr = MPI_Cart_coords(comm1d, my_id, NDIM, coords);
    dir = 0;
    dist = 1;
    ierr = MPI_Cart_shift(comm1d, dir, dist, &belowY, &aboveY);

/* Compute the actual decomposition */
    mpe_decomp1d(ny, dims[0], coords[0], &sy, &ey);
    printf("Task: %d, column %d to %d\n", my_id, sy, ey);

/* Allocate node local arrays with ghost cells. Note, we have a two-dimensional
   grid which we distribute on a one-dimensional virtual Cartesian topology. */
    arrsize = (nx+2)*(ey-sy+3);
    a = (double *) malloc(arrsize*sizeof(double));
    b = (double *) malloc(arrsize*sizeof(double));
    f = (double *) malloc(arrsize*sizeof(double));

/* Initialize the right-hand-side (f) and the initial solution guess (a) */
    onedinit(a, b, f, nx, sy, ey);

/* Actually do the computation.  Note the use of a collective operation to 
   check for convergence, and a do-loop to bound the number of iterations */
    ierr = MPI_Barrier(comm1d);
    t1 = MPI_Wtime();
    c1 = CK;

    for (it = 0; (it < ITMAX) && (diffnorm > tolerance); it += 2) {
	exchng1(a, nx, sy, ey, comm1d, belowY, aboveY);
	sweep1d(a, f, nx, sy, ey, b);
	exchng1(b, nx, sy, ey, comm1d, belowY, aboveY);
	sweep1d(b, f, nx, sy, ey, a);
	dwork = diff(a, b, nx, sy, ey);
	ierr = MPI_Allreduce(&dwork, &diffnorm, 1, MPI_DOUBLE, MPI_SUM,
			     comm1d);
	if (my_id == master_id) {
	    if ((it <= 1000) & (it%100 == 0))
		printf("%d its. Difference is %f\n", it, diffnorm);
	    if ((it > 1000) & (it%1000 == 0))
		printf("%d its. Difference is %f\n", it, diffnorm);
	}
    }

/* finish */
    c2 = CK-c1;
    t2 = MPI_Wtime()-t1;
    ierr = MPI_Reduce(&c2, &ctmax, 1, MPI_DOUBLE, MPI_MAX, master_id, comm1d);
    ierr = MPI_Reduce(&t2, &rtmax, 1, MPI_DOUBLE, MPI_MAX, master_id, comm1d);

    if (my_id == master_id) {
	printf("\nConverged after %d iterations\n", it);
	printf("%d its. Difference is %f\n", it, diffnorm);
	printf("CPU time (s): %f\n", ctmax);
	printf("Real time (s): %f\n", rtmax);

/* A safety check*/
	if (coords[0] != master_id) {
/* If this happens, the program will hang!! */
	    printf("FATAL ERROR! This code assumes that the process with rank 0\n");
	    printf("also have coord=0 in the Cartesian virtual topology\n");
	    printf("rank: %d, coord: %d\n", my_id, coords[0]);
	    return(1);
	}
/* output of result */
	fid = fopen("oned.cout","w");
/* Here, the success should be tested. Otherwise the program may hang! */

/* Write the master's part of the solution to file oned.out */
/* output order: column first */
/*!!!!!!!!!!!!!!!!!! indexing in y-direction !!!!!!!!!!!!!!!!!*/
	for (jj = 1; jj <= ey; jj++) {
	    for (i = 1; i <= nx; i++) fprintf(fid, "%f ", b[i+jj*(nx+2)]);
            fprintf(fid, "\n");
        }
/* wait for the parts from the slaves */
	for (py = 1; py < dims[0]; py++) {
	    ierr = MPI_Cart_rank(comm1d, &py, &sender_id);
	    ierr = MPI_Recv(&sy, 1, MPI_INT, sender_id, stag, comm1d, &status);
	    ierr = MPI_Recv(&ey, 1, MPI_INT, sender_id, etag, comm1d, &status);
	    ierr = MPI_Recv(b, arrsize, MPI_DOUBLE,
			    sender_id, ntag, comm1d, &status); /* Obs: b long enough?*/
	    for (jj = 1; jj <= ey-sy+1; jj++) {
		for (i = 1; i <= nx; i++) fprintf(fid, "%f ", b[i+jj*(nx+2)]);
                fprintf(fid, "\n");
            }
	}
	fclose(fid);
    }
    else {
/* on the slaves */
	ierr = MPI_Ssend(&sy, 1, MPI_INT, master_id, stag, comm1d);
	ierr = MPI_Ssend(&ey, 1, MPI_INT, master_id, etag, comm1d);
	ierr = MPI_Ssend(b, arrsize, MPI_DOUBLE,
		  master_id, ntag, comm1d);
    }

/* That's it */
    free(f);
    free(b);
    free(a);
    MPI_Finalize();
    return 0;
}
