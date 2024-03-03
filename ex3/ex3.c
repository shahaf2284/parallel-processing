#include "stdio.h"
#include "math.h"
#include <omp.h>
#include <stdlib.h>
#include <time.h>

//========================HOW TO RUN THE PROJECT =========================
//use gcc -o ex3 -fopenmp ex3.c -lm
//export OMP_NUM_THREADS=4 (or 2 or 6 or whatever)
//ex3 N dt

//========================define-Units====================================

#define LY 9*pow(10,12)              // km 
#define AvgSpeed 200000              // m/sec 
#define Mass 2*pow(10,30)            // kilogram /Each star has a mass equal 
#define Gravity 6.674*pow(10,-11)    // N*m^2/kg^2
#define Colollision  1000            //
#define LowerBound  1*LY
#define UpperBound 100*LY

//=========================structures==================================== 
    
typedef struct Star{
    long double mass; 
    double speed;
    double x_location;
    double y_location;
    double v_x;          // Speed in x direction |V|*cos(t)
    double v_y;          // Speed in x direction |V|*sin(t)
    double a_x;          // acc in x direction
    double a_y;          // acc in y direction
    }Star;

//=========================Prototypes==================================
void simulator();
Star* initStar(int numStar);
void StarsMovment(Star* stars,int numStar,double dt);
void calculateforces(Star* stars,int numStar,double dt);
void writeToFile(Star* stars,int numStar,FILE* fp);

//=========================Function=====================================

Star* initStar(int numStar){
    Star* stars = (Star*)malloc(sizeof(Star)*numStar);
//    srand(time(NULL));
    double t;
    double v;
    #pragma omp parallel for private(t,v)
    for(int i=0;i<numStar;i++){
        t = ((double)rand() /(double) RAND_MAX) * 2 * M_PI;
        v = (((double)rand()/(double)RAND_MAX)+0.5)*AvgSpeed;  //this should be between 0.5V and 1.5V
        stars[i].mass = Mass;
        stars[i].x_location = ((double)rand()/(double)RAND_MAX)*100*LY;      
        stars[i].y_location = ((double)rand()/(double)RAND_MAX)*100*LY;
        stars[i].v_x = v*cos(t);           // |V|*cos(t)
        stars[i].v_y = v*sin(t);           // |V|*sin(t)
        stars[i].a_x = 0.0;     
        stars[i].a_y = 0.0;
    }
    return stars;
}

void StarsMovment(Star* stars,int numStar,double dt){
    // calculate the star movment base on the forumla x(t)=x0+v*t+1/2*a*t^2
    
    
    #pragma omp parallel for
    for(int i=0;i<numStar;i++){
        if(stars[i].x_location > UpperBound){
            stars[i].x_location = LowerBound;}
            
        if(stars[i].x_location < LowerBound){
            stars[i].x_location = UpperBound;}
            
        if(stars[i].y_location > UpperBound){
            stars[i].y_location = LowerBound;}
            
        if(stars[i].y_location < LowerBound){
            stars[i].y_location = UpperBound;}
            
        // Calculation of the position of the star,considering the acceleration
        stars[i].x_location = stars[i].x_location + stars[i].v_x*dt+ 0.5*pow(dt,2)*(stars[i].a_x);
        stars[i].y_location = stars[i].y_location + stars[i].v_y*dt+ 0.5*pow(dt,2)*(stars[i].a_y);
    }
    #pragma omp parallel for
    for(int i=0;i<numStar;i++){
        //v = v_0 + a*t 
        stars[i].v_x = stars[i].v_x+(stars[i].a_x)*dt;
        stars[i].v_y = stars[i].v_y+(stars[i].a_y)*dt;
    }
}

void calculateforces(Star* stars,int numStar,double dt){
    #pragma omp parallel for
    for(int i=0;i<numStar;i++){
        for(int j=0;j<numStar;j++){
            if(i==j){
                continue;
            }
        // |x1-x2|^2+|y1-y2|^2
        double x = pow(stars[i].x_location - stars[j].x_location,2);
        double y = pow(stars[i].y_location - stars[j].y_location,2);
        double distances = (double)1/sqrt(x+y);
        distances = pow(distances,3);
        long double Temp = Gravity*Mass*distances;
        stars[i].a_x += Temp*(stars[j].x_location-stars[i].x_location);
        stars[i].a_y += Temp*(stars[j].y_location-stars[i].y_location);
        }
    }
}

void writeToFile(Star* stars,int numStar,FILE* fp){
    //write positions to a file so we can display them with python or smth
    for(int i=0;i<numStar;i++){
        fprintf(fp,"%lf %lf \n",stars[i].x_location ,stars[i].y_location);
    }
    fprintf(fp,"\n");
}

void simulator(int numStar,double dt){
    int simulations = 1000;
    FILE* fp = fopen("positionsStar.txt","w");
    Star* Stars = initStar(numStar);
    for(int i=0;i<simulations;i++){
        StarsMovment(Stars,numStar,dt);
        calculateforces(Stars,numStar,dt);
        
        if(i == simulations/2 || i==simulations-1 || i==0){
            writeToFile(Stars,numStar,fp);
        }
    }
    free(Stars);
    fclose(fp);
}


int main(int argc,char** argv)
{
    double start_time = omp_get_wtime();
    simulator(atoi(argv[1]),atof(argv[2]));
    double end_time = omp_get_wtime();
    printf("Simulation time: %lf \n",end_time-start_time);
    return 0;
}

