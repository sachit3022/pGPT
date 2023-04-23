#include <iostream>
#include <cstdlib>
#include <fstream>
#include "mpi.h"
#include <cstdio>
#include <vector>
#include <ctime>
#include <thread>
#include <sys/time.h>
#include <algorithm>
#include <random>

extern "C" {
#include "get_walltime.h"
}
extern "C" void get_walltime(double*);

 /**
    Folllow the commnds below to link this is the most accuract nano second time in C you dont have UTC wall time to nano sec.
    [gaudisac@dev-intel14 sachit]$ mpicc \-c -o time.o  get_walltime.c
    [gaudisac@dev-intel14 sachit]$ mpicxx -c -std=c++0x -o pingpong.o  pingpong.cpp
    [gaudisac@dev-intel14 sachit]$ mpicxx -o sachit time.o pingpong.o
    [gaudisac@dev-intel14 sachit]$ srun -n 2 ./sachit
  **/

using namespace std;

int main(int argc, char* argv[])
{

    MPI_Init(NULL,NULL);
    double begin, end;
    FILE *fp;

    

    int rank;
    int size;

    MPI_Status status;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    MPI_Request request;



    std::vector<int> foo = {16,14,13,5,15,18,12,4,20,7,8,10,19,9,2,6,11,17,1,3,0};

    // std::random_shuffle(foo.begin(), foo.end());
    
    //ptrdiff_t pos = find(foo.begin(), foo.end(), rank) - foo.begin();
    //rank = pos;

    //cout << foo[19];

    int r = rank %2 ;

    fp = fopen("sachit_exp4.txt","a");

    //fprintf(fp, " time :: server :: rank :: count");

    for(unsigned long j=1; j<=1; j*=2){

        int count = stoi(argv[1]);

        unsigned int array[count];
        if (r == 1){
            std::fill_n(array, count, rank);  
            
            get_walltime(&begin);
            srand(begin);

            string tmp  =   std::to_string(begin);
            string result=  tmp.substr(tmp.size() - 10) + RandomString(5) + std::to_string(rank) + std::getenv(env_var[0]);   
             
            
            
            get_walltime(&begin);
            //gettimeofday(&begin, 0);
            MPI_Isend(&array,count,MPI_INT,rank-1,0,comm,&request);
            MPI_Irecv(&array,count,MPI_INT, rank-1,0,comm,&request);
            MPI_Wait(&request,MPI_STATUS_IGNORE);
            //std::this_thread::sleep_for(std::chrono::milliseconds(10));
            get_walltime(&end);            
            double elapsed =  end - begin;



            //gettimeofday(&end, 0);
            //long seconds = end.tv_sec - begin.tv_sec;
            //long microseconds = end.tv_usec - begin.tv_usec;
            //double elapsed = seconds*1e6 + microseconds;            
            fprintf(fp, "sent :: %10.20lf :: %d :: %lu :: ", elapsed, rank,count);
            fprintf(fp," %10.20lf :: %10.20lf:: ",begin,end);
            
            MPI_Send(&begin,1,MPI_DOUBLE,rank-1,2,comm);
            MPI_Send(&end,1,MPI_DOUBLE,rank-1,3,comm);
            MPI_Send(result.c_str(), result.size(), MPI_CHAR, rank-1, 4, comm);


            //fprintf(fp," %20.20lf :: %20.20lf :: ",begin.tv_sec + begin.tv_usec*1e-6 ,end.tv_sec + end.tv_usec*1e-6d);
            fprintf(fp,std::getenv(env_var[0]));
            fprintf(fp," :: %s",result.c_str());
            fprintf(fp,"\n");
            std::cout << " Hello from processor " << rank << " sent " << count << " example " << array[0] <<endl;
        }else{

            //gettimeofday(&begin, 0);
            get_walltime(&begin);

            MPI_Irecv(&array,count,MPI_INT, rank+1,0,comm,&request);
            MPI_Wait(&request,MPI_STATUS_IGNORE);
            MPI_Isend(&array,count,MPI_INT,rank+1,0,comm,&request);

            //gettimeofday(&end, 0);
            //long seconds = end.tv_sec - begin.tv_sec;
            //long microseconds = end.tv_usec - begin.tv_usec;
            //double elapsed = seconds*1e6 + microseconds; 
            get_walltime(&end);            
            double elapsed =  end - begin;
            


            fprintf(fp, "recieve :: %10.20lf :: %i :: %lu :: ", elapsed,rank,count);
            fprintf(fp," %10.20lf :: %10.20lf :: ",begin,end);
            MPI_Recv(&begin,1,MPI_DOUBLE,rank+1,2,comm,&status);
            MPI_Recv(&end,1,MPI_DOUBLE,rank+1,3,comm,&status);
            fprintf(fp," %10.20lf :: %10.20lf :: ",begin,end);
            fprintf(fp,std::getenv(env_var[0]));

            char result[20];

            MPI_Recv(&result,25,MPI_CHAR,rank+1,4,comm,&status);
            int rec_count;
            MPI_Get_count(&status, MPI_CHAR, &rec_count);
            
            string temp_s = "";
            for(int k=0; k<rec_count;k++){
                temp_s += result[k];

            }

            fprintf(fp," :: %s",temp_s.c_str());
            fprintf(fp,"\n");

            std::cout << " Hello from processor " << rank << " " << " recieved "<< rec_count <<  " example " << array[0] << endl;
        }
        }
    
    fclose(fp);
    MPI_Finalize();
}