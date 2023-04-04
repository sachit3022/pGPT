#include <iostream>
#include<fstream>
#include <stdio.h>
using namespace std;
#include <sys/time.h>
#include <random>
#include <stack>
#include <set>
#include <vector>
#include <functional>
#include <chrono>
#include <mpi.h>
#include <omp.h>

/*
To run please use the above conditions.
mpicxx -std=c++17 -o micrograd micrograd.cpp -fopenmp


*/



default_random_engine generator;
normal_distribution<float> distribution(0,1.0);


class Value{
    
    public:
        float data;
        float grad = 0 ;
        Value(float d){
            data =d;
        }
        Value* prev0;
        Value* prev1;
        int type = 0;


    operator string () const {
        return to_string(data) +  " " + to_string(grad);
    }
    void _backward(){
        
        if (type == 1){
            prev0->grad += grad;
            prev1->grad += grad;
        }
        if (type == 2){
            prev0->grad += prev1->data * grad;
            prev1->grad += prev0->data * grad;
        }
        if (type == 3){
            prev0->grad += grad;
            prev1->grad -= grad;
        }
        if (type == 4){
            prev0->grad += (prev1->data * pow(prev0->data, (prev1->data-1))) * grad;
        }
    };

    void build_topo(Value* v, stack<Value* >& topo, set<Value*>&  visited ){
        // can be parallelise. know graph type and communicate. openGL.

        if(visited.find(v) == visited.end()){
            visited.insert(v);
            if(v->type !=0) {

                v->build_topo(v->prev0,topo,visited);
                v->build_topo(v->prev1,topo,visited);
                
            }     
            topo.push(v);
        };
    }      
    void _zero_grad(){
        if (type !=0){
            prev0->grad =0;
            prev1->grad =0;

        }

        grad =0;
    }

    void backward(){
        
        //topological order all of the children in the graph
        
        this -> grad = 1;
        

        stack<Value* > topo;
        set<Value *> visited;
        
        build_topo(this,topo, visited);
        // go one variable at a time and apply the chain rule to get its gradient
        
        while (!topo.empty()) {
            Value* temp = topo.top();
            topo.pop();
            temp->_backward();
        }

    }
    void zero_grad(){
        
        //topological order all of the children in the graph
        
        this -> grad = 0;
        

        stack<Value* > topo;
        set<Value *> visited;
        
        build_topo(this,topo, visited);
        
        while (!topo.empty()) {
            Value* temp = topo.top();
            topo.pop();
            temp->_zero_grad();
        }

    }

};

Value* operator+( Value &u, Value& v){
    float d = u.data + v.data;
    Value* z =  new Value(d);
    z->type = 1; 
    z->prev0 = &u;
    z->prev1 = &v;  
    return z;
}
Value* operator-( Value &u, Value& v){
    float d = u.data - v.data;
    Value* z =  new Value(d);
    z->type = 3; 
    z->prev0 = &u;
    z->prev1 = &v;  
    return z;
}

Value* operator*(Value &u, Value& v){
    float d = u.data* v.data;
    Value* z =  new Value(d);
    z->type = 2;
    z->prev0 = &u;
    z->prev1 = &v;

    return z;
}
Value*  operator^ (Value &u, Value& v){
    float d = pow(u.data,v.data);
    Value* z =  new Value(d);
    z->type = 4;
    z->prev0 = &u;
    z->prev1 = &v;
    return z;
}

std::ostream& operator<< (std::ostream &out, Value const& data) {
    out << data.data << "  "<< data.grad;
    return out;
}

void init_array(float* h_A, int size){
    
    for(int j=0; j<size; j++){
        h_A[j] = distribution(generator);
    }
}
void init_array(vector <Value*> & h_A, int size){
    
    for(int j=0; j<size; j++){
        h_A.push_back(new Value(distribution(generator))); //distribution(generator)
    }
}

void convert_to_value(vector <Value*> & h_A, float* f_A,int size){
    
    for(int j=0; j<size; j++){
        h_A.push_back(new Value(f_A[j])); //distribution(generator)
    }
}

void init_zeros(vector <Value*> & h_A, int size){
    
    for(int j=0; j<size; j++){
        h_A.push_back(new Value(0.000)); //distribution(generator)
    }
}
void init_zeros(float*h_A, int size){
    
    for(int j=0; j<size; j++){
        h_A[j] = 0; //distribution(generator)
    }
}

void print_matrix(vector <Value*> & h_A, int row,int col){
    for(int i=0; i<row;i++){
        for(int j=0; j<col; j++){
            cout << *h_A[i*col+j] <<  " ";
        }
        cout << endl;
    } 
    cout << endl;
}
void print_matrix(float*  h_A, int row,int col){
    for(int i=0; i<row;i++){
        for(int j=0; j<col; j++){
            cout << h_A[i*col+j] <<  " ";
        }
        cout << endl;
    } 
    cout << endl;
}


void matrix_multiplication(float*  h_A,float* h_B,float* h_c,int row,int hid,int col){
    
    omp_set_dynamic(0);
    #pragma omp parallel for collapse(2) num_threads(16)
    for(int i=0; i<row; i++){ 
        for(int j=0;j<col; j++){
            for(int k=0;k<hid; k++){
                h_c[i*col+j] += h_A[i*hid+k] * h_B[k*col+j];
            }
        }
    } 
    return;
}

void matrix_multiplication(vector <Value*> & h_A,vector <Value*> & h_B,vector <Value*> & h_c,int row,int hid,int col){
    
    //cout << "matrix" << row << hid << col <<endl; //omp num treads outside the threads.
    omp_set_dynamic(0);
    #pragma omp parallel for collapse(2) num_threads(16)
    for(int i=0; i<row; i++){ 
        for(int j=0;j<col; j++){
            for(int k=0;k<hid; k++){
                h_c[i*col+j] =  (*h_c[i*col+j]) + *((*h_A[i*hid+k]) *  (*h_B[k*col+j]));
            }
        }
    } 
    return;
}

void matrix_add(vector <Value*> & h_A,vector <Value*> & h_B,vector <Value*> & h_c, int row, int col){
    for(int i=0; i<row; i++){
        for(int j=0;j<col; j++){
                h_c[i*col+j] = (*h_A[i*col+j])+ (*h_B[i*col+j]);
            }
        }
}

class LinearLayer{
    

    public:

        int in_channels, out_channels, batch;
        float lr =0.001;
        vector <Value*> A;
        vector <Value*> B;


        void init();
        void print_weights();
        void forward(vector <Value*>&, vector <Value*>&);
        void step();
        LinearLayer(int b, int in,int out ){
            in_channels = in;
            out_channels = out;
            batch = b;
        }

};


void step_array(vector <Value*> &U, int size,float lr,int world_rank){


    int  rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
    for( int i; i<size;i++){

        float sub_avg = U[i]->grad;
        float *sub_avgs = NULL;
        
        if(rank == 0 ){
            sub_avgs = (float *)malloc(sizeof(float) * world_rank);
        }

        //cout << U[i]->grad << "  "<< rank <<"   "<< i <<endl;   
        
        MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); //Igather and wait.
        
        if(rank == 0 ){
            /*
            for(int l=1; l<world_rank;l++){
                sub_avg+=sub_avgs[l];
                //cout << sub_avgs[l] << "  "<< l << "  ";
            }
            sub_avg = sub_avg / world_rank;
            */
            //cout << sub_avg << endl;
            U[i]->data = U[i]->data - lr*sub_avg;
        } 
        float *buffer = (float *)malloc(sizeof(float) * 1);
        buffer[0] = U[i]->data;
        MPI_Bcast(buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);       //autoformat.
        U[i]->data = buffer[0];
    }
}

void LinearLayer::init(){
    init_array(A,out_channels*in_channels);
    init_array(B,batch);
}
void LinearLayer::step(){
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    step_array(A,out_channels*in_channels,lr,world_size);
    //step_array(B,out_channels,lr,size);

}

void LinearLayer::print_weights(){
    cout << "matrix A" << endl;
    print_matrix(A,in_channels,out_channels);
    cout << "matrix: B" <<  endl;
    print_matrix(B,1,out_channels);
}

void LinearLayer::forward(vector <Value*> & X, vector <Value*> & out){
    matrix_multiplication(X,A,out,batch,in_channels,out_channels);
    //matrix_add(out,B,out,batch,out_channels);
}

Value* mseloss(vector <Value*> & out, vector <Value*> & y, int batch){
    Value *pu = new Value(float(2));
    Value *div = new Value(float(0.5));
    Value *root = new Value(float(0.5));

    Value *c = new Value(float(0.0000));
    
    for(int i=0;i<batch;i++){
        c = *c + *(*(*out[i] - *y[i])^(*pu)); 
        //cout << pu-> data << endl;
        //cout << "c"<<c->data << "  " << out[i]->data << "   "<<y[i]->data <<endl;
    }
    c= *c*(*div);
    c = *c^(*root);
    return  c;
}

float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}




int main(int argc, char const *argv[])
{
    MPI_Init(NULL, NULL);

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    


    struct{
        int batch = 800;
        float lr = 0.001;
        int inp_dim = 2;
        int out_dim =1;
    } model_params ;


    int batch = model_params.batch;
    int out_channels = model_params.out_dim;
    int in_channels = model_params.inp_dim;
    int epocs =100;
    int num_elements_per_proc = 100;

   
    // init data to perform experiments

    float *X = (float *)malloc(sizeof(float) * model_params.batch*model_params.inp_dim);
    float *y = (float *)malloc(sizeof(float) * batch);


    if (rank == 0) {       

        float *AS = (float *)malloc(sizeof(float) * in_channels*out_channels);
        float *BD = (float *)malloc(sizeof(float) * batch);

        init_array(X,model_params.batch*model_params.inp_dim);
        init_array(AS,in_channels*out_channels);
        init_array(BD,batch);
        
        
        init_zeros(y,batch);
        matrix_multiplication(X,AS,y,batch,in_channels,out_channels);
        //matrix_add(y,BD,y,batch,out_channels);
        
        //cout << "Matrix y" << endl;
        // print_matrix(y,model_params.batch,model_params.out_dim);
        cout << "Matrix AS" << endl;
        print_matrix(AS,in_channels,model_params.out_dim);
        //cout << "Matrix BB" << endl;
        //print_matrix(BD,model_params.batch,model_params.out_dim);
        cout <<endl;
    }
    

    
    float *x_buffer = (float *)malloc(sizeof(float) * num_elements_per_proc*in_channels);
    float *y_buffer = (float *)malloc(sizeof(float) * num_elements_per_proc);
    
    MPI_Scatter(X, num_elements_per_proc*in_channels, MPI_FLOAT, x_buffer, num_elements_per_proc*in_channels, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, num_elements_per_proc, MPI_FLOAT, y_buffer, num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector <Value*> value_X;
    vector <Value*> value_y;

    // loop between the forward and backward layers.
    convert_to_value(value_X,x_buffer, num_elements_per_proc*in_channels);
    convert_to_value(value_y,y_buffer,num_elements_per_proc);

    batch = num_elements_per_proc ;
    model_params.batch = batch;
    

    

    //init neural network.
    LinearLayer l1(model_params.batch,model_params.inp_dim,model_params.out_dim);
    l1.init();

    auto start_time = std::chrono::high_resolution_clock::now();    
    
    for(int e=0; e<300; e++){

        vector <Value*> out;
        init_zeros(out,batch*out_channels);
        l1.forward(value_X,out);
        //print_matrix(C,model_params.batch,model_params.out_dim);
       
        Value* c = mseloss(out,value_y,batch);
        
        c->backward();
        l1.step();
        
        c->zero_grad();
        

        //l1.print_weights();
        //cout << "Loss for epoc" << e << " is " << c->data << endl;
    }

    if (rank == 0) {       
        auto current_time = std::chrono::high_resolution_clock::now();
        cout << "Program has been running for " << std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() << " seconds" << endl;
        l1.print_weights();
    }
   
    
    MPI_Finalize();
    /*
    vector <Value*> A,X,y,out;
    A.push_back(new Value(float(0)));
    A.push_back(new Value(float(0)));

    X.push_back(new Value(float(1)));
    X.push_back(new Value(float(1)));
    X.push_back(new Value(float(2)));
    X.push_back(new Value(float(2)));
    X.push_back(new Value(float(3)));
    X.push_back(new Value(float(3)));

    y.push_back(new Value(float(2)));
    y.push_back(new Value(float(4)));
    y.push_back(new Value(float(6)));

    for(int i =0; i<100;i++){
     


        //Value *c1 = *(*(*(*A[0] * *X[0]) +  *(*A[1] * *X[1])) - *y[0])^(*pow); 
        //Value *c2 = *(*(*(*A[0] * *X[2]) +  *(*A[1] * *X[3])) - *y[1])^(*pow);
        // Value* c3 = *c1 + *c2;
        vector <Value*> out;
        out.push_back(new Value(0));
        out.push_back(new Value(0));
        out.push_back(new Value(0));
        

        matrix_multiplication(X,A,out,3,2,1);

        Value* c = mseloss(out,y,3);
        c->backward();
        c->zero_grad();
        
        A[0]->data = A[0]->data - 0.01*A[0]->grad;
        A[1]->data = A[1]->data - 0.01*A[1]->grad;
        
        cout << "Loss value "<<c->data <<endl;
    }
    cout << A[0]->data << "   "<< A[1]->data <<endl;
    */


    return 0;

}
