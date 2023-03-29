#include <iostream>
#include<fstream>
#include <stdio.h>
using namespace std;
#include <sys/time.h>
#include <random>
#include <stack>
#include <set>

#include <functional>

std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0,1.0);

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


    Value operator+( Value&  other){
        float d = this->data + other.data;
        Value v =  Value(d);
        v.type = 1; 
        v.prev0 = &other;
        v.prev1 = this;  
        return v;
    }
    Value operator-( Value&  other){
        float d = this->data - other.data;
        Value v =  Value(d);
        v.type = 3; 
        v.prev0 = &other;
        v.prev1 = this;  
        return v;
    }
    Value operator*(Value&  other){
        float d = this->data * other.data;
        Value v =  Value(d);
        v.type = 2;
        v.prev0 = &other;
        v.prev1 = this;

        return v;
    }
    operator std::string () const {
        return std::to_string(data) ;
    }
    void _backward(){
        
        if (type == 1){
            prev0->grad = grad;
            prev1->grad += grad;
        }
        if (type == 2){
            prev0->grad += prev1->data * grad;
            prev1->grad += prev0->data * grad;
        }
        if (type == 3){
            prev0->grad += prev1->data * grad;
            prev1->grad -= prev0->data * grad;
        }
    };
    void build_topo(Value* v, stack<Value* >& topo, set<Value*>&  visited ){
        if(visited.find(v) == visited.end()){
            visited.insert(v);
            if(v->type !=0) {

                v->build_topo(v->prev0,topo,visited);
                v->build_topo(v->prev1,topo,visited);
                
            }     
            topo.push(v);
        };
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


};








std::ostream& operator<< (std::ostream &out, Value const& data) {
    out << data.data;
    return out;
}


Value* init_array(int size){
    Value* h_A = (Value *)malloc(size * sizeof(Value));
    for(int j=0; j<size; j++){
        h_A[j] =  Value(distribution(generator));
    }
    return h_A;
}


void print_matrix(Value* h_A, int row,int col){
    for(int i=0; i<row;i++){
        for(int j=0; j<col; j++){
            cout << h_A[i*col+j] <<  " ";
        }
        cout << endl;
    } 
}

Value* matrix_multiplication(Value* h_A,Value* h_B,int row,int hid,int col){
    Value* h_c = (Value *)malloc( row*col * sizeof(Value));

    for(int i=0; i<row; i++){
        for(int j=0;j<col; j++){
            for(int k=0;k<hid; k++){
                h_c[i*col+j] = h_A[i*hid+k]*h_B[k*col+j];
            }
        }
    } 
    return h_c;
}

Value* matrix_add(Value* h_A,Value* h_B,int row, int col){
    Value* h_c = (Value *)malloc( row*col * sizeof(Value));
    for(int i=0; i<row; i++){
        for(int j=0;j<col; j++){
                h_c[i*col+j] = h_A[i*col+j]+ h_B[i*col+j];
            }
        }
    return h_c;
}




class LinearLayer{
    
    int in_channels, out_channels, batch;
    Value *A, *B;
    public:
        void set_channels(int,int,int);
        void init();
        void print_weights();
        Value* forward(Value*);
        Value* backward(Value* );
        LinearLayer(int b, int in,int out ){
            in_channels = in;
            out_channels = out;
            batch = b;
        }
};



void LinearLayer::init(){
    A = init_array(out_channels*in_channels);
    B = init_array(out_channels);
}

void LinearLayer::print_weights(){
    cout << "matrix A" << endl;
    print_matrix(A,in_channels,out_channels);
    cout << "matrix: B" <<  endl;
    print_matrix(B,1,out_channels);
}

Value* LinearLayer::forward(Value* X){
   return matrix_add(matrix_multiplication(X,A,batch,in_channels,out_channels),B,batch,out_channels);
}

Value mseloss(Value* u, Value* v,int batch){
    Value out = Value(0);
    for(int i =0; i<batch; i++){
        out = out + u[i] - v[i];
    }
    return out;
}




int main(int argc, char const *argv[])
{

    struct{
        int batch = 2;
        float lr = 0.001;
        int inp_dim = 2;
        int out_dim =1;
    } model_params ;


    int batch = model_params.batch;
    int out_channels = model_params.out_dim;
    int in_channels = model_params.inp_dim;
    int epocs =1;


    // init data to perform experiments
    Value* X = init_array(model_params.batch*model_params.inp_dim);
    Value *A = init_array(in_channels*out_channels);
    Value* B = init_array(out_channels);
    Value* y = matrix_add(matrix_multiplication(X,A,batch,in_channels,out_channels),B,batch,out_channels);
    cout << "Matrix y" << endl;
    print_matrix(y,model_params.batch,model_params.out_dim);

    //init neural network.
    LinearLayer l1(model_params.batch,model_params.inp_dim,model_params.out_dim);
    mse_loss = MSELoss();
    l1.init();
    //l1.print_weights();
    
    
    // loop between the forward and backward layers.
    
    for(int e=0; e<epocs; e++){
        Value* C = l1.forward(X);
        print_matrix(C,model_params.batch,model_params.out_dim);
        Value& l = mse_loss(C,y);
        l.backward();
        printf(f"Loss for epoc %i is %f",e,l.item());
    }
    Value a = Value(10);
    Value b = Value(20);
    Value c = a*b;
    c.backward();
    cout << a.grad;
    
    return 0;
}
