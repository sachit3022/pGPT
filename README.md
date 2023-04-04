# pGPT

Scaling goldilock constraints of transformers.
Some of the initial observations and tests.
Understanding the Nural networks and the way they are constructed.
//

Explanation of graph structure 
How the existing code can be generalised to other frame works. like transformers.
What are the parallel parts in the code and how I have used openMP and MPI to achieve the performace.

//


All the operations are floating point operations.

95.069 seconds vs 10.842 seconds while training 8000 samples vs 8, 1000 samples on 8 different servers.
But surprisingly I didnot gain any performace on the parallel pragma omp ( if we make this work ) then we are gaurenteed to have a maximum performace.
So In genral proving that the nural network training is a function of layers and not the data, as we can parallelise the data, as we have not reached the limits on the data part.
But as the embedings get larger and larger typically in the BERT / transformer models.

And to my knowledge pyTorch also has the same implementations.
