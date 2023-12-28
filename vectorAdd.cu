/**
 * Suma dos vectores: C = A + B.
 */
#include <stdio.h>
#include <time.h>

#define checkError(ans) { asserError((ans), __FILE__, __LINE__); }
inline void asserError(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define TSET(time)  clock_gettime( CLOCK_MONOTONIC, &(time) )
#define TINT(ts,te) { ( (double) 1000.*( (te).tv_sec - (ts).tv_sec ) + ( (te).tv_nsec - (ts).tv_nsec )/(double) 1.e6 ) }

// Numero maximo de threads por bloque
#define MAX_TH_PER_BLOCK 1024

// Tamanho por defecto de los vectores
#define NELDEF 1000

// Numero de threads por bloque por defecto
#define TPBDEF 256

// Numwero de repeticiones
#define NREPDEF 1

// Tipo de datos
typedef float basetype;

/**
 * Codigo host
 */
__host__ void
h_vectorAdd(const basetype *A, const basetype *B, basetype *C, int numElements)
{
    for (int i = 0; i < numElements; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Codigo CUDA
 */
__global__ void
vectorAdd(const basetype *A, const basetype *B, basetype *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Funcion main en el host
 * Parametros: nElementos threadsPerBlock nreps
 */
int
main(int argc, char *argv[])
{
    basetype *h_A=NULL, *h_B=NULL, *h_C=NULL, *h_C2=NULL;
    basetype *d_A=NULL, *d_B=NULL, *d_C=NULL;
    unsigned int numElements = 0, tpb = 0, nreps=1;
    size_t size = 0;
    // Valores para la medida de tiempos
    struct timespec tstart, tend;
    double tint;

    // Tamanho de los vectores
    numElements = (argc > 1) ? atoi(argv[1]):NELDEF;
    // Tamanho de los vectores en bytes
    size = numElements * sizeof(basetype);

    // Numero de threads por bloque
    tpb = (argc > 2) ? atoi(argv[2]):TPBDEF;
	// Comprueba si es superior al máximo
	tpb = (tpb > MAX_TH_PER_BLOCK) ? TPBDEF:tpb;

    // Numero de repeticiones de la suma
    nreps = (argc > 3) ? atoi(argv[3]):NREPDEF;

    // Caracteristicas del Grid
   
    dim3 threadsPerBlock( tpb );
    // blocksPerGrid = ceil(numElements/threadsPerBlock)
    dim3 blocksPerGrid( (numElements + threadsPerBlock.x - 1) / threadsPerBlock.x );
    printf("Suma de vectores de %u elementos (%u reps), con %u bloques de %u threads\n",
      numElements, nreps, blocksPerGrid.x, threadsPerBlock.x);

    // Reserva memoria en el host
    h_A = (basetype *) malloc(size);
    h_B = (basetype *) malloc(size);
    h_C = (basetype *) malloc(size);
    h_C2 = (basetype *) malloc(size);

    // Comprueba errores
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Error reservando memoria en el host\n");
        exit(EXIT_FAILURE);
    }

    // Inicializa los vectores en el host
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(basetype)RAND_MAX;
        h_B[i] = rand()/(basetype)RAND_MAX;
    }

    /*
    * Hace la suma en el host
    */
    // Inicio tiempo
    TSET(tstart);
    // Suma los vectores en el host nreps veces
    for(unsigned int r = 0; r < nreps; ++r)
      h_vectorAdd( h_A, h_B, h_C, numElements );
    // Fin tiempo
    TSET( tend );
    tint = TINT(tstart, tend);
    printf( "HOST: Tiempo para hacer %u sumas de vectores de tamaño %u: %lf ms\n", nreps, numElements, tint );

    /*
    * Hace la suma en el dispositivo
    */
    // Inicio tiempo
    TSET( tstart );
    // Reserva memoria en la memoria global del dispositivo
    checkError( cudaMalloc((void **) &d_A, size) );
    checkError( cudaMalloc((void **) &d_B, size) );
    checkError( cudaMalloc((void **) &d_C, size) );

    // Copia los vectores h_A y h_B del host al dispositivo
    checkError( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    checkError( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

    // Lanza el kernel CUDA nreps veces
    for(unsigned int r = 0; r < nreps; ++r) {
      vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

      // Comprueba si hubo un error al el lanzamiento del kernel
      // Notar que el lanzamiento del kernel es asíncrono por lo que
      // este chequeo podría no detectar errores en la ejecución del mismo
      checkError( cudaPeekAtLastError() );
      // Sincroniza los hilos del kernel y chequea errores
      // Este chequeo detecta posibles errores en la ejecución
      // Notar que esta sincrinización puede degradar el rendimiento
      checkError( cudaDeviceSynchronize() );
    }

    // Copia el vector resultado del dispositivo al host
    checkError( cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost) );

    // Fin tiempo
    TSET( tend );
    // Calcula tiempo para la suma en el dispositivo
    tint = TINT(tstart, tend);
    printf( "DEVICE: Tiempo para hacer %u sumas de vectores de tamaño %u: %lf ms\n", nreps, numElements, tint );


    // Verifica que la suma es correcta
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_C2[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Verificacion de resultados falla en el elemento %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Suma correcta.\n");

    // Liberamos la memoria del dispositivo
    checkError( cudaFree(d_A) );
    checkError( cudaFree(d_B) );
    checkError( cudaFree(d_C) );

    // Liberamos la memoria del host
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Terminamos\n");
    return 0;
}

