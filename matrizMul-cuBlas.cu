
/**
 * Multiplica dos matrices cuadradas: C = A * B.
 */
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"



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

// Numero maximo de threads por cada dimensión del bloque
// Consideramos threadsPerBlock.x == threadsPerBlock.y
//
#define MAX_TH_PER_BLOCK_DIM 32

// Tamanho por defecto de las matrices
#define MATDIMDEF 1000

// Numero de threads por cada dimensión bloque por defecto
#define TPBDIMDEF 4

// Tipo de datos
typedef float basetype;

void check_memoria(const unsigned int matrizDim);

/**
 * Codigo host
 */
__host__ void
h_matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int matrizDim)
{
  for (unsigned int i = 0; i < matrizDim; ++i)
    for (unsigned int j = 0; j < matrizDim; ++j) {
      basetype sum = (basetype) 0.0;
      for (unsigned int k = 0; k < matrizDim; ++k)
        sum += A[i*matrizDim + k]*B[k*matrizDim + j];
      C[i*matrizDim + j] = sum;
  }
}

/**
 * Codigo CUDA
 * Cada thread computa un elemento de C
 */
__global__ void
matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int matrizDim)
{
  // TODO: Calcula el indice de la fila de C y A
  int i = (blockDim.y * blockIdx.y + threadIdx.y);
  // TODO Calcula el indice de la columna de C y B
  int j = (blockDim.x * blockIdx.x + threadIdx.x);

  if ((i < matrizDim) && (j < matrizDim))
  {
    basetype sum = (basetype) 0.0;
    for(unsigned int k = 0; k < matrizDim; ++k)
    {
      sum += A[i*matrizDim + k]*B[k*matrizDim + j];
    }
    C[i*matrizDim + j] = sum;
  }
}

/**
 * Funcion main en el host
 * Parametros: nElementos threadsPerBlock
 */
int
main(int argc, char *argv[])
{
  basetype *h_A=NULL, *h_B=NULL, *h_C=NULL, *h_C2=NULL;
  basetype *d_A=NULL, *d_B=NULL, *d_C=NULL;
  unsigned int matrizDim = 1, tpbdim = 1, numElem = 1;
  size_t size = 0;
  // Valores para la medida de tiempos
  struct timespec tstart, tend;
  double tint;

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  // Tamanho de los vectores
  matrizDim = (argc > 1) ? atoi(argv[1]):MATDIMDEF;
  // Número de elementos de las matrices
  numElem = matrizDim*matrizDim;
  // Tamanho de las matrices en bytes
  size = numElem * sizeof(basetype);

  // Numero de threads por cada dimension  del bloque
  tpbdim = (argc > 2) ? atoi(argv[2]):TPBDIMDEF;
  // Comprueba si es superior al máximo
  tpbdim = (tpbdim > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM:tpbdim;

  check_memoria( numElem );

  const int m = matrizDim;
  const int n = matrizDim;
  const int k = matrizDim;

  const int lda = matrizDim;
  const int ldb = matrizDim;
  const int ldc = matrizDim;

  const float alpha = 1.0;
  const float beta = 0.0;

  // Caracteristicas del Grid
  // Hilos por bloque: primer parámetro dim_x, segundo dim_y
  dim3 threadsPerBlock( tpbdim, tpbdim, 1 );
  // TODO: Calcula el número de bloques en el Grid (bidimensional)
  dim3 blocksPerGrid( (matrizDim + threadsPerBlock.x -1 ) / threadsPerBlock.x, (matrizDim + threadsPerBlock.y -1 ) / threadsPerBlock.y, 1 );

  printf("Multiplicación de matrices de dimension (%u,%u), con (%u,%u) bloques de (%u,%u) threads\n",
    matrizDim, matrizDim, blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

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

  // Inicializa las matrices en el host
  for (int i = 0; i < numElem; ++i)
  {
    h_A[i] = rand()/(basetype)RAND_MAX;
    h_B[i] = rand()/(basetype)RAND_MAX;
  }

  printf("Matriz A\n");
  print_matrix(m, k, h_A, lda);
  printf("Matriz B\n");
  print_matrix(k, n, h_B, ldb);

  printf("IMpresion casera: \n");
  printf("1: %f, 2: %f, 3: %f, 4: %f\n", h_A[0], h_A[1], h_A[2], h_A[3]);


  // Inicio tiempo
  TSET(tstart);
  //clock_gettime( CLOCK_MONOTONIC, &tstart );
  // Multiplica las matrices en el host
  h_matrizMul( h_A, h_B, h_C, matrizDim );

  
  printf("Resultado normal\n");
  print_matrix(m, n, h_C, ldc);
  // Fin tiempo
  TSET( tend );
  tint = TINT(tstart, tend);
  printf( "HOST: Tiempo multiplicacion: %lf ms\n", tint );

  // Inicio tiempo multiplicacion GPU
  TSET( tstart );

  

  //STEP 1: Create cublas handle
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  //STEP 2: Copy data to device
  // Reserva memoria para las matrices en el dispositivo
  checkError( cudaMalloc((void **) &d_A, size) );
  checkError( cudaMalloc((void **) &d_B, size) );
  checkError( cudaMalloc((void **) &d_C, size) );

  
  // Copia las matrices h_A y h_B del host al dispositivo
  checkError( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
  checkError( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

  //CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  //CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  //STEP 3: Compute
  // TODO: Lanza el kernel CUDA
  //matrizMul<<<blocksPerGrid, threadsPerBlock>>>( d_A, d_B, d_C, matrizDim );

  CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

  // Comprueba si hubo un error al el lanzamiento del kernel
  // Notar que el lanzamiento del kernel es asíncrono por lo que
  // este chequeo podría no detectar errores en la ejecución del mismo
  checkError( cudaPeekAtLastError() );
  // Sincroniza los hilos del kernel y chequea errores
  // Este chequeo detecta posibles errores en la ejecución
  // Notar que esta sincrinización puede degradar el rendimiento
  checkError( cudaDeviceSynchronize() );

  // Copia el vector resultado del dispositivo al host
  checkError( cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost) );

  //checkError( cudaMemcpyAsync(h_C2, d_C, size, cudaMemcpyDeviceToHost, stream) );
  //CUDA_CHECK(cudaStreamSynchronize(stream));

  
  printf("\nResultado cublas\n");
  print_matrix(m, n, h_C2, ldc);
  // Fin tiempo multiplicacion GPU
  TSET( tend );
  // Calcula tiempo para la multiplicacion GPU
  tint = TINT(tstart, tend);
  printf( "DEVICE: Tiempo multiplicacion: %lf ms\n", tint );


  // Verifica que la multiplicacion es correcta
  for (unsigned int i = 0; i < numElem; ++i)
  {
    if (fabs(h_C2[i] - h_C[i]) > 1e-3)
    {
      fprintf(stderr, "Verificacion de resultados falla en el elemento %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Multiplicacion correcta.\n");

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

void
check_memoria(const unsigned int numElem)
{
  cudaDeviceProp prop;
  checkError( cudaGetDeviceProperties(&prop, 0) );

  size_t gmem = prop.totalGlobalMem;
  size_t bytes_arrays = 3*numElem*sizeof(basetype);
  double gib = (double)(1073741824.0);

  printf( "GiB ocupados en la GPU: %g GiB, memoria global %g GiB\n", bytes_arrays/gib, gmem/gib );
  if( gmem >= bytes_arrays )
    printf( "GiB libres en la GPU: %g GiB\n", (gmem-bytes_arrays)/gib );
  else
    printf( "Los arrays no caben en la memoria de la GPU\n" );
}
