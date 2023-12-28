
/**
 * Multiplica dos matrices cuadradas: C = A * B.
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

// Numero maximo de threads por cada dimensión del bloque
//
#define MAX_TH_PER_BLOCK_DIM 32

// Tamanho por defecto de las matrices
#define MATDIMDEF 1000

// Numero de threads por cada dimensión bloque por defecto
#define TPBDIMDEF 4

// Tipo de datos
typedef float basetype;

void check_memoria(const unsigned int numElemA, const unsigned int numElemB, const unsigned int numElemC);

/**
 * Codigo host
 */
__host__ void
h_matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int NA, unsigned int YB, unsigned int YA)
{
  for (unsigned int i = 0; i < NA; ++i)
    for (unsigned int j = 0; j < YB; ++j) {
      basetype sum = (basetype) 0.0;
      for (unsigned int k = 0; k < YA; ++k)
        sum += A[i*YA + k]*B[k*YB + j];
      C[i*YB + j] = sum;
  }
}

/**
 * Codigo CUDA
 * Cada thread computa un elemento de C
 */
__global__ void
matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int NA, unsigned int YB, unsigned int YA)
{
  // TODO: Calcula el indice de la fila de C y A
  int i = (blockDim.y * blockIdx.y + threadIdx.y);
  // TODO Calcula el indice de la columna de C y B
  int j = (blockDim.x * blockIdx.x + threadIdx.x);

  if ((i < YB) && (j < NA))
  {
    basetype sum = (basetype) 0.0;
    for(unsigned int k = 0; k < YA; ++k)
    {
      sum += A[i*YA + k]*B[k*YB + j];
    }
    C[i*YB + j] = sum;
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
  unsigned int numElemA = 1, numElemB = 1, numElemC = 1;
  unsigned int NA = 1, YA = 1, NB = 1, YB = 1;
  unsigned int tpbdimX=1, tpbdimY=1;
  size_t sizeA = 0, sizeB = 0, sizeC = 0;
  // Valores para la medida de tiempos
  struct timespec tstart, tend;
  double tint;

  //####################AÑADIDO####################
  /*Ahora se recibirán 4 argumentos que indican el tamaño de las matrices NA, YA, NB e YB. El problema
  es que ahora la matriz resultante tendrá dimensiones diferentes que A y B. */
  NA = (argc > 1) ? atoi(argv[1]):MATDIMDEF;
  YA = (argc > 2) ? atoi(argv[2]):MATDIMDEF;
  NB = (argc > 3) ? atoi(argv[3]):MATDIMDEF;
  YB = (argc > 4) ? atoi(argv[4]):MATDIMDEF;

  if (YA != NB){
    printf("Error: las matrices no se pueden multiplicar\n");
    exit(EXIT_FAILURE);
  }

  numElemA = NA*YA;
  numElemB = NB*YB;
  numElemC = NA*YB;

  sizeA = numElemA * sizeof(basetype);
  sizeB = numElemB * sizeof(basetype);
  sizeC = numElemC * sizeof(basetype);

  // Tamanho de los vectores
  //matrizDim = (argc > 1) ? atoi(argv[1]):MATDIMDEF;
  // Número de elementos de las matrices
  //numElem = matrizDim*matrizDim;
  // Tamanho de las matrices en bytes
  //size = numElem * sizeof(basetype);

  // Numero de threads por cada dimension  del bloque
  tpbdimX = (argc > 2) ? atoi(argv[5]):TPBDIMDEF;
  tpbdimY = (argc > 3) ? atoi(argv[6]):TPBDIMDEF;
  // Comprueba si es superior al máximo
  tpbdimX = (tpbdimX > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM:tpbdimX;
  tpbdimY = (tpbdimY > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM:tpbdimY;


  check_memoria( numElemA, numElemB, numElemC );

  // Caracteristicas del Grid
  // Hilos por bloque: primer parámetro dim_x, segundo dim_y
  dim3 threadsPerBlock( tpbdimX, tpbdimY, 1 );
  // TODO: Calcula el número de bloques en el Grid (bidimensional)
  dim3 blocksPerGrid( (YB + threadsPerBlock.x -1 ) / threadsPerBlock.x, (NA + threadsPerBlock.y -1 ) / threadsPerBlock.y, 1 );

  printf("Multiplicación de matrices de dimension (%u,%u) * (%u,%u) = (%u,%u), con (%u,%u) bloques de (%u,%u) threads\n",
    NA, YA, NB, YB, NA, YB, blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

  h_A = (basetype *) malloc(sizeA);
  h_B = (basetype *) malloc(sizeB);
  h_C = (basetype *) malloc(sizeC);
  h_C2 = (basetype *) malloc(sizeC);

  // Comprueba errores
  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
    fprintf(stderr, "Error reservando memoria en el host\n");
    exit(EXIT_FAILURE);
  }

  /*// Inicializa las matrices en el host
  for (int i = 0; i < numElem; ++i)
  {
    h_A[i] = rand()/(basetype)RAND_MAX;
    h_B[i] = rand()/(basetype)RAND_MAX;
  }*/

  // Inicializamos las matrices por separado
  for (int i = 0; i < numElemA; i++){
    h_A[i] = rand()/(basetype)RAND_MAX;
  }

  for (int i = 0; i < numElemB; i++){
    h_B[i] = rand()/(basetype)RAND_MAX;
  }

  // Inicio tiempo
  TSET(tstart);
  //clock_gettime( CLOCK_MONOTONIC, &tstart );
  // Multiplica las matrices en el host
  //h_matrizMul( h_A, h_B, h_C, NA, YB, YA);
  // Fin tiempo
  //TSET( tend );
  //tint = TINT(tstart, tend);
  //printf( "HOST: Tiempo multiplicacion: %lf ms\n", tint );

  // Inicio tiempo multiplicacion GPU
  TSET( tstart );

  // Reserva memoria para las matrices en el dispositivo
  checkError( cudaMalloc((void **) &d_A, sizeA) );
  checkError( cudaMalloc((void **) &d_B, sizeB) );
  checkError( cudaMalloc((void **) &d_C, sizeC) );

  // Copia las matrices h_A y h_B del host al dispositivo
  checkError( cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice) );
  checkError( cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice) );

  // TODO: Lanza el kernel CUDA
  matrizMul<<<blocksPerGrid, threadsPerBlock>>>( d_A, d_B, d_C, NA, YB, YA );

  // Comprueba si hubo un error al el lanzamiento del kernel
  // Notar que el lanzamiento del kernel es asíncrono por lo que
  // este chequeo podría no detectar errores en la ejecución del mismo
  checkError( cudaPeekAtLastError() );
  // Sincroniza los hilos del kernel y chequea errores
  // Este chequeo detecta posibles errores en la ejecución
  // Notar que esta sincrinización puede degradar el rendimiento
  checkError( cudaDeviceSynchronize() );

  // Copia el vector resultado del dispositivo al host
  checkError( cudaMemcpy(h_C2, d_C, sizeC, cudaMemcpyDeviceToHost) );

  // Fin tiempo multiplicacion GPU
  TSET( tend );
  // Calcula tiempo para la multiplicacion GPU
  tint = TINT(tstart, tend);
  printf( "DEVICE: Tiempo multiplicacion: %lf ms\n", tint );


  // Verifica que la multiplicacion es correcta
  for (unsigned int i = 0; i < numElemC; ++i)
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
check_memoria(const unsigned int numElemA, const unsigned int numElemB, const unsigned int numElemC)
{
  cudaDeviceProp prop;
  checkError( cudaGetDeviceProperties(&prop, 0) );

  size_t gmem = prop.totalGlobalMem;
  //########################AÑADIDO########################
  //Ahora cada matriz tiene un numero de elementos diferente
  size_t bytes_arrays = numElemA*sizeof(basetype) + numElemB*sizeof(basetype) + numElemC*sizeof(basetype);
  //#########################################################
  double gib = (double)(1073741824.0);

  printf( "GiB ocupados en la GPU: %g GiB, memoria global %g GiB\n", bytes_arrays/gib, gmem/gib );
  if( gmem >= bytes_arrays )
    printf( "GiB libres en la GPU: %g GiB\n", (gmem-bytes_arrays)/gib );
  else
    printf( "Los arrays no caben en la memoria de la GPU\n" );
}
