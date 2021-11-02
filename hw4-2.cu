#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>

const int INF = ((1 << 30) - 1); 


int* host_matrix;
int** device_matrix;
__device__ __host__ int cood(int i, int j, int N) {
	return i*N+j;
}
__global__ void phase1(int* devMatrix, int B, int N, int r) {
	int bi, bj;
	bi = r;
	bj = r;
    
	extern __shared__ int device_sm[];

	int offset_i = B * bi;
	int offset_j = B * bj;
	int offset_r = B * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	device_sm[cood(i, j, B)] = devMatrix[cood(i+offset_i, j+offset_j, N)];
	device_sm[cood(i+B, j, B)] = devMatrix[cood(i+offset_i, j+offset_r, N)];
	device_sm[cood(i+2*B, j, B)] = devMatrix[cood(i+offset_r, j+offset_j, N)];
	__syncthreads();

	for (int k = 0; k < B; k++) {
		if (device_sm[cood(i, j, B)] > device_sm[cood(i+B, k, B)] + device_sm[cood(k+2*B, j, B)]) {
            device_sm[cood(i, j, B)] = device_sm[cood(i+B, k, B)] + device_sm[cood(k+2*B, j, B)];
            device_sm[cood(i+2*B, j, B)] = device_sm[cood(i, j, B)];
            device_sm[cood(i+B, j, B)] = device_sm[cood(i, j, B)];
		}	

	}
	devMatrix[cood(i+offset_i, j+offset_j, N)] = device_sm[cood(i, j, B)];
	__syncthreads();
}

__global__ void phase2(int* devMatrix, int B, int N, int r) {
	int bi, bj;
	
	if (blockIdx.x == 1) {
		//column
		bi = (r + blockIdx.y + 1) % (N/B);
		bj = r;
	} else {
		//row
		bi = r;
		bj = (r + blockIdx.y + 1) % (N/B);
            }

	extern __shared__ int device_sm[];
	
	int offset_i = B * bi;
	int offset_j = B * bj;
	int offset_r = B * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	
	device_sm[cood(i, j, B)] = devMatrix[cood(i+offset_i, j+offset_j, N)];
	device_sm[cood(i+B, j, B)] = devMatrix[cood(i+offset_i, j+offset_r, N)];
	device_sm[cood(i+2*B, j, B)] = devMatrix[cood(i+offset_r, j+offset_j, N)];
	__syncthreads();

	// device_sm[i][j] = min{ device_sm[i][j], device_sm[i+bs][k] + device_sm[k+2bs][j] }
	for (int k = 0; k < B; k++) {
		if (device_sm[cood(i, j, B)] > device_sm[cood(i+B, k, B)] + device_sm[cood(k+2*B, j, B)]) {
            device_sm[cood(i, j, B)] = device_sm[cood(i+B, k, B)] + device_sm[cood(k+2*B, j, B)];
            if (r == bi) device_sm[cood(i+2*B, j, B)] = device_sm[cood(i, j, B)];
            if (r == bj) device_sm[cood(i+B, j, B)] = device_sm[cood(i, j, B)];
		}	
	}
	// device_sm[i][j] = devMatrix[i+bsbi][j+bsbj]
	devMatrix[cood(i+offset_i, j+offset_j, N)] = device_sm[cood(i, j, B)];
	__syncthreads();
}
__global__ void phase3(int* devMatrix, int B, int N, int r, int offset) {
	int bi, bj;
    
	bi = blockIdx.x + offset;
	bj = blockIdx.y;
     
	extern __shared__ int device_sm[];
	
	int offset_i = B * bi;
	int offset_j = B * bj;
	int offset_r = B * r;

	int i = threadIdx.y;
	int j = threadIdx.x;


	device_sm[cood(i, j, B)] = devMatrix[cood(i+offset_i, j+offset_j, N)];
	device_sm[cood(i+B, j, B)] = devMatrix[cood(i+offset_i, j+offset_r, N)];
	device_sm[cood(i+2*B, j, B)] = devMatrix[cood(i+offset_r, j+offset_j, N)];
	__syncthreads();
	
	for (int k = 0; k < B; k++) {
        device_sm[cood(i, j, B)] = min(device_sm[cood(i+B, k, B)] + device_sm[cood(k+2*B, j, B)], device_sm[cood(i, j, B)]);
	}
	
    devMatrix[(i+offset_i)*N + j+offset_j] = device_sm[i*B + j];
	__syncthreads();
}

int main(int argc, char* argv[]) {   
    int i, j;
	int B = 32;

	//有兩張GPU num_devices = 2
	int num_devices;
	cudaGetDeviceCount(&num_devices);

	//Input: 

	//io time measurement
    int *result, len;
    struct stat buf;
    int fd = open(argv[1], O_RDONLY);
    fstat(fd, &buf);
    len = (int)buf.st_size;
	result = (int*)mmap(0, len, PROT_READ, MAP_FILE|MAP_PRIVATE, fd, 0);
	
	
    int n = result[0];
	int edge = result[1];
	
	// printf("n: %d", n);
	// printf("edge num: %d ", edge);
    //blocking factor大於n 就把 blocking factor設成n
    if (B > n) B = n;
    
    //做padding: 把原本開vertex_num * n 開成 可以被blocking factor整除的 V_padding * V_padding
	int V_padding = n + (B - ((n-1) % B + 1));

	//allocate memory
    //cudaMallocHost((void**) &host_matrix, sizeof(int) * V_padding*V_padding);
    host_matrix = (int*)calloc(V_padding*V_padding, sizeof(int));


	//device_matrix設一個二為矩陣: 
	/*
		device_matrix[0][V_padding * VERTEXT_EXT]
		device_matrix[1][V_padding * VERTEXT_EXT]
	*/
	device_matrix = (int**) malloc(sizeof(int*) * num_devices);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void**) &device_matrix[omp_get_thread_num()], sizeof(int) * V_padding*V_padding);
	}

	//初始化 host_matrix 除了對角線是INF以外其他都是0
	#pragma omp parallel num_threads(num_devices)
	{
		for(i = 0; i < V_padding; ++i){
			for(j = 0; j < V_padding; ++j){
				if(i != j) host_matrix[i * V_padding + j] = INF;
			}
		}
	}
	
    //製造adjacency matrix
    for(i=2; i<3*edge+2; i+=3){
        host_matrix[result[i]*V_padding + result[i+1]] = result[i+2];
    }


	//開始blockedFW 


	//設block dimension 
	dim3 BLOCK_DIM(B, B);
	
	int round = (V_padding + B - 1) / B;
	
	//設grid dimension 
	//phase 1
	dim3 grid_phase1(round);
	// phase 2
	dim3 grid_phase2(2, round-1);
	// phase 3
	#pragma omp parallel num_threads(num_devices)
	{
		int gpu_id = omp_get_thread_num();
		cudaSetDevice(gpu_id);


		//num_blocks_per_gpu = 執行的round數 / gpu的樟樹

		//num_blocks_per_gpu: c2 的那個2 等於一張gpu裡面又有兩個process
		int num_blocks_per_gpu = round / num_devices;
		int row_start = num_blocks_per_gpu * gpu_id * B;
		//allocate the remain num
		if (gpu_id == num_devices-1)
			num_blocks_per_gpu += round % num_devices;

		dim3 grid_phase3(num_blocks_per_gpu, round);
		
		
			
		int index_copy_start = row_start * V_padding;
        //把上半部copy到gpu0 把下半部copy到gpu1
        
        //算memory copy host to device
        int sm_size = sizeof(int)*3*B*B;
        cudaMemcpy((void*) &(device_matrix[gpu_id][index_copy_start]), (void*) &(host_matrix[index_copy_start]), sizeof(int) * V_padding*B*num_blocks_per_gpu, cudaMemcpyHostToDevice);
        
       
        // float total_communication_time = 0;
		// float et = 0;
		
		for (int r = 0; r < round; r++) {
            
			int r_idx = cood(r * B, 0, V_padding);
            //exchange the pivot site to another
            
            //算communication time
            // cudaEvent_t communication_start, communication_end;
               
			// 	cudaEventCreate(&communication_start); 
			// 	cudaEventCreate(&communication_end); 
			// 	cudaEventRecord(communication_start); 
            if (r >= row_start/B && r < (row_start/B + num_blocks_per_gpu)) {
				cudaMemcpy((void*) &(host_matrix[r_idx]), (void*) &(device_matrix[gpu_id][r_idx]), sizeof(int) * V_padding * B, cudaMemcpyDeviceToHost);
			}
			#pragma omp barrier
			cudaMemcpy((void*) &(device_matrix[gpu_id][r_idx]), (void*) &(host_matrix[r_idx]), sizeof(int) * V_padding * B, cudaMemcpyHostToDevice);
            // cudaEventRecord(communication_end ); 
			// 	cudaEventSynchronize(communication_end);
			// 	cudaEventElapsedTime(&et, communication_start, communication_end);
			// 	total_communication_time += et;
			

			    
				
				phase1<<< grid_phase1, BLOCK_DIM, sm_size >>>(device_matrix[gpu_id], B, V_padding, r);
				cudaDeviceSynchronize();
				phase2<<< grid_phase2, BLOCK_DIM, sm_size >>>(device_matrix[gpu_id], B, V_padding, r);
				cudaDeviceSynchronize();
				phase3<<< grid_phase3, BLOCK_DIM, sm_size >>>(device_matrix[gpu_id], B, V_padding, r, row_start/B);
				
				

           
        }
        
        // printf("========\nCommunication time: %f\n", total_communication_time);

		//device to host mem_copy
		
			cudaMemcpy((void*) &(host_matrix[index_copy_start]), (void*) &(device_matrix[gpu_id][index_copy_start]), sizeof(int) * V_padding*B*num_blocks_per_gpu, cudaMemcpyDeviceToHost);
            
         
		#pragma omp barrier


	}
	

	// output
   
	
	FILE *fh_out = fopen(argv[2],"w");
	for(int i = 0; i < n; ++i) {
        fwrite(&host_matrix[i*V_padding],sizeof(int), n, fh_out);
	}
	fclose(fh_out);
	

	return 0;
}