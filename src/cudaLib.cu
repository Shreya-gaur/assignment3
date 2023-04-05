
#include "cudaLib.cuh"

#define BLOCK_SIZE 16
__device__ __constant__ float bias_d[512];

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}




int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

int medianFilter_gpu (uint8_t inPixels, ImageDim imgDim, 
	uint8_t outPixels, MedianFilterArgs args) {

	return 0;
}


int runGpuConv (int argc, char ** argv) {

	TensorShape iShape = AlexL1_InShape;
	TensorShape fShape = AlexL1_FilterShape;
	ConvLayerArgs convArgs = AlexL1_ConvArgs;

	std::cout << "Evaluate convolution : \n";
	std::cout << "Input : " << iShape << " \n";
	std::cout << "Filter : " << fShape << " \n";

	TensorShape oShape;

	uint64_t errorCount = evaluateGpuConv(iShape, fShape, oShape, convArgs);
	std::cout << "\nFound " << errorCount << " / " << tensorSize(oShape) << " errors \n";
	return 0;
}

uint64_t evaluateGpuConv (TensorShape iShape, TensorShape fShape, 
	TensorShape & oShape, ConvLayerArgs args) {

	uint64_t errorCount = 0;

	float *input_d, *output_d, *filter_d;

	if (iShape.channels == 0) iShape.channels = 1;

	printf("iShape.height: %i, iShape.width: %i, iShape.channel: %i, iShape.count: %i \n", iShape.height, iShape.width, iShape.channels, iShape.count);
	printf("args.strideH: %i, args.strideW: %i, args.padH: %i, args.padW: %i \n", args.strideH, args.strideW, args.padH, args.padW);
	printf("filter.height: %i, filter.width: %i, filter.channel: %i, filter.batch: %i \n", fShape.height, fShape.width, fShape.channels, fShape.count); 

	oShape.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShape.channels	= (fShape.count);
	oShape.count 	= (iShape.count);				//	Might scale to batch size

	printf("oShape.height: %i, oShape.width: %i, oShape.channel: %i \n", oShape.height, oShape.width, oShape.channels);

	float * in = nullptr;
	float * filter = nullptr;
	float * bias = nullptr; 
	float * out = nullptr;
	float * out_cpu = nullptr;

	int retVal;
	retVal = makeTensor(&in, iShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return -1;
	}
	retVal = makeTensor(&filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return -1;
	}
	retVal = makeVector(&bias, oShape.channels);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return -1;
	}

	if(args.padH != 0 || args.padW != 0){

		TensorShape_t padShape;

		padShape.count = iShape.count;
		padShape.channels = iShape.channels;
		padShape.height = iShape.height + 2 * args.padH;
		padShape.width = iShape.width + 2 * args.padW;

		float* paddedin = (float*) malloc(tensorSize(padShape) *  sizeof(float)); 
		
		for(int ch =0; ch< padShape.channels; ch++){
			for( int i = 0; i < padShape.height; i++ ) {
				for( int j = 0; j < padShape.width; j++ ) {
					int paddedPixelPos = ch * padShape.height * padShape.width + i * padShape.width + j;

					if( i >= args.padH && i < iShape.height + args.padH &&
						j >= args.padW && j < iShape.width + args.padW ) {
						int pixelPos = ch * iShape.height * iShape.width + ( i - args.padH ) * iShape.width + ( j - args.padW);
						paddedin[paddedPixelPos] = in[pixelPos];
					} else {
						paddedin[paddedPixelPos] = 0.0;
					}
				}
			}
		}
		
		free(in);

		in = paddedin;
		iShape.width = padShape.width;
		iShape.height = padShape.height;

	}

	if(cudaMalloc(&input_d,  tensorSize(iShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(iShape) * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! INPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	if(cudaMalloc(&filter_d, tensorSize(fShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(fShape) * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! FILTER MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	// if(cudaMalloc(&bias_d,  oShape.channels * sizeof(float))!=cudaSuccess){
	// 	std::cout<< "Size Requested: "<< oShape.channels * sizeof(float);
	// 	std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! BIAS MEMORY ALLOCATION FAILURE \n";
	// 	return -1;
	// }

	std::cout << "OutShape : " << oShape << " \n";
	out = (float *) malloc (tensorSize(oShape) * sizeof(float));

	if(cudaMalloc(&output_d,tensorSize(oShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(oShape) * sizeof(float);
		std::cout<< "ERROR ERROR!!!!! RUN FOR THE HILLS!!!!!OUTPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	// std::cout << "Input" << "\n"; 

	// for(uint32_t n = 0; n < iShape.count ; n++){
	// 	std::cout<< "Batch: "<< n << "\n";
	// 	for(uint32_t ch = 0; ch < iShape.channels; ch++){
	// 		std::cout<< "Channel: "<< ch << "\n";
	// 		for (uint32_t i = 0; i < iShape.height; i++){
	// 			for(uint32_t j = 0; j < iShape.width; j++){
	// 				std::cout << in[ch * iShape.width * iShape.height + i * iShape.width + j] << " ";
	// 			}
	// 			std::cout << "\n";
	// 		}
	// 		std::cout << "\n";
	// 	}
	// }


	// std::cout << "\n"; 	

	// std::cout << "Filter" << "\n"; 

	// for(uint32_t ch = 0; ch < fShape.channels; ch++){
	// 	std::cout<< "Channel: "<< ch << "\n";
	// 	for (uint32_t i = 0; i < fShape.height; i++){
	// 		for(uint32_t j = 0; j < fShape.width; j++){
	// 			// std::cout << output[ch * oShape.width * oShape.height + i * oShape.width + j] << " @ (" << i << ", " << j << ")" << "\n";
	// 			std::cout << filter[ch * fShape.width * fShape.height + i * fShape.width + j] << " ";
	// 		}
	// 		std::cout << "\n";
	// 	}
	// 	std::cout << "\n";
	// }

	// std::cout << "Bias" << "\n";  

	// for(uint32_t ch = 0; ch < oShape.channels; ch++ ){
	// 	std::cout<< "Channel: "<< ch << "\n";
	// 	std::cout << bias[ch] << "\n ";
	// }

	// std::cout <<"\n";

	cudaMemcpy(input_d, in, tensorSize(iShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_d, filter, tensorSize(fShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(bias_d, bias, oShape.channels * sizeof(float), 0, cudaMemcpyHostToDevice);

	int tileW = BLOCK_SIZE + (fShape.width -  args.strideW);
	int tileH = BLOCK_SIZE + (fShape.height - args.strideH);
	int threadPerBlockH = 16;

	int sharedmemSize = tileW * tileH * sizeof(float);

	dim3 dimBlock(tileW, threadPerBlockH, 1);
	printf("\ndimBlock: (%i, %i, %i)\n", tileW, threadPerBlockH, dimBlock.z);
    dim3 dimGrid(ceil((float)iShape.width / (float)BLOCK_SIZE), ceil((float)iShape.height / (float)BLOCK_SIZE), iShape.channels);
	std::cout << "dimGrid: ("<< dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << ")\n";

	convLayer_gpu<<<dimGrid, dimBlock, sharedmemSize>>>(input_d, iShape, filter_d, fShape, bias_d, output_d, oShape, args, iShape.count);


	dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2(ceil((float)oShape.width / (float)BLOCK_SIZE), ceil((float)oShape.height / (float)BLOCK_SIZE), oShape.channels);
	
	bias_Add<<<dimGrid2, dimBlock2>>>(bias_d, output_d, oShape);

	cudaMemcpy(out, output_d, tensorSize(oShape) * sizeof(float), cudaMemcpyDeviceToHost);	

	// std::cout << "\n"; 	
	// std::cout << "Output GPU" << "\n"; 
	// for(uint32_t n = 0; n < oShape.count; n++){
	// 	std::cout<< "Batch: "<< n << "\n";
	// 	for(uint32_t ch = 0; ch < oShape.channels; ch++){
	// 		std::cout<< "Channel: "<< ch << "\n";
	// 		for (uint32_t i = 0; i < oShape.height; i++){
	// 			for(uint32_t j = 0; j < oShape.width; j++){
	// 				std::cout << out[ch * oShape.width * oShape.height + i * oShape.width + j] << " ";
	// 			}
	// 			std::cout << "\n";
	// 		}
	// 		std::cout << "\n";
	// 	}
	// }

	#ifndef CONV_CHECK_DISABLE
		//	STUDENT: Verify number of errors in output matrix generated by convLayer_gpu
		//	STUDENT: Compare results with CPU output
		//	STUDENT: Return error count
		out_cpu = (float *) malloc (tensorSize(oShape) * sizeof(float));
		
		auto tStart = std::chrono::high_resolution_clock::now();		

		convLayer_cpu(in, iShape, filter, fShape, bias, out_cpu, oShape, args, iShape.count);
		
		auto tEnd= std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = (tEnd- tStart);

		for(uint32_t ch = 0; ch < oShape.channels; ch++){
			for (uint32_t i = 0; i < oShape.height; i++){
				for(uint32_t j = 0; j < oShape.width; j++){
					float output_gpu = out[ch * oShape.width * oShape.height + i * oShape.width + j];
					float output_cpu = out_cpu[ch * oShape.width * oShape.height + i * oShape.width + j];
					if(floor(fabs(output_gpu - output_cpu)) != 0 ){
						// printf("Error at (%i, %i, %i) -> Actual Value: %f GPU Value: %f\n", i, j, ch, out_cpu[ch * oShape.width * oShape.height + i * oShape.width + j], out[ch * oShape.width * oShape.height + i * oShape.width + j]);
						// printf("Error at (%i, %i, %i) -> Difference: %f\n", i, j, ch, floor(output_gpu - output_cpu));
						errorCount += 1;
					}
				}
			}
		}

		// std::cout << "\n"; 	
		// std::cout << "Output CPU" << "\n"; 
	
		// for(uint32_t n = 0; n < oShape.count ; n++){
		// 	std::cout<< "Batch: "<< n << "\n";
		// 	for(uint32_t ch = 0; ch < oShape.channels; ch++){
		// 		std::cout<< "Channel: "<< ch << "\n";
		// 		for (uint32_t i = 0; i < oShape.height; i++){
		// 			for(uint32_t j = 0; j < oShape.width; j++){
		// 				std::cout << out_cpu[ch * oShape.width * oShape.height + i * oShape.width + j] << " ";
		// 			}
		// 			std::cout << "\n";
		// 		}
		// 		std::cout << "\n";
		// 	}
		// }

		std::cout << "\n"; 
		std::cout << "It took " << time_span.count() << " seconds on CPU.";	

	#endif

	free(in);
	free(filter);
	free(bias);
	free(out);

	cudaFree(input_d);
	cudaFree(filter_d);
	// cudaFree(bias_d);
	cudaFree(output_d);

	return errorCount;
}

__global__
void bias_Add (float * bias, float * output, TensorShape oShape){
	uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t channel = blockIdx.z;

	if(row < oShape.height && col < oShape.width && channel < oShape.channels){
		output[channel * oShape.width * oShape.height + row * oShape.width + col] += bias_d[channel];
	}

	return;
}


__global__
void convLayer_gpu ( float * input, TensorShape iShape, 
	float * filter, TensorShape fShape, 
	float * bias, float * output, TensorShape oShape, //removed & after TensorShap 
	ConvLayerArgs args, uint32_t batchSize) {

    extern __shared__ float shrinput[];

	const uint32_t tileW = BLOCK_SIZE + (fShape.width - args.strideW);
	const uint32_t tileH = BLOCK_SIZE + (fShape.height - args.strideH);
	
	const uint32_t nosubBlk = ceil((float)tileH / (float)blockDim.y);

    const uint32_t blockStartCol = blockIdx.x * BLOCK_SIZE;
    const uint32_t blockEndCol = blockStartCol + BLOCK_SIZE;
    const uint32_t blockStartRow = blockIdx.y * BLOCK_SIZE;
    const uint32_t blockEndRow = blockStartRow + BLOCK_SIZE;

    const uint32_t tileStartCol = blockStartCol;
    const uint32_t tileEndCol = blockEndCol + (fShape.width - args.strideW);
    const uint32_t tileEndClampedCol = min(tileEndCol, iShape.width);

    const uint32_t tileStartRow = blockStartRow;
    const uint32_t tileEndRow = blockEndRow + fShape.width - args.strideW;
    const uint32_t tileEndClampedRow = min(tileEndRow, iShape.height);

    uint32_t tilePosCol = threadIdx.x;
    uint32_t iPosCol = tileStartCol + tilePosCol;

	for( uint32_t subBlockNo = 0; subBlockNo < nosubBlk; subBlockNo++ ) {
		uint32_t tilePosRow = subBlockNo * blockDim.y + threadIdx.y;		
		uint32_t iPosRow = tileStartRow + tilePosRow;	
		uint32_t tilePos = tilePosRow * tileW + tilePosCol;

		if( iPosCol < tileEndClampedCol && iPosRow < tileEndClampedRow ) {
			uint32_t iPos = blockIdx.z * iShape.width * iShape.height + iPosRow * iShape.width + iPosCol;
			shrinput[tilePos] = input[iPos];
		}
	
	}

	__syncthreads();

    tilePosCol = threadIdx.x;
    iPosCol = tileStartCol + tilePosCol * args.strideW;

	for(uint32_t m = 0; m < oShape.channels; m++){

		float conv_op = 0;
		
		for(uint32_t subBlockNo = 0; subBlockNo < nosubBlk; subBlockNo++ ) {

			uint32_t tilePosRow = subBlockNo * blockDim.y + threadIdx.y;
			uint32_t iPosRow = tileStartRow + tilePosRow * args.strideH;

			if( iPosCol >= tileStartCol && iPosCol < tileEndClampedCol - (fShape.width - args.strideW) &&
				iPosRow >= tileStartRow && iPosRow < tileEndClampedRow - (fShape.height - args.strideH) ) {
				
				uint32_t oPixelPosCol = iPosCol / args.strideW;
				uint32_t oPixelPosRow = iPosRow / args.strideH;
				uint32_t oPixelPos = oPixelPosRow * oShape.width + oPixelPosCol;
				uint32_t tilePos = (tilePosRow * args.strideH) * tileW + (tilePosCol * args.strideW);

				for( uint32_t i = 0; i < fShape.height; i++ ) {
					for( uint32_t j = 0; j < fShape.width; j++ ) {
							int tilePosOffset = i * tileW + j;
							int coefPos = m * fShape.channels * fShape.width * fShape.height + blockIdx.z * fShape.width * fShape.height + i * fShape.width + j;
							int input_idx = tilePos + tilePosOffset;
							conv_op += shrinput[input_idx] * filter[coefPos];
					}
				}

				__syncthreads();

				atomicAdd(&output[m * oShape.height * oShape.width + oPixelPos], conv_op);

			}

		}
	}
	return;
}


int runGpuGemm (int argc, char ** argv) {

	TensorShape aShape = {1, 1, 6, 4};
	TensorShape bShape = {1, 1, 4, 8};
	TensorShape cShape;
	GemmLayerArgs args = {2, 2, 1};

	// evaluateGpuGemm();

	return 0;
}

int evaluateGpuGemm() {

	return 0;
}

//	STUDENT: Add functions here

