
#include "cudaLib.cuh"

#define BLOCK_SIZE 16


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

	//	STUDENT: Add code here --> Added

	float *input_d, *output_d, *bias_d, *filter_d;

	if (iShape.channels == 0) iShape.channels = 1;

	printf("iShape.height: %i, iShape.width: %i, iShape.channel: %i, iShape.count: %i \n", iShape.height, iShape.width, iShape.channels, iShape.count);
	printf("args.strideH: %i, args.strideW: %i, args.padH: %i, args.padW: %i \n", args.strideH, args.strideW, args.padH, args.padW);
	printf("filter.height: %i, filter.width: %i, filter.channel: %i, filter.batch: %i \n", fShape.height, fShape.width, fShape.channels, fShape.count); 

	oShape.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShape.channels	= (fShape.count);
	oShape.count 	= 1;				//	Might scale to batch size

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

	if(cudaMalloc(&bias_d,  oShape.channels * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< oShape.channels * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! BIAS MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	std::cout << "OutShape : " << oShape << " \n";
	out = (float *) malloc (tensorSize(oShape) * sizeof(float));

	if(cudaMalloc(&output_d,tensorSize(oShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(oShape) * sizeof(float);
		std::cout<< "ERROR ERROR!!!!! RUN FOR THE HILLS!!!!!OUTPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	std::cout << "Input" << "\n"; 

	for(uint32_t ch = 0; ch < iShape.channels; ch++){
		std::cout<< "Channel: "<< ch << "\n";
		for (uint32_t i = 0; i < iShape.height; i++){
			for(uint32_t j = 0; j < iShape.width; j++){
				std::cout << in[ch * iShape.width * iShape.height + i * iShape.width + j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

	std::cout << "\n"; 	

	std::cout << "Filter" << "\n"; 

	for(uint32_t ch = 0; ch < fShape.channels; ch++){
		std::cout<< "Channel: "<< ch << "\n";

		for (uint32_t i = 0; i < fShape.height; i++){
			for(uint32_t j = 0; j < fShape.width; j++){
				// std::cout << output[ch * oShape.width * oShape.height + i * oShape.width + j] << " @ (" << i << ", " << j << ")" << "\n";
				std::cout << filter[ch * fShape.width * fShape.height + i * fShape.width + j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

	std::cout << "Bias" << "\n";  

	for(uint32_t ch = 0; ch < oShape.channels; ch++ ){
		std::cout<< "Channel: "<< ch << "\n";
		std::cout << bias[ch] << " ";
	}

	std::cout <<"\n";

	cudaMemcpy(input_d, in, tensorSize(iShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_d, filter, tensorSize(fShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias_d, bias, oShape.channels * sizeof(float), cudaMemcpyHostToDevice);

	int tileW = BLOCK_SIZE + (fShape.width -  args.strideW);
	int tileH = BLOCK_SIZE + (fShape.height - args.strideH);
	int threadPerBlockH = 16;

	int sharedmemSize = tileW * tileH * iShape.channels * sizeof(float);

	dim3 dimBlock(tileW, threadPerBlockH);
	printf("\ndimBlock: (%i, %i)\n", tileW, threadPerBlockH);
    dim3 dimGrid(ceil((float)iShape.width / (float)BLOCK_SIZE), ceil((float)iShape.height / (float)BLOCK_SIZE));
	std::cout << "dimGrid: ("<< ceil((float)iShape.width / (float)BLOCK_SIZE) << "," << ceil((float)iShape.height / (float) BLOCK_SIZE) << ")\n";

	convLayer_gpu<<<dimGrid, dimBlock, sharedmemSize>>>(input_d, iShape, filter_d, fShape, bias_d, output_d, oShape, args, 1);

	cudaMemcpy(out, output_d, tensorSize(oShape) * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "\n"; 	
	std::cout << "Output GPU" << "\n"; 

	for(uint32_t ch = 0; ch < oShape.channels; ch++){
		std::cout<< "Channel: "<< ch << "\n";
		for (uint32_t i = 0; i < oShape.height; i++){
			for(uint32_t j = 0; j < oShape.width; j++){
				std::cout << out[ch * oShape.width * oShape.height + i * oShape.width + j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

	#ifndef CONV_CHECK_DISABLE
		//	STUDENT: Verify number of errors in output matrix generated by convLayer_gpu
		//	STUDENT: Compare results with CPU output
		//	STUDENT: Return error count
		out_cpu = (float *) malloc (tensorSize(oShape) * sizeof(float));
		
		auto tStart = std::chrono::high_resolution_clock::now();		

		convLayer_cpu(in, iShape, filter, fShape, bias, out_cpu, oShape, args, 1);
		
		auto tEnd= std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = (tEnd- tStart);

		for(uint32_t ch = 0; ch < oShape.channels; ch++){
			for (uint32_t i = 0; i < oShape.height; i++){
				for(uint32_t j = 0; j < oShape.width; j++){
					float output_gpu = out[ch * oShape.width * oShape.height + i * oShape.width + j];
					float output_cpu = out_cpu[ch * oShape.width * oShape.height + i * oShape.width + j];
					if(floor(fabs(output_gpu - output_cpu)) != 0){
						printf("Error at (%i, %i, %i) -> Actual Value: %f GPU Value: %f\n", i, j, ch, out_cpu[ch * oShape.width * oShape.height + i * oShape.width + j], out[ch * oShape.width * oShape.height + i * oShape.width + j]);
						printf("Error at (%i, %i, %i) -> Difference: %f\n", i, j, ch, floor(output_gpu - output_cpu));
						errorCount += 1;
					}
				}
			}
		}

		std::cout << "\n"; 	
		std::cout << "Output CPU" << "\n"; 

		for(uint32_t ch = 0; ch < oShape.channels; ch++){
			std::cout<< "Channel: "<< ch << "\n";
			for (uint32_t i = 0; i < oShape.height; i++){
				for(uint32_t j = 0; j < oShape.width; j++){
					std::cout << out_cpu[ch * oShape.width * oShape.height + i * oShape.width + j] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}	

		std::cout << "\n"; 
		std::cout << "It took " << time_span.count() << " seconds on CPU.";	

	#endif

	free(in);
	free(filter);
	free(bias);
	free(out);

	cudaFree(input_d);
	cudaFree(filter_d);
	cudaFree(bias_d);
	cudaFree(output_d);

	return errorCount;
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

    // Copy the tile into shared memory
    uint32_t tilePixelPosCol = threadIdx.x;
    uint32_t iPixelPosCol = tileStartCol + tilePixelPosCol;

	for(uint32_t ch = 0; ch < iShape.channels; ch++){
		for( uint32_t subBlockNo = 0; subBlockNo < nosubBlk; subBlockNo++ ) {
			// printf("subBlockNo.: %i\n",subBlockNo);

			uint32_t tilePixelPosRow = subBlockNo * blockDim.y + threadIdx.y;		
			uint32_t iPixelPosRow = tileStartRow + tilePixelPosRow;	
			uint32_t tilePixelPos = ch * tileH * tileW + tilePixelPosRow * tileW + tilePixelPosCol;

			if( iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow ) {

				uint32_t iPixelPos = ch * iShape.width * iShape.height + iPixelPosRow * iShape.width + iPixelPosCol;
				shrinput[tilePixelPos] = input[iPixelPos];
				// printf("Loaded element (%i, %i, %i) in shrinput (%i, %i, %i) for block (%i, %i) by thread (%i, %i): %f\n", ch, iPixelPosRow, iPixelPosCol, ch, tilePixelPosRow, tilePixelPosCol, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, shrinput[tilePixelPos]);
			
			}
		
		}
	}

	__syncthreads();

    tilePixelPosCol = threadIdx.x;
    iPixelPosCol = tileStartCol + tilePixelPosCol;

	// for(uint32_t n = 0; n < batchSize; n++){
	for(uint32_t m = 0; m < oShape.channels; m++){
		for(uint32_t ch = 0; ch < iShape.channels; ch++){
			for(uint32_t subBlockNo = 0; subBlockNo < nosubBlk; subBlockNo++ ) {

				uint32_t tilePixelPosRow = subBlockNo * blockDim.y + threadIdx.y;
				uint32_t iPixelPosRow = tileStartRow + tilePixelPosRow;

				if( iPixelPosCol >= tileStartCol && iPixelPosCol < tileEndClampedCol - (fShape.width - args.strideW) &&
					iPixelPosRow >= tileStartRow && iPixelPosRow < tileEndClampedRow - (fShape.height - args.strideH) ) {
					
					uint32_t oPixelPosCol = iPixelPosCol;
					uint32_t oPixelPosRow = iPixelPosRow;
					uint32_t oPixelPos = m * oShape.height * oShape.width + oPixelPosRow * oShape.width + oPixelPosCol;
					uint32_t tilePixelPos = tilePixelPosRow * args.strideH * tileW + tilePixelPosCol * args.strideW;

					float conv_op = bias[m];
					// float conv_op = 0.0;

					for( uint32_t i = 0; i < fShape.height; i++ ) {
						for( uint32_t j = 0; j < fShape.width; j++ ) {
							for (uint32_t k = 0; k < fShape.channels; k++){
								int tilePixelPosOffset = i * tileW + j;
								int coefPos = m * fShape.channels * fShape.width * fShape.height + k * fShape.width * fShape.height + i * fShape.width + j;
								// printf("Loaded element shrinput (%i, %i, %i) and filter (%i, %i, %i, %i) for block (%i, %i) by thread (%i, %i): %f\n", k, tilePixelPosRow, tilePixelPosCol, m, k, i, j, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, shrinput[k * tileW * tileH + tilePixelPos]);
								conv_op += shrinput[k * tileW * tileH + tilePixelPos + tilePixelPosOffset] * filter[coefPos];
								// conv_op += shrinput[tilePixelPos];
							}
						}
					}

					output[oPixelPos] = conv_op;
					// printf("Output Element (%i, %i, %i): %f\n", m, iPixelPosRow, iPixelPosCol, output[oPixelPos]);

				}
			}
		}
	}
	// }
        

	// if (col_gl < oShape.width && row_gl < oShape.height && channel_gl < oShape.channels) {

	// 	for (uint32_t n = 0; n < batchSize; ++ n) {
	// 		//	For each output fmap value
	// 		//	STUDENT: Set output fmap to bias
	// 		// O[n][m][x][y] = B[m];

	// 		conv_op = bias[channel_gl];
			
	// 		// output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] = bias[channel_gl];\
			
	// 		for (uint32_t i = 0; i < fShape.height; ++ i) {
	// 			for (uint32_t j = 0; j < fShape.width; ++ j) {
	// 				for (uint32_t k = 0; k < fShape.channels; ++ k) {

	// 					//	STUDENT: Calculate
	// 					//	O[n][m][x][y] += 
	// 					//		I[n][k][args.strideH * x][args.strideW * y] *
	// 					//		W[m][k][i][j];

	// 					uint32_t input_row = (args.strideH * row_gl) + i;
	// 					uint32_t input_col = (args.strideW * col_gl) + j;

	// 					float input_element = input[n * iShape.channels * iShape.height * iShape.width + k * iShape.height * iShape.width + input_row * iShape.width + input_col];
	// 					// printf("input[%i][%i][%i][%i] is %f \n", n, k, input_row, input_col, input_element);
	// 					float filter_element = filter[channel_gl * fShape.channels * fShape.height * fShape.width + k * fShape.height * fShape.width + i * fShape.width + j];
	// 					// output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] += input_element * filter_element;
	// 					conv_op += input_element * filter_element;
	// 				}
	// 			}
	// 		}

	// 		output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] = conv_op;
		
	// 	//	STUDENT: Check by disabling activation
	// 	//	STUDENT: Apply Activation here
	// 		if (args.activation) {
	// 			//	O[n][m][x][y] = ReLU( O[n][m][x][y] );
	// 			if (output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] < 0){
	// 				output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] = 0;
	// 			}
	// 		}
	// 	}
	// }
	return;
}


int runGpuGemm (int argc, char ** argv) {

	TensorShape aShape = {1, 1, 6, 4};
	TensorShape bShape = {1, 1, 4, 8};
	TensorShape cShape;
	GemmLayerArgs args = {2, 2, 1};

	evaluateGpuGemm();
	return 0;
}

int evaluateGpuGemm() {

	// if (aShape.width != bShape.height || aShape.channels != bShape.channels 
	// 	|| aShape.count != bShape.count) {
	// 	std::cout << "Dimensions dont match : " << aShape << " x " << bShape << " \n";
	// 	return -1;
	// }

	// cShape.height = aShape.height;
	// cShape.width = bShape.width;
	// cShape.channels = aShape.channels;
	// cShape.count = aShape.count;

	// float * a = nullptr;
	// float * b = nullptr;

	// makeTensor(& a, aShape);
	// makeTensor(& b, bShape);

	// // float * c = (float *) malloc(tensorSize(cShape) * sizeof(float));

	// // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, cShape.channels);
    // // dim3 dimGrid(ceil((float)oShape.width / (float)dimBlock.x), ceil((float)oShape.height / (float)dimBlock.y));

	// // gemmLayer_cpu<<<dimGrid, dimBlock>>>(a, aShape, b, bShape, c, cShape, args, 1);

	return 0;
}

//	STUDENT: Add functions here

