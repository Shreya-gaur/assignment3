
#include "cudaLib.cuh"

#define BLOCK_SIZE 8


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

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, oShape.channels);
    dim3 dimGrid(ceil((float)oShape.width / (float)dimBlock.x), ceil((float)oShape.height / (float)dimBlock.y));

	convLayer_gpu<<<dimGrid, dimBlock>>>(input_d, iShape, filter_d, fShape, bias_d, output_d, oShape, args, 1);

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
	
	int row_gl = blockDim.y * blockIdx.y + threadIdx.y;
	int col_gl = blockDim.x * blockIdx.x + threadIdx.x;
	int channel_gl = blockDim.z * blockIdx.z + threadIdx.z;

	float conv_op = 0;

	if (col_gl < oShape.width && row_gl < oShape.height && channel_gl < oShape.channels) {

		for (uint32_t n = 0; n < batchSize; ++ n) {
			//	For each output fmap value
			//	STUDENT: Set output fmap to bias
			// O[n][m][x][y] = B[m];

			conv_op = bias[channel_gl];
			
			// output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] = bias[channel_gl];\
			
			for (uint32_t i = 0; i < fShape.height; ++ i) {
				for (uint32_t j = 0; j < fShape.width; ++ j) {
					for (uint32_t k = 0; k < fShape.channels; ++ k) {

						//	STUDENT: Calculate
						//	O[n][m][x][y] += 
						//		I[n][k][args.strideH * x][args.strideW * y] *
						//		W[m][k][i][j];

						uint32_t input_row = (args.strideH * row_gl) + i;
						uint32_t input_col = (args.strideW * col_gl) + j;

						float input_element = input[n * iShape.channels * iShape.height * iShape.width + k * iShape.height * iShape.width + input_row * iShape.width + input_col];
						// printf("input[%i][%i][%i][%i] is %f \n", n, k, input_row, input_col, input_element);
						float filter_element = filter[channel_gl * fShape.channels * fShape.height * fShape.width + k * fShape.height * fShape.width + i * fShape.width + j];
						// output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] += input_element * filter_element;
						conv_op += input_element * filter_element;
					}
				}
			}

			output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] = conv_op;
		
		//	STUDENT: Check by disabling activation
		//	STUDENT: Apply Activation here
			if (args.activation) {
				//	O[n][m][x][y] = ReLU( O[n][m][x][y] );
				if (output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] < 0){
					output[n * oShape.channels * oShape.height * oShape.width + channel_gl * oShape.height * oShape.width + row_gl * oShape.width + col_gl] = 0;
				}
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

	evaluateGpuGemm(aShape, bShape, cShape, args);
	return 0;
}

int evaluateGpuGemm(TensorShape aShape, TensorShape bShape, 
	TensorShape & cShape, GemmLayerArgs args) {

	int errorCount = 0;
	
	float * d_a, * d_b, *d_c;

	if (aShape.width != bShape.height || aShape.channels != bShape.channels 
		|| aShape.count != bShape.count) {
		std::cout << "Dimensions dont match : " << aShape << " x " << bShape << " \n";
		return -1;
	}

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = aShape.count;

	printf("cShape.height: %i, cShape.width: %i \n", cShape.height, cShape.width);
	

	float * a = nullptr;
	float * b = nullptr;

	makeTensor(& a, aShape);
	makeTensor(& b, bShape);

	float * c = (float *) malloc(tensorSize(cShape) * sizeof(float));
	
	if(cudaMalloc(&d_a,  tensorSize(aShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(aShape) * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! INPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	if(cudaMalloc(&d_b,  tensorSize(bShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(bShape) * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! INPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	if(cudaMalloc(&d_c,  tensorSize(cShape) * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< tensorSize(cShape) * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! INPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	std::cout << "\n";
	std::cout << "Input a" <<"\n";	

	for (uint32_t i = 0; i < aShape.height; i++){
		for(uint32_t j = 0; j < aShape.width; j++){
			std::cout << a[i * aShape.width + j] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "\n";
	std::cout << "Input b" <<"\n";

	for (uint32_t i = 0; i < bShape.height; i++){
		for(uint32_t j = 0; j < bShape.width; j++){
			std::cout << b[i * bShape.width + j] << " ";
		}
		std::cout << "\n";
	}

	cudaMemcpy(d_a, a, tensorSize(aShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, tensorSize(bShape) * sizeof(float), cudaMemcpyHostToDevice);

	int sharedMemorySize = args.tileW * args.tileH * sizeof(float); 

    dim3 dimBlock(args.tileW, args.tileH);
	dim3 dimGrid(((cShape.width + args.tileW - 1)/args.tileW), ((cShape.height + args.tileH - 1)/args.tileH));

	gemmLayer_gpu<<<dimGrid, dimBlock>>>(d_a, aShape, d_b, bShape, d_c, cShape, args, 1);

	cudaMemcpy(c, d_c, tensorSize(cShape) * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "\n"; 	
	std::cout << "Output GPU" << "\n"; 

	for (uint32_t i = 0; i < cShape.height; i++){
		for(uint32_t j = 0; j < cShape.width; j++){
			std::cout << c[i * cShape.width + j] << " ";
		}
		std::cout << "\n";
	}

	#ifndef CONV_CHECK_DISABLE
		//	STUDENT: Verify number of errors in output matrix generated by convLayer_gpu
		//	STUDENT: Compare results with CPU output
		//	STUDENT: Return error count
		float* out_cpu = (float *) malloc (tensorSize(cShape) * sizeof(float));
		
		auto tStart = std::chrono::high_resolution_clock::now();		

		gemmLayer_cpu (a, aShape, b, bShape, out_cpu, cShape, args, 1);

		auto tEnd= std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = (tEnd- tStart);

		for (uint32_t i = 0; i < cShape.height; i++){
			for(uint32_t j = 0; j < cShape.width; j++){
				float output_gpu = c[i * cShape.width + j];
				float output_cpu = out_cpu[i * cShape.width + j];
				if(floor(fabs(output_gpu - output_cpu)) != 0){
					printf("Error at (%i, %i) -> Actual Value: %f GPU Value: %f\n", i, j, out_cpu[i * cShape.width + j], c[i * cShape.width + j]);
					printf("Error at (%i, %i) -> Difference: %f\n", i, j, floor(output_gpu - output_cpu));
					errorCount += 1;
				}
			}
		}

		std::cout << "\n"; 	
		std::cout << "Output CPU" << "\n"; 

		for (uint32_t i = 0; i < cShape.height; i++){
			for(uint32_t j = 0; j < cShape.width; j++){
				std::cout << out_cpu[i * cShape.width + j] << " ";
			}
			std::cout << "\n";
		}

		std::cout << "\n"; 
		std::cout << "It took " << time_span.count() << " seconds on CPU.";	

	#endif

	std::cout << "\nFound " << errorCount << " / " << tensorSize(cShape) << " errors \n";

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

__global__
void gemmLayer_gpu (float * a, TensorShape aShape,
	float * b, TensorShape bShape,
	float * c, TensorShape cShape,
	GemmLayerArgs args, uint32_t batchSize) {

	int row_l = threadIdx.y;
	int col_l = threadIdx.x;

	int offsetH = blockIdx.y * args.tileH;
	int offsetW = blockIdx.x * args.tileW;

	int row_gl = blockIdx.y * blockDim.y + threadIdx.y;
	int col_gl = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t subTilesAlongK = (aShape.width + args.tileH - 1) / args.tileH;	

	int rowIdx, colIdx;
	int subTile, subTileK, k;

	#ifdef PRINT_DEBUG
		printf("%d @ (%03d, %03d)  = %d\n", threadIdx, 
		row + offsetH, col + offsetW, IDX2R(row + offsetH, col + offsetW, TILE_W));
	#endif

     for (subTile = 0; subTile < subTilesAlongK; ++ subTile) { // Which tile on the Cshape am I on?
	
		rowIdx = row_gl; 
		colIdx = col_gl;

		//  Check bounds of actual output matrix
		if (rowIdx < cShape.height && colIdx < cShape.width) {
			if (subTile == 0)
                c[IDX2R(rowIdx, colIdx, cShape.width)] = 0; //cannot understand this

			for (subTileK = 0; subTileK < args.tileH; ++ subTileK) {
				k = subTile * args.tileH + subTileK;
				if (k < aShape.width) {
					c[IDX2R(rowIdx, colIdx, cShape.width)] += a[IDX2R(rowIdx, k, aShape.width)] * b[IDX2R(k, colIdx, bShape.width)];
				}
			}
		}
	}
}
