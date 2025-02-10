// cd /home/hork/cuda-workspace/CudaSHA256/Debug/files
// time ~/Dropbox/FIIT/APS/Projekt/CpuSHA256/a.out -f ../file-list
// time ../CudaSHA256 -f ../file-list


#if SIZE_MAX == UINT_MAX
typedef int ssize_t;        /* common 32 bit case */
#define SSIZE_MIN  INT_MIN
#define SSIZE_MAX  INT_MAX
#elif SIZE_MAX == ULONG_MAX
typedef long ssize_t;       /* linux 64 bits */
#define SSIZE_MIN  LONG_MIN
#define SSIZE_MAX  LONG_MAX
#elif SIZE_MAX == ULLONG_MAX
typedef long long ssize_t;  /* windows 64 bits */
#define SSIZE_MIN  LLONG_MIN
#define SSIZE_MAX  LLONG_MAX
#elif SIZE_MAX == USHRT_MAX
typedef short ssize_t;      /* is this even possible? */
#define SSIZE_MIN  SHRT_MIN
#define SSIZE_MAX  SHRT_MAX
#elif SIZE_MAX == UINTMAX_MAX
typedef intmax_t ssize_t;  /* last resort, chux suggestion */
#define SSIZE_MIN  INTMAX_MIN
#define SSIZE_MAX  INTMAX_MAX
#else
#error platform has exotic SIZE_MAX
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "sha256.cuh"
#include "dirent.h">
#include "getopt.h"
#include <ctype.h>

size_t getline(char** lineptr, size_t* n, FILE* stream) {
	char* bufptr = NULL;
	char* p = bufptr;
	size_t size;
	int c;

	if (lineptr == NULL) {
		return -1;
	}
	if (stream == NULL) {
		return -1;
	}
	if (n == NULL) {
		return -1;
	}
	bufptr = *lineptr;
	size = *n;

	c = fgetc(stream);
	if (c == EOF) {
		return -1;
	}
	if (bufptr == NULL) {
		bufptr = (char *)malloc(128);
		if (bufptr == NULL) {
			return -1;
		}
		size = 128;
	}
	p = bufptr;
	while (c != EOF) {
		if ((p - bufptr) > (size - 1)) {
			size = size + 128;
			bufptr = (char *)realloc(bufptr, size);
			if (bufptr == NULL) {
				return -1;
			}
		}
		*p++ = c;
		if (c == '\n') {
			break;
		}
		c = fgetc(stream);
	}

	*p++ = '\0';
	*lineptr = bufptr;
	*n = size;

	return p - bufptr - 1;
}

char* trim(char* str) {
	size_t len = 0;
	char* frontp = str;
	char* endp = NULL;

	if (str == NULL) { return NULL; }
	if (str[0] == '\0') { return str; }

	len = strlen(str);
	endp = str + len;

	/* Move the front and back pointers to address the first non-whitespace
	 * characters from each end.
	 */
	while (isspace((unsigned char)*frontp)) { ++frontp; }
	if (endp != frontp)
	{
		while (isspace((unsigned char)*(--endp)) && endp != frontp) {}
	}

	if (str + len - 1 != endp)
		*(endp + 1) = '\0';
	else if (frontp != str && endp == frontp)
		*str = '\0';

	/* Shift the string so that it starts at str so that if it's dynamically
	 * allocated, we can still free it on the returned pointer.  Note the reuse
	 * of endp to mean the front of the string buffer now.
	 */
	endp = str;
	if (frontp != str)
	{
		while (*frontp) { *endp++ = *frontp++; }
		*endp = '\0';
	}


	return str;
}

__global__ void sha256_cuda(JOB** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n) {
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB** jobs, int n) {
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda << < numBlocks, blockSize >> > (jobs, n);
}


JOB* JOB_init(BYTE* data, long size, char* fname) {
	JOB* j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}


BYTE* get_file_data(char* fname, unsigned long* size) {
	FILE* f = 0;
	BYTE* buffer = 0;
	unsigned long fsize = 0;

	f = fopen(fname, "rb");
	if (!f) {
		fprintf(stderr, "get_file_data Unable to open '%s'\n", fname);
		return 0;
	}
	fflush(f);

	if (fseek(f, 0, SEEK_END)) {
		fprintf(stderr, "Unable to fseek %s\n", fname);
		return 0;
	}
	fflush(f);
	fsize = ftell(f);
	rewind(f);

	//buffer = (char *)malloc((fsize+1)*sizeof(char));
	checkCudaErrors(cudaMallocManaged(&buffer, (fsize + 1) * sizeof(char)));
	fread(buffer, fsize, 1, f);
	fclose(f);
	*size = fsize;
	return buffer;
}

void print_usage() {
	printf("Usage: CudaSHA256 [OPTION] [FILE]...\n");
	printf("Calculate sha256 hash of given FILEs\n\n");
	printf("OPTIONS:\n");
	printf("\t-f FILE1 \tRead a list of files (separeted by \\n) from FILE1, output hash for each file\n");
	printf("\t-h       \tPrint this help\n");
	printf("\nIf no OPTIONS are supplied, then program reads the content of FILEs and outputs hash for each FILEs \n");
	printf("\nOutput format:\n");
	printf("Hash following by two spaces following by file name (same as sha256sum).\n");
	printf("\nNotes:\n");
	printf("Calculations are performed on GPU, each seperate file is hashed in its own thread\n");
}

int main(int argc, char** argv) {
	int i = 0, n = 0;
	size_t len;
	unsigned long temp;
	char* a_file = 0, * line = 0;
	BYTE* buff;
	char option, index;
	ssize_t read;
	JOB** jobs;

	// parse input
	while ((option = getopt(argc, argv, "hf:")) != -1)
		switch (option) {
		case 'h':
			print_usage();
			break;
		case 'f':
			a_file = optarg;
			break;
		default:
			break;
		}


	if (a_file) {
		FILE* f = 0;
		f = fopen(a_file, "r");
		if (!f) {
			fprintf(stderr, "Unable to open %s\n", a_file);
			return 0;
		}

		for (n = 0; getline(&line, &len, f) != -1; n++) {}
		checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB*)));
		fseek(f, 0, SEEK_SET);

		n = 0;
		read = getline(&line, &len, f);
		while (read != -1) {
			//printf("%s\n", line);
			read = getline(&line, &len, f);
			line = trim(line);
			buff = get_file_data(line, &temp);
			jobs[n++] = JOB_init(buff, temp, line);
		}

		pre_sha256();
		runJobs(jobs, n);

	}
	else {
		// get number of arguments = files = jobs
		n = argc - optind;
		if (n > 0) {

			checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB*)));

			// iterate over file list - non optional arguments
			for (i = 0, index = optind; index < argc; index++, i++) {
				buff = get_file_data(argv[index], &temp);
				jobs[i] = JOB_init(buff, temp, argv[index]);
			}

			pre_sha256();
			runJobs(jobs, n);
		}
	}

	cudaDeviceSynchronize();
	print_jobs(jobs, n);
	cudaDeviceReset();
	return 0;
}
