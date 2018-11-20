gcc -O3 -o sse-o3 sharpenSSE.c ctimer.c
gcc -O2 -o sse-o2 sharpenSSE.c ctimer.c
gcc -O0 -o sse-o0 sharpenSSE.c ctimer.c
gcc -o sse-noflag sharpenSSE.c ctimer.c
gcc -O3 -o t-o3 sharpen_t.c ctimer.c
gcc -O2 -o t-o2 sharpen_t.c ctimer.c
gcc -O2 -o t-o0 sharpen_t.c ctimer.c
gcc -o t-noflag sharpen_t.c ctimer.c