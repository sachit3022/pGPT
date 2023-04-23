#include <time.h>
static void get_walltime_(double* wcTime) {
  struct timespec tp;
  timespec_get(&tp, TIME_UTC);
  *wcTime = (double)(tp.tv_sec + tp.tv_nsec/1000000000.0);
}

void get_walltime(double* wcTime) {
  get_walltime_(wcTime);
}

