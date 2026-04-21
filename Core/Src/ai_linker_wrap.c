#include <stddef.h>

void *__real_malloc(size_t bytes);
void __real_free(void *ptr);

void *__wrap_malloc(size_t bytes)
{
  return __real_malloc(bytes);
}

void __wrap_free(void *ptr)
{
  __real_free(ptr);
}
