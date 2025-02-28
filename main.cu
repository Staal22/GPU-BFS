#include <iostream>
#include <nvshmem.h>

#ifdef __JETBRAINS_IDE__
#include <host/nvshmem_api.h>
#endif

int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}