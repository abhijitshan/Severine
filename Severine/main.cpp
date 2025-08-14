#include "iostream"
#include "mlx/mlx.h"
int main(void){
    std::cout<<"MLX Installation Test\n";
    try {
        auto mlxArray = mlx::core::array({1.0f, 2.0f, 3.0f});
        std::cout<<"Array created as "<<mlxArray<<std::endl;
    } catch (const std::exception& errorRef){
        
        std::cout<<"Test failed with Exception: "<< errorRef.what() << std::endl;
    }
}
