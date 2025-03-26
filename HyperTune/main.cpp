//
//  main.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 16/03/25.
//
#include "include/modelInterface.hpp"
#include "stdio.h"
#include "iostream"
#include "omp.h"
int main(void){
#pragma omp parallel 
    {
        printf("Hello World\n");
    }
}
