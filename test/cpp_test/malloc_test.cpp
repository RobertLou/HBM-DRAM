#include <iostream>
#include "../time/timecalculate.h"

int main(){
    int N;
    int *a, *b;
    CTimeCalculate iTimeCal;

    iTimeCal.StartWork("malloc 1");
    a = (int *)malloc(sizeof(int) * 100000);
    iTimeCal.EndWork("malloc 1");

    iTimeCal.StartWork("malloc 100000");
    for(int i = 0; i < 100000; i++){
        b = (int *)malloc(sizeof(int));
    }
    iTimeCal.EndWork("malloc 100000");
    return 0;
}