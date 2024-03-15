#include <iostream>
#include "../time/timecalculate.h"

int main(){
    int N;
    int *a, *b;
    CTimeCalculate iTimeCal;

    iTimeCal.StartWork("malloc 1");
    a = new int(100000);
    iTimeCal.EndWork("malloc 1");

    iTimeCal.StartWork("malloc 100000");
    for(int i = 0; i < 100000; i++){
        b = new int(1);
    }
    iTimeCal.EndWork("malloc 100000");
    return 0;
}