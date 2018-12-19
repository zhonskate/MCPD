#include <iostream>
#include <tbb/task_scheduler_init.h>
using namespace std;
using namespace tbb;
int main() {
    task_scheduler_init init;
    cout << "Numero de threads = " << init.default_num_threads() << endl;
    return 0;
}