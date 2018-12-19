#include <iostream>
#include <vector>
using namespace std;

void pointer (int *a)
{
    *a += 1;
    cout << *a << endl;
}

void reference (int &a)
{
    a += 1;
    cout << a << endl;
}

int main( int argc, char *argv[] ) {
    cout << "Hola mundo." << endl;
    int i = 1;
    cout << i << endl;
    int *x;
    x = &i;
    cout << x << endl;
    i=5;
    cout << *x << endl;

    pointer(x);
    reference(i);

    return 0;
} 