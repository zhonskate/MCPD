#include <iostream>
#include <vector>
using namespace std;
int main( int argc, char *argv[] ) {
 cout << "Hola mundo." << endl;
 int i = 1;
 cout << i << endl;
 int *x;
 x = &i;
 cout << x << endl;
 i=5;
 cout << *x << endl;
 return 0;

} 