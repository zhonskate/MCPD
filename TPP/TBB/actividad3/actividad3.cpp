#include <stdio.h> 
#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;


/* El código desarrollado en la Actividad 3 debe funcionar con el siguiente programa principal. */
/* Si el código está correcto mostrará lo siguiente */
/* a = ( 12, 0 )            */
/* b = ( 0, -11 )           */
/* c = ( 0, 0 )             */
/* d = ( 12, 0 )            */
/* a += b es ( 12, -11 )    */
/* c = ( 12, -22 )          */
/* c = b -a es ( -12, 0 )   */
/* c++ es ( -12, 0 )        */
/* ++c es ( -10, 2 )        */
/* ++a es ( 13, -10 )       */
/* a++ es ( 13, -10 )       */
/* --a es ( 13, -10 )       */
/* a-- es ( 13, -10 )       */
/* b = a-- es ( 0, -11 )    */
/* b = --a es ( 0, -11 )    */

class NumeroR2 {
  double num1, num2;
public:
  NumeroR2() : num1(0.0),num2(0.0) {}
  NumeroR2(double x, double y) : num1(x), num2(y) {}
  NumeroR2(const NumeroR2 &k) : num1(k.num1), num2(k.num2) {}
  friend std::ostream& operator<<(std::ostream& os, const NumeroR2& obj){
    // write obj to stream
    return os << "(" << obj.num1 << "," << obj.num2 << ")" ;
  }

  NumeroR2& operator+=(const NumeroR2& k) {                           
    this->num1 += k.num1;
    this->num2 += k.num2;
    return *this;
  }

  NumeroR2& operator-=(const NumeroR2& k) {                           
    this->num1 -= k.num1;
    this->num2 -= k.num2;
    return *this;
  }

  NumeroR2& operator+(const NumeroR2& k) {                           
    NumeroR2 res;
    res.num1 = this->num1 + k.num1;
    res.num2 = this->num2 + k.num2;
    return res;
  }

  NumeroR2& operator-(const NumeroR2& k) {                           
    NumeroR2 res;
    res.num1 = this->num1 - k.num1;
    res.num2 = this->num2 - k.num2;
    return res;
  }

  NumeroR2& operator++(){
    ++this->num1;
    ++this->num2;
    return *this;
  }

  NumeroR2& operator--(){
    --this->num1;
    --this->num2;
    return *this;
  }

  NumeroR2 operator++(int){
    NumeroR2 tmp(*this);
    operator++(); 
    return tmp;   
  }

  NumeroR2 operator--(int){
    NumeroR2 tmp(*this);
    operator--(); 
    return tmp;   
  }

  NumeroR2& operator=(const NumeroR2& k) {                           
    this->num1 = k.num1;
    this->num2 = k.num2;
    return *this;
  }

};


int main( int argc, char *argv[] ) {
  
  NumeroR2 a( 12.0, 0.0 ), b( 0.0, -11.0 ), c, d(a);
  
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "c = " << c << endl;
  cout << "d = " << d << endl;

  a += b;
  cout << "a += b es " << a << endl;

  c = a + b;
  cout << "c = " << c << endl;

  c = b - a;
  cout << "c = b -a es " << c << endl;

  cout << "c++ es " << c++ << endl;
  cout << "++c es " << ++c << endl;
  cout << "++a es " << ++a << endl;
  cout << "a++ es " << a++ << endl;
  cout << "--a es " << --a << endl;
  cout << "a-- es " << a-- << endl;

  cout << "b = a-- es " << b << endl;
  cout << "b = --a es " << b << endl;

}

