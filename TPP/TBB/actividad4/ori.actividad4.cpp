#include <stdio.h> 
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
using namespace std;

template <typename T>
class NumeroR2 {
  public:
    T num1, num2;
    NumeroR2() : num1(0.0),num2(0.0) {}
    NumeroR2(T x, T y) : num1(x), num2(y) {}
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

};

template <typename T>
class Complejo : public NumeroR2<T>{
  public:
    T module;
    Complejo() : NumeroR2<T>() {modulo();}
    Complejo(T x, T y) : num1(x), num2(y) {modulo();}
    Complejo(const NumeroR2<T> &k) : NumeroR2<T>(const NumeroR2<T> &k) {modulo();}
    Complejo(const Complejo &k) : num1(k.num1), num2(k.num2), module(k-module) {}
    friend std::ostream& operator<<(std::ostream& os, const NumeroR2<T>& obj){
      // write obj to stream
      return os << " " << obj.num1 << " + " << obj.num2 << "i" ;
    }

    Complejo& operator+=(const Complejo& k) {                           
      NumeroR2<T>::operator+=(k);
      modulo();
      return *this;
    }

    Complejo& operator-=(const Complejo& k) {                           
      NumeroR2<T>::operator-=(k);
      modulo();
      return *this;
    }

    Complejo& operator+(const Complejo& k) {                           
      Complejo res;
      res.num1 = this->num1 + k.num1;
      res.num2 = this->num2 + k.num2;
      res.modulo();
      return res;
    }

    Complejo& operator-(const Complejo& k) {                           
      Complejo res;
      res.num1 = this->num1 - k.num1;
      res.num2 = this->num2 - k.num2;
      res.modulo();
      return res;
    }

    Complejo& operator++(){
      ++this->num1;
      ++this->num2;
      modulo();
      return *this;
    }

    Complejo& operator--(){
      --this->num1;
      --this->num2;
      modulo();
      return *this;
    }

    Complejo operator++(int){
      Complejo tmp(*this);
      operator++();
      modulo(); 
      return tmp;   
    }

    Complejo operator--(int){
      Complejo tmp(*this);
      operator--(); 
      modulo();
      return tmp;   
    }

  Complejo& operator=(const Complejo& k) {                           
    this->num1 = k.num1;
    this->num2 = k.num2;
    this->module = k.module;
    return *this;
  }

  Complejo& operator<(const Complejo& k) {                           
    return this->module < k.module;
  }

  Complejo& operator>(const Complejo& k) {                           
    return this->module > k.module;
  }

  Complejo& operator>=(const Complejo& k) {                           
    return this->module >= k.module;
  }

  Complejo& operator<=(const Complejo& k) {                           
    return this->module <= k.module;
  }

  Complejo& operator==(const Complejo& k) {                           
    return this->num1 == k.num1 && this->num2 == k.num2;
  }

  Complejo& operator!=(const Complejo& k) {                           
    return this->num1 != k.num1 || this->num2 != k.num2;
  }



  private:
    void modulo(){
      this->module = sqrt(this->num1*this->num1+this->num2*this->num2);
    }
};




int main( int argc, char *argv[] ) {
  
  Complejo<double> a( 12.0, 0.0 ), b( 0.0, -11.0 ), c, d(a);
  
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

