#include <cstdlib>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

template<class T>
class NumeroR2{
	
  friend std::ostream& operator<<(std::ostream& os, const NumeroR2& obj){
    return os << "(" << obj.num1 << "," << obj.num2 << ")" ;
  }
 protected:
	T num1;
	T num2;
 public:
	NumeroR2(){num1 = 0.0; num2 = 0.0;}
	NumeroR2(T x,T y){num1 = x; num2 = y;}
	NumeroR2(const NumeroR2& k){num1 = k.num1; num2 = k.num2;}

	NumeroR2& operator+=(const NumeroR2& k){
    this->num1 += k.num1;
    this->num2 += k.num2;
		return *this;	 
  }

	NumeroR2& operator-=(const NumeroR2& k){
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

template<class H>
class Complejo:public NumeroR2<H>{
  friend ostream& operator<<(ostream& output, const Complejo k){
    output << k.num1 <<" + "<<k.num2 << "i" << endl;
  };

private:
	H module;	
	void modulo(){
	  module = sqrt(NumeroR2<H>::num1*NumeroR2<H>::num1+NumeroR2<H>::num2*NumeroR2<H>::num2);
	}
public:
	Complejo():NumeroR2<H>(){modulo();}
  Complejo(H x,H y):NumeroR2<H>(x,y){modulo();}
  Complejo(const Complejo& k):NumeroR2<H>(k){modulo();}
	
	Complejo& operator-=(const Complejo& c){
	  NumeroR2<H>::operator-=( c );
	  modulo();
	  return *this;
	}

	Complejo& operator+=(const Complejo& c){
    NumeroR2<H>::operator+=( c );
    modulo();
    return *this;
  }

	Complejo operator-(const Complejo& c){
	  Complejo aux(this->num1 - c.num1,this->num2 - c.num2);
    modulo();
    return aux;
  }

	Complejo operator+(const Complejo& c){
	  Complejo aux(this->num1 + c.num1,this->num2 + c.num2);
    modulo();
    return aux;
  }
	
	Complejo& operator++(){
    NumeroR2<H>::operator++();
    modulo();
    return *this;
  }
		
	Complejo& operator++(int){
    NumeroR2<H>::operator++();
    modulo();
    return *this;
  }
	
	Complejo& operator--(){
    NumeroR2<H>::operator--();
    modulo();
    return *this;
  }

	Complejo& operator--(int){
    NumeroR2<H>::operator--();
    modulo();
    return *this;
  }
	
	Complejo& operator=(const Complejo& c){
    NumeroR2<H>::operator=( c );
    modulo();
    return *this;
  }

  Complejo& operator()(H x,H y){
    NumeroR2<H>::num1 = x;
    NumeroR2<H>::num2 = y;
    modulo();
    return *this;
  }

  Complejo& operator()(H x){
    NumeroR2<H>::num1 = x;
    NumeroR2<H>::num2 = 0.0;
    modulo();
    return *this;
  }
	
	bool operator<(const Complejo& c){
		return module < c.module;
	}

  bool operator>(const Complejo& c){
          return module > c.module;
  }

  bool operator<=(const Complejo& c){
          return module <= c.module;
  }

  bool operator>=(const Complejo& c){
          return module >= c.module;
  }

  bool operator==(const Complejo& c){
    return NumeroR2<H>::num1==c.num1 && NumeroR2<H>::num2==c.num2;
  }

  bool operator!=(const Complejo& c){
    return NumeroR2<H>::num1!=c.num1 || NumeroR2<H>::num2!=c.num2;
  }

	H getMod(){return module;}
		
};
	
void modul(Complejo<double> c){
	cout << c.getMod() << endl;
}

class functor{
  public:
    void operator() (Complejo<double> &c){cout << c.getMod() << endl;}
};

int main( int argc, char *argv[] ) {
  
  Complejo<double> a,b,c;
  vector< Complejo<double> > v;

  a(1.0,3.0);
  	
  b(-11.0,0);
  c = a + b;
  v.push_back(a);
  v.push_back(b);
  v.push_back(c);
  vector< Complejo<double> >::iterator it;
  for_each (v.begin(),v.end(),modul);
  
  for(it = v.begin();it!=v.end();it++){
	  cout << *it << endl;
	}
  	
}

