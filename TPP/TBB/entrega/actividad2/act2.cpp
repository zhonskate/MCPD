#include <stdio.h> 
#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;

class Tabla {
    private:
        int *elements;
        int stored;
    public:
        Tabla () : stored(10){
            elements = new int[stored];
        }
        Tabla (const int n ) : stored(n){
            elements = new int[stored];
        }
        Tabla(const Tabla& t) : stored(t.stored){
            if (&t!=this){
                if (elements!=NULL){
                    delete [] elements;
                }
                elements = new int[stored];
                for (int i=0;i<stored;i++){
                    elements[i]=t.elements[i];
                }
            }
        }
        int& operator[](const int i){
            if(i<0 || i>stored) throw 10;
            return elements[i];
        }
        int getN (){
            return stored;
        }

        Tabla& operator= (const Tabla& t){
            if (&t!=this){
                stored = t.stored;
                if (elements!=NULL){
                    delete [] elements;
                }
                elements = new int[stored];
                for (int i=0; i<stored; i++){
                    elements[i] = t.elements[i];
                }
            }
            return *this;
        }

        ~Tabla (){
            delete [] elements;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tabla& obj){
        // write obj to stream
        return os << obj.elements;
        }

};


int main() {

   Tabla t1; 
   Tabla t2(5);

   for( int i=0; i<t1.getN(); i++ ) { 
     t1[i] = rand() % 100;
   }   
   cout << "Tabla 1: " << t1; 
   for( int i=0; i<t2.getN(); i++ ) { 
     t2[i] = rand() % 100;
   }   
   cout << "Tabla 2: " << t2; 
   Tabla t3(t2);
   cout << "Tabla 3: " << t3; 
   Tabla t4(5);
   t4 = t1; 
   cout << "Tabla 4: " << t4; 

}