
#include <iostream>
using namespace std;

void Foo( double& f ) { 
  f = f*f; 
}

struct s_item {
  double data;
  struct s_item *next;
};

typedef struct s_item Item;

void SerialApplyFooToList( Item *root ) { 
  for( Item* ptr=root; ptr!=NULL; ptr=ptr->next )
    Foo(ptr->data);
}

int main( )  {

  long n = 10;
  Item *root = NULL;
  root = new( Item );
  root->data = 0.0;
  Item *p; 
  size_t i;
  for( i=1, p = root; i<n; i++, p = p->next ) {
    p->next = new( Item );
    p->next->data = (double) i;
    p->next->next = NULL;
  }
  SerialApplyFooToList( root );
  cout << "Valores = " << endl;
  for( p = root; p!=NULL; p=p->next ) {
    cout << "Dato = " << p->data << endl;
  }

}

