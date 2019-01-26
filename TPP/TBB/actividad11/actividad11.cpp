
#include <iostream>
#include <list>
#include <iterator>
#include "tbb/parallel_while.h"
using namespace std;
using namespace tbb;

void Foo( double& f ) { 
  f = f*f; 
}

struct s_item {
  double data;
  struct s_item *next;
};

typedef struct s_item Item;

/*void SerialApplyFooToList( Item *root ) { 
  for( Item* ptr=root; ptr!=NULL; ptr=ptr->next )
    Foo(ptr->data);
}*/

class ExecuteFoo {
  public:
	void operator()( Item* it ) const {
	  Foo(it->data);
	}
	typedef Item* argument_type;
};

class LinkedListItems {
  Item* point;
  public:
    bool pop_if_present(Item*& it){
      if(point){
        it = point;
        point = point->next;
        return true;
      }
      else{
        return false;
      }
    }
	LinkedListItems(Item* it) : point(it){}
};

void ParaApplyFooToList(Item* it){
	parallel_while<ExecuteFoo> k;
	LinkedListItems llist(it);
	ExecuteFoo body;
	k.run(llist, body);
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
  ParaApplyFooToList( root );
  cout << "Valores = " << endl;
  for( p = root; p!=NULL; p=p->next ) {
    cout << "Dato = " << p->data << endl;
  }

}

