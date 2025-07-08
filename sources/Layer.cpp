#include <iostream>
#include "../headers/Layer.hpp"

using namespace std;

Layer::Layer(){a = b = w = nullptr; Size = 0;}
Layer::Layer(double* a, double* b, double* w, int Size){this->a = a; this->b = b; this->w = w; this->Size = Size;}
Layer::Layer(double* a){this->a = a; Size = 1; b = w = nullptr;}

double* Layer::set_a(double* a){this->a = a; return a;}
double* Layer::set_b(double* b){this->b = b; return b;}
double* Layer::set_w(double* w){this->w = w; return w;}
int Layer::set_Size(int Size){this->Size = Size; return Size;}

double* Layer::get_a(){return this->a;}
double* Layer::get_b(){return this->b;}
double* Layer::get_w(){return this->w;}
int Layer::get_Size(){return this->Size;}
