#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

class Layer
{
private:
    int Size;
    double* a;
    double* b;
    double* w;
public:
///diffrent types of layers
    Layer();
    Layer(double* a, double* b, double* w, int Size);
    Layer(double* a);

    double* set_a(double* a);
    double* set_b(double* b);
    double* set_w(double* w);
    int set_Size(int Size);

    double* get_a();
    double* get_b();
    double* get_w();
    int get_Size();

};

#endif // LAYER_HPP_INCLUDED
