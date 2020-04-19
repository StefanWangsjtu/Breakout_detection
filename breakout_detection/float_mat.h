#ifndef FLOAT_MAT_H
#define FLOAT_MAT_H

#include <vector>
#include <cstdio>
#include <cstddef>             // for size_t
#include <cmath>               // for fabs
#include <vector>

using namespace std;


//! default convergence
static const double TINY_FLOAT = 1.0e-30;

//! comfortable array of doubles
using float_vect = std::vector<double>;
//! comfortable array of ints;
using int_vect = std::vector<int>;



class float_mat:public std::vector<float_vect>
{
private:
    explicit float_mat(); // 关闭隐式转换
    float_mat &operator =(const float_mat &); // 重载赋值运算符

public:
    //! constructor with sizes
        float_mat(const size_t rows, const size_t cols, const double def = 0.0);

        //! copy constructor for matrix
        float_mat(const float_mat &m);

        //! copy constructor for vector
        float_mat(const float_vect &v);

        //! use default destructor
        // ~float_mat() {};

        //! get size
        size_t nr_rows(void) const ;
        //! get size
        size_t nr_cols(void) const;

};

#endif // FLOAT_MAT_H
