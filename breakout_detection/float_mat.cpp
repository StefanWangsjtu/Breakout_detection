#include "float_mat.h"
#include <QDebug>

float_mat::float_mat()
{
    qDebug()<<"Enter default constructor of float_mat"<<endl;
}

// Overload assignment operator
float_mat & float_mat::operator =(const float_mat &)
{ return *this; }

//get row number qDebug()<<"Enter default constructor of float_mat"<<endl;
size_t float_mat::nr_rows(void) const
{
    return (*this).size(); qDebug()<<"Enter default constructor of float_mat"<<endl;
}

size_t float_mat::nr_cols(void) const
{
    return this->front().size();
}


// constructor with sizes
float_mat::float_mat(const size_t rows, const size_t cols, const double defval): std::vector<float_vect>(rows)
{
    int i;
    for (i = 0; i < rows; ++i)
    {
        (*this)[i].resize(cols, defval); // defval，如果resize的个数大于原来的，用defval填充
    }
    if ((rows < 1) || (cols < 1))
    {
        char buffer[1024];

        sprintf(buffer, "cannot build matrix with %d rows and %d columns\n",
                rows, cols);
        qDebug()<<buffer;
    }
}

// copy constructor for matrix
float_mat::float_mat(const float_mat &m) : std::vector<float_vect>(m.size())
{
    float_mat::iterator inew = begin();
    float_mat::const_iterator iold = m.begin();

    for (/* empty */; iold < m.end(); ++inew, ++iold)
    {
        const size_t oldsz = iold->size();
        inew->resize(oldsz);
        const float_vect oldvec(*iold);
        *inew = oldvec;
    }
}

// copy constructor for vector
float_mat::float_mat(const float_vect &v): std::vector<float_vect>(1)
{
    const size_t oldsz = v.size();
    front().resize(oldsz);
    front() = v;
}



