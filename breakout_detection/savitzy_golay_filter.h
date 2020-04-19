#ifndef SAVITZY_GOLAY_FILTER_H
#define SAVITZY_GOLAY_FILTER_H

#include <vector>
#include <QVector>
#include "float_mat.h"
using namespace std;



// savitzky golay smoothing.
/*!
 * \brief sg_smooth
 * \param v vector<double>
 * \param w 2*w = window width
 * \param deg degree of fitting ploymials
 * \return smoothed vector with the same size
 */

std::vector<double> sg_smooth(const std::vector<double> &v, const int w, const int deg);
//! numerical derivative based on savitzky golay smoothing.
std::vector<double> sg_derivative(const std::vector<double> &v, const int w,
    const int deg, const double h = 1.0);

#endif // SAVITZY_GOLAY_FILTER_H
