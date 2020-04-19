#ifndef INDEXSELCT_H
#define INDEXSELCT_H

#include <QWidget>
#include<QFileDialog>
#include<QDebug>
#include<QTime>
#include<QTimer>
#include<QVector>
#include<QFile>
#include<QTextStream>
#include<qwt_plot.h>
#include<qwt_plot_curve.h>
#include<qwt_legend.h>
#include<math.h>
#include<qwt_plot_zoomer.h>
#include<qwt_plot_panner.h>
#include<qwt_plot_magnifier.h>
#include<qwt_plot_grid.h>
#include<qwt_scale_draw.h>
#include<QMessageBox>
#include<QFile>

#include<PythonQt/PythonQt.h>
#include<PythonQt/PythonQtSignal.h>
#include<Python.h>
#include<PythonQt/PythonQtObjectPtr.h>
#include <vector>
#include "savitzy_golay_filter.h"
#include <ctime>

namespace Ui {
class IndexSelct;
}

class IndexSelct : public QWidget
{
    Q_OBJECT

public:
    explicit IndexSelct(QWidget *parent = 0);
    void setupFeedRatePlot();
    void setupMainPlot();


    ~IndexSelct();

private:
    Ui::IndexSelct *ui;
public:
    QString fileName;
    QString modelName;
    QString curveName1;
    QString curveName2;
    QString curveName3;
    QString curveName4;
    QString curveName5;

    QwtPlotCurve *gap_current;
    QwtPlotCurve *Smoothed_current;

    QwtPlotCurve *breakout;
    QwtPlotCurve *feed_curve;
    QwtPlotCurve *Smoothed_feed;

    QTimer updateTimer;
    QVector<double> time;
//    QVector<double> part_time;

    QVector<double> current_data;
    vector<double> Smoothed_c_data;// 平滑后平均电流

    QVector<double> brkout_data;
    QVector<double> feed_data;
    vector<double> Smoothed_f_data; //平滑后进给率
    bool isBrkout = false;
    bool CallPy = true;
    int brk=0;
    QVector<double> Labels;
    PythonQtObjectPtr mainModule;
    QFile file;
    int i = 0;
    int numFeat=0;



private slots:
    void showcurve();
    void readFile();
    void stdOut(const QString);
    void stdErr(const QString);
    void infoSlot(QString);
    void on_pushButton_pressed();
};

#endif // INDEXSELCT_H
