#include "indexselct.h"
#include "ui_indexselct.h"
#include<QIODevice>
#include<dialog.h>
#include<QVariantList>
#include<QObject>
#include<qmetatype.h>
#include<QVariantList>
#include<QColor>
#include<QPalette>
#include<QtGlobal>
#include<qwt_scale_div.h>

#define numExp 499

IndexSelct::IndexSelct(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::IndexSelct)
{
    ui->setupUi(this);
    ui->frame->setFrameShape(QFrame::Box);
    ui->frame->setAutoFillBackground(true);
    QColor c = QColor(Qt::gray);
    ui->frame->setPalette(QPalette(c));

    PythonQt::init();
    mainModule = PythonQt::self()->getMainModule();
    connect(PythonQt::self(),SIGNAL(pythonStdOut(QString)),this,SLOT(stdOut(const QString)));
    connect(PythonQt::self(),SIGNAL(pythonStdErr(QString)),this,SLOT(stdErr(const QString)));
    mainModule.evalFile("/home/stefan/QtObjects/breakout_detection/loadModel.py");
    //    modelName = mainModule.call("getModelName").toString();

    mainModule.call("hello");
    ui->modelNameLabel->setText(modelName);
    ui->pushButton->hide();


    for(int i = 0;i<numExp+1;i++)
    {
        time.append((double(i)));// * 10 ms

//        if (i >= 400)
//            part_time.append(i);

    }

    Smoothed_c_data = vector<double>(500, 0.0);
    Smoothed_f_data = vector<double>(500, 0.0);
    feed_data = QVector<double>(500, 0.0);
    current_data = QVector<double>(500, 0.0);
    brkout_data = QVector<double>(500, 0.0);
//    qDebug()<<part_time[0]<<","<<part_time.size();


}

IndexSelct::~IndexSelct()
{
    delete ui;
}

// 进给率曲线设置
void IndexSelct::setupFeedRatePlot()
{

    ui->feedrateplot->setGeometry(40,410,650,200);
    ui->feedrateplot->setTitle("FeedRate");
    ui->feedrateplot->setCanvasBackground(Qt::white);
    ui->feedrateplot->insertLegend(new QwtLegend(), QwtPlot::RightLegend);


    curveName3 = "FeedRate" ;
    feed_curve = new QwtPlotCurve;
    feed_curve->setTitle(curveName3);
    feed_curve->setPen(Qt::green, 1);
    feed_curve->attach(ui->feedrateplot);

    curveName5 = "Smoothed Feedrate" ;
    Smoothed_feed = new QwtPlotCurve;
    Smoothed_feed->setTitle(curveName5);
    Smoothed_feed->setPen(Qt::black, 2);
    Smoothed_feed->attach(ui->feedrateplot);

    //设置轴的标签和坐标刻度
    ui->feedrateplot->setAxisTitle(QwtPlot::xBottom,"Times(x10ms)");
    ui->feedrateplot->setAxisTitle(QwtPlot::yLeft,"FeedRate \n (x 16mm/s)");
    ui->feedrateplot->setAxisScale(QwtPlot::yLeft,-50,50,10);
    ui->feedrateplot->setAxisScale(QwtPlot::xBottom,0,numExp,100);

    //设置网格线
    QwtPlotGrid *grid = new QwtPlotGrid();
    grid->enableX(true);
    grid->enableY(true);
    grid->setMajorPen(Qt::black,0,Qt::DotLine);
    grid->attach(ui->feedrateplot);

    QwtPlotZoomer *zoomer = new QwtPlotZoomer(ui->feedrateplot->canvas());
    zoomer->setRubberBandPen(QColor(Qt::blue));//选择框颜色
    zoomer->setTrackerPen(QColor(Qt::green));//选择框字体颜色
    zoomer->setMousePattern(QwtEventPattern::MouseSelect2,Qt::RightButton,Qt::ControlModifier);//左键拖拽放大，ctrl+右键返回初始状态
    zoomer->setMousePattern(QwtEventPattern::MouseSelect3,Qt::RightButton);//右键返回上一级，
    connect(&updateTimer,SIGNAL(timeout()),this,SLOT(showcurve()));
    updateTimer.start(10);

}

// 脉冲频率
void IndexSelct::setupMainPlot()
{

    ui->mainplot->setGeometry(40,70,650,300);
    ui->mainplot->setTitle("Average Current");
    ui->mainplot->setCanvasBackground(Qt::white);
    ui->mainplot->insertLegend(new QwtLegend(),QwtPlot::RightLegend);


    curveName1 = "Average \n Current";
    gap_current = new QwtPlotCurve;
    gap_current->setTitle(curveName1);
    gap_current->setPen(Qt::blue,1);
    gap_current->attach(ui->mainplot);

    curveName2 = "Breakout";
    breakout = new QwtPlotCurve;
    breakout->setTitle(curveName2);
    breakout->setPen(Qt::black,3);
    breakout->attach(ui->mainplot);

    curveName4 = "Smoothed current";
    Smoothed_current = new QwtPlotCurve;
    Smoothed_current->setTitle(curveName4);
    Smoothed_current->setPen(Qt::red,2);
    Smoothed_current->attach(ui->mainplot);


    //设置轴的标签和坐标刻度
    ui->mainplot->setAxisTitle(QwtPlot::xBottom,"Times(x10ms)");
    ui->mainplot->setAxisTitle(QwtPlot::yLeft,"Average Current(A)");
    ui->mainplot->setAxisScale(QwtPlot::yLeft,0,3000,500);
    ui->mainplot->setAxisScale(QwtPlot::xBottom,0,numExp,100);


    QwtPlotGrid *grid = new QwtPlotGrid();
    grid->enableX(true);
    grid->enableY(true);
    grid->setMajorPen(Qt::black,0,Qt::DotLine);
    grid->attach(ui->mainplot);

    QwtPlotZoomer *zoomer = new QwtPlotZoomer(ui->mainplot->canvas());
    zoomer->setRubberBandPen(QColor(Qt::blue));//选择框颜色
    zoomer->setTrackerPen(QColor(Qt::green));//选择框字体颜色
    zoomer->setMousePattern(QwtEventPattern::MouseSelect2,Qt::RightButton,Qt::ControlModifier);//左键拖拽放大，ctrl+右键返回初始状态
    zoomer->setMousePattern(QwtEventPattern::MouseSelect3,Qt::RightButton);//右键返回上一级，


}

void IndexSelct::showcurve()
{
    QTime s;
    s.start();
    readFile();
    feed_curve->setSamples(time, feed_data);
    gap_current->setSamples(time, current_data);
    breakout->setSamples(time, brkout_data);



//    vector<double> temp1 = current_data.toStdVector();
//     vector<double> temp2 = feed_data.toStdVector();

     //    Smoothed_c_data = sg_smooth(vector<double>(temp1.begin() + 400, temp1.end()), 10, 2);
     //    Smoothed_f_data = sg_smooth(vector<double>(temp2.begin() + 400, temp2.end()), 10, 2);


     Smoothed_c_data = sg_smooth(current_data.toStdVector(), 5, 2);
     Smoothed_f_data = sg_smooth(feed_data.toStdVector(), 5, 2);


    Smoothed_current->setSamples(time, QVector<double>::fromStdVector(Smoothed_c_data));
    Smoothed_feed->setSamples(time, QVector<double>::fromStdVector(Smoothed_f_data));

    ui->mainplot->replot();
    ui->feedrateplot->replot();

    qDebug()<<s.elapsed();
}




void IndexSelct::readFile()
{
    if(!file.atEnd())
    {
        QVector<double> a;
        QString line = QString(file.readLine());
        QStringList list = line.split(QRegExp("[ \n]"),QString::SkipEmptyParts);

//        double label;
        if(i==0)
        {numFeat = list.size();i=1;}


        for(int j=0;j<numFeat;j++)
        {
            double y = list[j].toDouble();
            //            if(CallPy==true)
            //            {mainModule.call("creatData",QVariantList()<<y);}//在Python文件中创建testingSet,如果需要处理数据，在python中处理
            a.append(y);
        }
#if 0
        //获得标签
        /*
        if(isBrkout == false)
        {
            mainModule.call("creatLabel",QVariantList()<<0);

        }
        else
        {mainModule.call("creatLabel",QVariantList()<<1);}
*/
        if(CallPy==true)
        {
            label = mainModule.call("callClassifier", QVariantList()<<modelName).toDouble();//对每一行，调用模型进行分类
            if(label == 1)
            {brk++;}
            mainModule.call("clearData");//清除testingSet

        }
        else
        {
            label =0.7;
            QColor c = QColor(Qt::green);
            ui->frame->setPalette(QPalette(c));
            ui->brekoutLabel->setText(tr("已穿透"));
        }
        if(brk>4)//5个采样周期内均判断为正，则穿透
        {
            label=0.7;
            CallPy=false;
        }
        else
            label=0;
#endif
        int m;

//#if defined(_OPENMP)
//#pragma omp parallel for private(m) schedule(static)
//#endif

        // 更新数据
        for(m=0;m < numExp; m++)
        {
            current_data[m] = current_data[m+1];
            brkout_data[m] = brkout_data[m+1];
            feed_data[m] = feed_data[m+1];
        }
        current_data[numExp] = a[0];
        feed_data[numExp] = a[1];
        brkout_data[numExp] = 0;


        /* 测试画图工具
        i++;
        for(int j=0;j<999;j++)
        {

            curve1y[j] = curve1y[j+1];
            curve3y[j] = curve3y[j+1];
            curve2y[j] = curve2y[j+1];
        }
        qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
        double a = double(qrand()%99)/100;
        curve3y[999] = 0.5*sin(double(i)*M_PI/90)+0.5;
        curve1y[999] = a;curve2y[999] = 0.5*sin(double(i)*M_PI/90)+0.5;

   */
    }
    else
    {
        //        mainModule.call("closefile");

        for(int m=0;m<numExp;m++)
        {
            current_data[m] = current_data[m+1];
            brkout_data[m] = brkout_data[m+1];
            feed_data[m] = feed_data[m+1];
        }
        current_data[numExp] = 0;
        brkout_data[numExp] = 0;
        feed_data[numExp] = 0;
    }


}

void IndexSelct::stdOut(const QString output)
{
    qDebug()<<output;
}
void IndexSelct::stdErr(const QString outPut)
{
    qDebug()<<outPut;
}


void IndexSelct::infoSlot(QString str1)
{
    fileName = str1;
    ui->lineEdit->setText(str1);
    file.setFileName(fileName);

    if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        qDebug()<<"Cannot open this file!";

    }
    setupFeedRatePlot();
    setupMainPlot();


}



void IndexSelct::on_pushButton_pressed()
{
    isBrkout = true;
}
