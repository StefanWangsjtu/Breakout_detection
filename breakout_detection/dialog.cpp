#include "dialog.h"
#include "ui_dialog.h"
#include<QDebug>
#include<QMessageBox>
#include<indexselct.h>
Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{

    ui->setupUi(this);
    m_indexSelct = new IndexSelct;
    connect(this,SIGNAL(infoSignals(QString)),m_indexSelct,SLOT(infoSlot(QString)));
    setWindowTitle(tr("模拟加工文件选择"));


}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::on_selectFile_clicked()
{
    filename = QFileDialog::getOpenFileName(this,"open file","/home/stefan/QtObject/Breakout_detection/data");
    ui->lineEdit->setText(filename);
    qDebug()<<filename;
}

void Dialog::on_OkBtn_clicked()
{
    switch(QMessageBox::question(this,tr("Question"),tr("确定加工？"),QMessageBox::Ok|QMessageBox::Cancel,QMessageBox::Ok))
    {
    case QMessageBox::Ok:
    {
        emit infoSignals(filename);
        close();
        m_indexSelct->show();
        break;
    }
    case QMessageBox::Cancel:
        break;
    default:
        break;
    }

}
