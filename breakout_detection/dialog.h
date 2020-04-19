#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include<QFileDialog>
#include <indexselct.h>

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

private slots:
    void on_selectFile_clicked();

    void on_OkBtn_clicked();

private:
    Ui::Dialog *ui;
    IndexSelct* m_indexSelct;
public:
    QString filename;

signals:
    void infoSignals(QString);

};

#endif // DIALOG_H
