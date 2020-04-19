/********************************************************************************
** Form generated from reading UI file 'indexselct.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_INDEXSELCT_H
#define UI_INDEXSELCT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>
#include "qwt_plot.h"

QT_BEGIN_NAMESPACE

class Ui_IndexSelct
{
public:
    QLabel *titleLabel;
    QwtPlot *mainplot;
    QwtPlot *feedrateplot;
    QLabel *currModelLabel;
    QFrame *frame;
    QLabel *brekoutLabel;
    QLineEdit *lineEdit;
    QLabel *modelNameLabel;
    QLabel *fileLabel;
    QPushButton *pushButton;

    void setupUi(QWidget *IndexSelct)
    {
        if (IndexSelct->objectName().isEmpty())
            IndexSelct->setObjectName(QStringLiteral("IndexSelct"));
        IndexSelct->resize(800, 750);
        titleLabel = new QLabel(IndexSelct);
        titleLabel->setObjectName(QStringLiteral("titleLabel"));
        titleLabel->setGeometry(QRect(290, 0, 201, 61));
        QFont font;
        font.setFamily(QStringLiteral("Noto Sans CJK JP"));
        font.setPointSize(20);
        titleLabel->setFont(font);
        mainplot = new QwtPlot(IndexSelct);
        mainplot->setObjectName(QStringLiteral("mainplot"));
        mainplot->setGeometry(QRect(40, 70, 650, 300));
        feedrateplot = new QwtPlot(IndexSelct);
        feedrateplot->setObjectName(QStringLiteral("feedrateplot"));
        feedrateplot->setGeometry(QRect(40, 410, 650, 200));
        currModelLabel = new QLabel(IndexSelct);
        currModelLabel->setObjectName(QStringLiteral("currModelLabel"));
        currModelLabel->setGeometry(QRect(20, 650, 100, 20));
        frame = new QFrame(IndexSelct);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setGeometry(QRect(610, 650, 100, 20));
        frame->setAutoFillBackground(true);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        brekoutLabel = new QLabel(IndexSelct);
        brekoutLabel->setObjectName(QStringLiteral("brekoutLabel"));
        brekoutLabel->setGeometry(QRect(530, 650, 67, 17));
        lineEdit = new QLineEdit(IndexSelct);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(160, 690, 230, 27));
        modelNameLabel = new QLabel(IndexSelct);
        modelNameLabel->setObjectName(QStringLiteral("modelNameLabel"));
        modelNameLabel->setGeometry(QRect(170, 650, 230, 20));
        fileLabel = new QLabel(IndexSelct);
        fileLabel->setObjectName(QStringLiteral("fileLabel"));
        fileLabel->setGeometry(QRect(20, 690, 100, 20));
        pushButton = new QPushButton(IndexSelct);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(610, 690, 100, 27));

        retranslateUi(IndexSelct);

        QMetaObject::connectSlotsByName(IndexSelct);
    } // setupUi

    void retranslateUi(QWidget *IndexSelct)
    {
        IndexSelct->setWindowTitle(QApplication::translate("IndexSelct", "\345\237\272\344\272\216\346\234\272\345\231\250\345\255\246\344\271\240\347\232\204\347\224\265\347\201\253\350\212\261\345\260\217\345\255\224\345\212\240\345\267\245\347\251\277\351\200\217\346\243\200\346\265\213", 0));
        titleLabel->setText(QApplication::translate("IndexSelct", "\346\224\276\347\224\265\345\212\240\345\267\245\347\212\266\346\200\201", 0));
        currModelLabel->setText(QApplication::translate("IndexSelct", "\347\251\277\351\200\217\346\243\200\346\265\213\346\250\241\345\236\213", 0));
        brekoutLabel->setText(QApplication::translate("IndexSelct", "\346\234\252\347\251\277\351\200\217", 0));
        modelNameLabel->setText(QApplication::translate("IndexSelct", "\345\275\223\345\211\215\347\224\250\344\272\216\347\251\277\351\200\217\346\243\200\346\265\213\347\232\204\346\250\241\345\236\213", 0));
        fileLabel->setText(QApplication::translate("IndexSelct", "\346\250\241\346\213\237\344\277\241\345\217\267\346\226\207\344\273\266", 0));
        pushButton->setText(QApplication::translate("IndexSelct", "\350\247\202\345\257\237\345\210\260\347\251\277\351\200\217", 0));
    } // retranslateUi

};

namespace Ui {
    class IndexSelct: public Ui_IndexSelct {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_INDEXSELCT_H
