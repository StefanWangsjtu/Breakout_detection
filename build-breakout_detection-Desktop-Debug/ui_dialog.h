/********************************************************************************
** Form generated from reading UI file 'dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOG_H
#define UI_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QPushButton *selectFile;
    QLineEdit *lineEdit;
    QPushButton *OkBtn;
    QLabel *label;

    void setupUi(QDialog *Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QStringLiteral("Dialog"));
        Dialog->setWindowModality(Qt::ApplicationModal);
        Dialog->resize(600, 400);
        selectFile = new QPushButton(Dialog);
        selectFile->setObjectName(QStringLiteral("selectFile"));
        selectFile->setGeometry(QRect(30, 330, 100, 27));
        lineEdit = new QLineEdit(Dialog);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(150, 330, 200, 27));
        OkBtn = new QPushButton(Dialog);
        OkBtn->setObjectName(QStringLiteral("OkBtn"));
        OkBtn->setGeometry(QRect(470, 350, 99, 27));
        label = new QLabel(Dialog);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(100, 30, 421, 41));
        QFont font;
        font.setFamily(QStringLiteral("Noto Sans Mono CJK KR"));
        font.setPointSize(18);
        label->setFont(font);

        retranslateUi(Dialog);

        QMetaObject::connectSlotsByName(Dialog);
    } // setupUi

    void retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", "Dialog", 0));
        selectFile->setText(QApplication::translate("Dialog", "\351\200\211\346\213\251\345\212\240\345\267\245\346\226\207\344\273\266", 0));
        OkBtn->setText(QApplication::translate("Dialog", "\347\241\256\345\256\232", 0));
        label->setText(QApplication::translate("Dialog", "\345\237\272\344\272\216\346\234\272\345\231\250\345\255\246\344\271\240\347\232\204\347\224\265\347\201\253\350\212\261\345\260\217\345\255\224\347\251\277\351\200\217\346\243\200\346\265\213", 0));
    } // retranslateUi

};

namespace Ui {
    class Dialog: public Ui_Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOG_H
