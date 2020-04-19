#include "indexselct.h"
#include <QApplication>
#include<dialog.h>
#include "savitzy_golay_filter.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    Dialog dlg;
    dlg.show();


    return a.exec();


}
