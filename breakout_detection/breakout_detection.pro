#-------------------------------------------------
#
# Project created by QtCreator 2018-04-23T15:50:57
#
#-------------------------------------------------

QT       += core gui multimedia multimediawidgets

CONFIG += C++11


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = breakout_detection
TEMPLATE = app


SOURCES += main.cpp\
        indexselct.cpp \
    dialog.cpp \
    float_mat.cpp \
    savitzy_golay_filter.cpp

HEADERS  += indexselct.h \
    dialog.h \
    float_mat.h \
    savitzy_golay_filter.h

FORMS    += indexselct.ui \
    dialog.ui
INCLUDEPATH += /usr/local/qwt-6.1.4-svn/include
INCLUDEPATH += PythonQt
LIBS += -L"/usr/local/qwt-6.1.4-svn/lib" -lqwt

include(common.prf)
include(PythonQt.prf)

#QMAKE_CXXFLAGS += -std=c++11

#QMAKE_CXXFLAGS += -fopenmp -O2

#LIBS += -lgomp -lpthread

DISTFILES += \
    loadModel.py
