from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from pytweezer import *
from pytweezer.servers import Properties, tweezerpath,  icon_path, PropertyAttribute
from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.servers.datamgr import DataSummary, SingleDataSummary, SingleImageSummary
from pytweezer.servers import send_info
from pytweezer.servers import tweezerpath
import subprocess
import os
import time
from pytweezer.GUI.pytweezerQt import BWidget, SearchComboBox
import pyqtgraph as pg
import numpy as np
from scipy import odr
from scipy.stats import sem
import math
import matplotlib.pyplot as plt
from subprocess import call
from fitfunctions import *
from uncertainties import ufloat, ufloat_fromstr
import collections
import pandas as pd
import datetime
from pytweezer.analysis.print_messages import print_error
import traceback


class H5StorageGui(BWidget):
    _currentIndex = PropertyAttribute('CurrentTab', 0)

    def __init__(self, parent=None):
        super().__init__('GUI/H5Storage', parent)
        self.storageGUI = H5StorageDataMgr(self._props)
        self.plottingGUI = H5DataPlotter(self._props, self.storageGUI)

        self.qlayout = QVBoxLayout()
        tabs = QTabWidget()
        self.qlayout.addWidget(tabs)
        tabs.addTab(self.storageGUI, 'Select Data')
        tabs.addTab(self.plottingGUI, 'Plot Data')

        tabs.setCurrentIndex(self._currentIndex)
        tabs.currentChanged.connect(self.updateSelectedTab)

        self.setLayout(self.qlayout)
        saveBtn = QPushButton('Save')
        saveBtn.clicked.connect(self.storageGUI.savetoFile)
        clearBtn = QPushButton('Clear')
        clearBtn.clicked.connect(self.storageGUI.clear)

        bottomline = QHBoxLayout()
        bottomline.addWidget(self.storageGUI.foldernameLEd)
        bottomline.addWidget(saveBtn)
        bottomline.addWidget(clearBtn)
        self.qlayout.addLayout(bottomline)

        # self.resize(250, 150)
        # self.move(300, 300)

    def updateSelectedTab(self, index):
        self._currentIndex = index


# ----------------- PLOTTING --------------------------------------------------


class BinningWidget(QFrame):
    def __init__(self, datasource, parent=None):
        '''

        :datasource:  (a class containing a nextDataProcess Variable)
                    the parent data source
        '''
        super().__init__(parent)
        self.setStyleSheet(
            "BinningWidget {background-color: rgb(180,200,225);color:red; margin:2px; border:3px solid rgb(0, 0, 80);} ")

        self._source = datasource
        self._props = self._source._props
        self._plotitem = self._source._plotitem
        self._label = self._source._label + '/bin'
        self.data = np.array([])
        self.nextDataProcess = None

        self.binMode = QComboBox()  # binning modes
        self.binMode.addItem('Summarize equal x values')
        self.binMode.addItem('Binning according to bins below')
        closeBtn = QPushButton('')
        closeBtn.setMaximumWidth(20)
        closeBtn.setIcon(QtGui.QIcon(icon_path+'terminate.png'))
        closeBtn.clicked.connect(self._close)
        self.binEdit = QLineEdit('np.linspace(1,20,7)')
        self.binEdit.hide()
        self.averagingMode = QComboBox()
        self.averagingMode.addItem('Default averaging')

        # Layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.binMode)
        hlayout.addWidget(closeBtn)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(3, 3, 3, 5)
        self.setLayout(layout)
        layout.addLayout(hlayout)
        layout.addWidget(self.binEdit)
        layout.addWidget(self.averagingMode)

        self._initPlot()

    def _initPlot(self):
        ''' configure the plot of this curve

            :plotitem: (pyqtgraph.PlotItem)  plot environment this plot should be added to
        '''
        plotitem = self._plotitem

        self.errorbar = pg.ErrorBarItem(x=np.array([]), y=np.array([]), height=0)
        plotitem.addItem(self.errorbar)

        self.plot = pg.PlotDataItem()
        plotitem.addItem(self.plot)

        self._configurePlot()

    def _configurePlot(self):
        self.plot.setPen(None)
        self.plot.setSymbol(self._props.get(self._label + '/Symbol[o,s,t,d,+]', 'o'))
        col = QColor(self._props.get(self._label + '/Linecolor', int(255 * 256 ** 3 + 255 * 256 + 255)))
        width = self._props.get(self._label + '/linewidth', 1)
        pen = pg.mkPen(col, width=width)
        self.plot.setSymbolPen(pen)
        self.plot.setSymbolBrush(
            pg.mkBrush(QColor(self._props.get(self._label + '/Fillcolor', int(255 * 256 ** 2 + 255)))))
        self.plot.setSymbolSize(self._props.get(self._label + '/SymbolSize', 5))
        self.plotData()

    def _close(self):
        ''' close this widget and remove from parent '''
        self._source.nextDataProcess = self.nextDataProcess
        if self.nextDataProcess is not None:
            self.nextDataProcess._source = self._source
        self._plotitem.removeItem(self.plot)
        self._plotitem.removeItem(self.errorbar)
        self.close()

    def updateData(self, data):
        ''' process new data'''
        if data.shape[0] > 1:
            if self.binMode.currentIndex() == 0:
                self.summarizeData(data)
        else:
            self.data = np.array([])
        if self.nextDataProcess is not None:
            self.nextDataProcess.updateData(self.data)
        self.plotData()

    def plotData(self):
        # plot the data
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            self.plot.setData(x=self.data[0], y=self.data[1])

            col = QColor(self._props.get(self._label + '/Linecolor', int(256 ** 4 - 1)))
            width = self._props.get(self._label + '/linewidth', 1)
            pen = pg.mkPen(col, width=width)
            x = self.data[0]
            beam = self._props.get(self._label + '/ErorBarBeam', 0.3) * (max(x) - min(x)) / len(x)
            self.errorbar.setData(x=self.data[0], y=self.data[1], height=self.data[2], beam=beam, pen=pen)
            # self.errplot.setData(x=w.dat[0],y=w.dat[1])
        else:
            self.plot.setData(x=[], y=[])
            self.errorbar.setData(x=np.array([]), y=np.array([]), height=0)

    def summarizeData(self, data):
        ''' summarize data with equal x values '''
        xvals = np.array(list(set(data[0])))
        yvals = [data[1][data[0] == i] for i in xvals]
        self.binnedData = [xvals, yvals]
        self.averageData()

    def averageData(self):
        if self.averagingMode.currentText() == 'Default averaging':
            xvals = self.binnedData[0]
            yvals = np.array([i.mean() for i in self.binnedData[1]])
            yerrs = np.array([i.std() for i in self.binnedData[1]])
            ny = np.array([float(len(i)) for i in self.binnedData[1]])
            valid = ny - 1 > 0
            yerrs[valid] = yerrs[valid] / np.sqrt(ny[valid] - 1)
            self.data = np.array([xvals, yvals, yerrs])

        self.plotData()

    def plotmpl(self, ax):
        ''' plot into matplotlib pyplot axis'''
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            width = self._props.get(self._label + '/linewidth', 1)
            ax.errorbar(self.data[0], self.data[1], self.data[2], fmt='.', elinewidth=width, capsize=3)
        if self.nextDataProcess is not None:
            self.nextDataProcess.plotmpl(ax)


def errorformat(number, uc):
    """
        This function takes a number with uncertainty uc,
                    number +/- uc,
        and returns the formatted string in shorthand notation as in https://pythonhosted.org/uncertainties/user_guide.html.

        Examples:
            - number = 321615, uc = 0.1645: '3.21615000(165)e+05'

        The number in the formatted string will have <= number_digits digits, the uncertainty <= uc_digits.

        number: float, the decimal without uncertainty
        uc: float, the uncertainty of number

        returns: str, the formatted string in shorthand notation
    """

    number_digits = 8
    uc_digits = 3

    try:
        if uc < 1e-300 and uc != 0:
            uc = 0
        if number < 1e-300 and number != 0:
            uc = 0

        uc_number = ufloat(number, uc)
        short_hand = uc_number
        short_hand = '{num:.{digits}ueS}'.format(num=uc_number, digits=uc_digits)

        if '(' in short_hand:
            if len(short_hand.split('(')[0]) > number_digits + 2:
                short_hand = '{num:.{digits}e}'.format(num=uc_number, digits=number_digits)
                short_hand = '{num:.{digits}ueS}'.format(num=ufloat_fromstr(short_hand), digits=number_digits)
    except:
        print_error('h5storage.py - errorformat(): Error parsing the fit result. Input number = {0}, input uncertainty = {1}'.format(number, uc), 'error')
        traceback.print_exc()

    return short_hand


class FitParameters(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.function_text = QLabel('')
        layout.addWidget(self.function_text, 1, 0, 1, 4)
        self.setLayout(layout)
        self.qlayout = layout
        self.parnames = []
        self.fixbuttons = []
        self.startvalues = []
        self.fitresults = []
        self.nparameters = 0

    def setFit(self, fitinfo):
        ''' configure box for new fit'''
        for label in self.parnames:
            label.hide()
        for label in self.fixbuttons:
            label.hide()
        for label in self.startvalues:
            label.hide()
        for label in self.fitresults:
            label.hide()

        self.function_text.setText(fitinfo['function_text'])
        for i in range(len(fitinfo['par_names']) - len(self.parnames)):  ## create missing lines
            self.parnames.append(QLabel(''))
            self.qlayout.addWidget(self.parnames[-1], len(self.parnames) + 1, 0)
            self.fixbuttons.append(QPushButton('lock'))
            self.fixbuttons[-1].setCheckable(True)
            self.qlayout.addWidget(self.fixbuttons[-1], len(self.fixbuttons) + 1, 1)
            self.startvalues.append(QLineEdit(''))
            self.qlayout.addWidget(self.startvalues[-1], len(self.startvalues) + 1, 2)
            self.fitresults.append(QLabel('0'))
            self.qlayout.addWidget(self.fitresults[-1], len(self.fitresults) + 1, 3)

        self.parnames_str = []
        for i, parname in enumerate(fitinfo['par_names']):
            self.parnames[i].show()
            self.parnames[i].setText(parname)
            self.parnames_str.append(parname)

        for i, v in enumerate(fitinfo['startvalues']):
            self.fixbuttons[i].show()
            self.startvalues[i].show()
            self.startvalues[i].setText(str(v))
            self.fitresults[i].show()
            self.fitresults[i].setText('0')
        self.nparameters = len(fitinfo['par_names'])

    def setFitResult(self, results, errors=None):
        for i, res in enumerate(results):
            restxt = '{0:5g}'.format(res)
            if errors is not None:
                restxt = errorformat(res, errors[i])
                # restxt = '{0:.5f}'.format(ufloat(res, errors[i]))

            self.fitresults[i].setText(restxt)
            self.fitresults[i].setToolTip('{0} +/- {1}'.format(res, errors[i]))

    def getStartValues(self):
        startvalues = []
        for i in range(self.nparameters):
            try:
                v = float(self.startvalues[i].text())
                startvalues.append(v)
            except:
                print_error('h5storage fitparameter exception {0} {1}'.format(i, self.startvalues[i].text()), 'error')
                startvalues.append(0)
                self.startvalues[i].setText('0')
        self.getFixValues()
        return startvalues

    def getFixValues(self):
        ''' return an array containing 0 for each fixed parameter'''
        fixarray = []
        for i in range(self.nparameters):
            fixarray.append(int(not self.fixbuttons[i].isChecked()))
        return fixarray


class FittingWidget(QFrame):
    def __init__(self, datasource, parent=None):
        '''

        :datasource:  (a class containing a nextDataProcess Variable)
                    the parent data source
        '''
        super().__init__(parent)
        self.setStyleSheet(
            "FittingWidget {background-color: rgb(180,200,225);color:red; margin:2px; border:3px solid rgb(0, 0, 80);} ")
        self._source = datasource
        self._props = self._source._props
        self._plotitem = self._source._plotitem
        self._label = self._source._label + '/fit'
        self.data = np.array([[0.0021, 0.0041, 0.0051], [30, 55, 70]])
        self._rawdata = np.array([])
        self.nextDataProcess = None
        self.current_function = None

        self.fitMode = QComboBox()  # binning modes
        # self.fitMode.addItem('--None--')
        for key, v in sorted(fitfunctions.items()):
            self.fitMode.addItem(key)
        self.fitMode.currentTextChanged.connect(self.fitfunctionChanged)
        closeBtn = QPushButton('')
        closeBtn.setMaximumWidth(20)
        closeBtn.setIcon(QtGui.QIcon(icon_path + 'terminate.png'))
        closeBtn.clicked.connect(self._clos)
        fitBtn = QPushButton('fit')  # manually trigger fit (e.g. if you have changed startvalues)
        fitBtn.clicked.connect(self._refit)
        self.fitEdit = QLineEdit('np.linspace(1,20,7)')
        self.fitEdit.hide()
        self.parameterBox = FitParameters()

        # Layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.fitMode)

        self.sigmaFilterButton = QPushButton('Enable fit filter:')
        self.sigmaFilterButton.setMinimumWidth(150)
        self.sigmaFilterButton.clicked.connect(self.toggle_sigma_filter)
        hlayout.addWidget(self.sigmaFilterButton)
        self.sigmaFilterBox = QDoubleSpinBox()
        self.sigmaFilterBox.setMaximum(100)
        self.sigmaFilterBox.setMinimum(0)
        self.sigmaFilterBox.setDecimals(2)
        self.sigmaFilterBox.setSingleStep(0.1)
        self.sigmaFilterBox.setSuffix(' σ')
        self.sigmaFilterBox.setValue(0)

        hlayout.addWidget(self.sigmaFilterBox)
        self.sigmaFilterBox.valueChanged.connect(self.update_sigma_filter_QQDoubleSpinBox)

        self.sigmaFilterON = False
        self.sigmaFilterVAL = 0

        self.sigmaFilterLabel = QLabel('')
        hlayout.addWidget(self.sigmaFilterLabel)
        self.update_outlier_number([], 0)

        hlayout.addWidget(fitBtn)
        hlayout.addWidget(closeBtn)

        layout = QVBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(3, 3, 3, 1)
        self.setLayout(layout)
        layout.addLayout(hlayout)
        layout.addWidget(self.parameterBox)
        layout.addWidget(self.fitEdit)

        self._initPlot()

    def _initPlot(self):
        ''' configure the plot of this curve

            :plotitem: (pyqtgraph.PlotItem)  plot environment this plot should be added to
        '''
        plotitem = self._plotitem

        self.plot = pg.PlotDataItem()
        plotitem.addItem(self.plot)

        self._configurePlot()

    def _clos(self):
        ''' close this widget and remove from parent '''
        print('closing fit manager')
        self._source.nextDataProcess = None
        self._plotitem.removeItem(self.plot)
        # self._plotitem.removeItem(self.errorbar)
        self.close()

    def _configurePlot(self):
        # self.plot.setPen(None)
        self.plot.setSymbol(self._props.get(self._label + '/Symbol[o,s,t,d,+]', 'o'))
        col = QColor(self._props.get(self._label + '/Linecolor', int(255 * 256 ** 3 + 255 * 256 + 255)))
        width = self._props.get(self._label + '/linewidth', 1)
        pen = pg.mkPen(col, width=width)
        self.plot.setSymbolPen(None)
        self.plot.setPen(pen)
        # self.plot.setSymbolBrush(pg.mkBrush(QColor(self._props.get(self._label+'/Fillcolor',int(255*256**2+255)))))
        self.plot.setSymbolBrush(None)
        self.plot.setSymbolSize(self._props.get(self._label + '/SymbolSize', 5))
        self.plotData()

    def toggle_sigma_filter(self):
        self.sigmaFilterON = not self.sigmaFilterON
        self.update_sigma_filter_Button()

    def update_sigma_filter_Button(self):
        if self.sigmaFilterON:
            self.sigmaFilterButton.setText(self.sigmaFilterButton.text().replace('Enable', 'Disable'))
        else:
            self.sigmaFilterButton.setText(self.sigmaFilterButton.text().replace('Disable', 'Enable'))
        self.updateData(self._rawdata)

    def update_sigma_filter_QQDoubleSpinBox(self):
        self.sigmaFilterVAL = self.sigmaFilterBox.value()
        if self.sigmaFilterON:
            self.updateData(self._rawdata)

    def update_outlier_number(self, outliers, n, fit_success=True):
        self.sigmaFilterLabel.setText('Removed ' + str(len(outliers)) + ' / ' + str(n))
        str_outliers = 'outliers:'
        if len(outliers) == 0:
            str_outliers += ' --'
        for enum, oulier in enumerate(outliers):
            str_outliers += '\n({0}, {1})'.format(oulier['x'], oulier['y'])
            if len(str_outliers) > 400:
                str_outliers += '\n...'
                break
        if fit_success:
            self.sigmaFilterLabel.setStyleSheet('''QLabel{background-color: rgba(0, 0, 0, 0)}''')
        else:
            self.sigmaFilterLabel.setStyleSheet('''QLabel{background-color: red}''')
        self.sigmaFilterLabel.setToolTip(str_outliers)

    def set_sigma_filter(self, activated=False, sigma=0.):
        self.sigmaFilterON = activated
        self.sigmaFilterVAL = sigma
        self.sigmaFilterBox.setValue(sigma)
        self.update_sigma_filter_Button()

    def fitfunctionChanged(self, function_name):
        ''' called whenever a different fit function is selected'''
        self.parameterBox.setFit(fitfunctions[function_name])
        self.current_function = fitfunctions[function_name]['function']
        self.updateData(self._rawdata)

    def plotData(self):
        # plot the data
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            self.plot.setData(x=self.data[0], y=self.data[1])

            col = QColor(self._props.get(self._label + '/Linecolor', int(256 ** 4 - 1)))
            width = self._props.get(self._label + '/linewidth', 1)
            pen = pg.mkPen(col, width=width)
            x = self.data[0]
            beam = self._props.get(self._label + '/ErorBarBeam', 0.3) * (max(x) - min(x)) / len(x)
        else:
            self.plot.setData(x=[], y=[])

    def _refit(self):
        """fit same data again
        This is usefull if the first fit dit not converge and you want to fit again with changed starting values
        """
        self.updateData(self._rawdata)

    def updateData(self, data):
        """process new data"""
        self._rawdata = data
        self.npars = 3  # number of fit parameters
        self.startvalues = [1, 1]
        self.fitresult = self.startvalues
        updated_outlier_number = False
        outlier_success = True

        if data.shape[0] > 1 and self.current_function is not None:
            # print('Fitting widget got new data: ',data.shape,len(data.shape))
            if len(data.shape) > 1 and data.shape[0] > 1:  # it needs minimum x and y values
                if data.shape[1] >= self.npars:
                    Tx = np.asarray(data[0])
                    Ty = np.asarray(data[1])
                    my_data = odr.Data(Tx, Ty)
                    # my_data = odr.Data(x, y, wd=1./power(sx,2), we=1./power(sy,2)) sx,sy are the errors:
                    mymodel = odr.Model(self.current_function)
                    fitstart = [1, 1]
                    startvalues = self.parameterBox.getStartValues()
                    fix = self.parameterBox.getFixValues()
                    myodr = odr.ODR(my_data, mymodel, beta0=startvalues, ifixb=fix)
                    myodr.set_job(fit_type=2)
                    fit = myodr.run()

                    if self.sigmaFilterON:
                        discrepancies = Ty - self.current_function(fit.beta, Tx)
                        # standard deviation sigma = sqrt(n) * standard error of mean:
                        n = len(Tx)
                        sigma = sem(discrepancies) * np.sqrt(n)
                        outliers = [{'x': x, 'y': y} for x, y in
                                    zip(Tx[np.abs(discrepancies) > self.sigmaFilterVAL * sigma],
                                        Ty[np.abs(discrepancies) > self.sigmaFilterVAL * sigma])]

                        Ty = Ty[np.abs(discrepancies) <= self.sigmaFilterVAL * sigma]
                        Tx = Tx[np.abs(discrepancies) <= self.sigmaFilterVAL * sigma]
                        """
                        if n != len(Tx):
                            print_error('h5storage.py - updateData(): Sigma filter (filter out data <= {2} sigma) '
                                        'reduced data for refit from {0} to {1}, relative discrepancies within '
                                        '[{2}, {3}] sigma.'.format(n, len(Tx), self.sigmaFilterVAL,
                                                                   np.round(min(discrepancies) / sigma, 2),
                                                                   np.round(max(discrepancies) / sigma, 2)),
                                        'warning')
                        """

                        if len(Tx) >= self.npars:
                            my_data = odr.Data(Tx, Ty)
                            myodr = odr.ODR(my_data, mymodel, beta0=startvalues, ifixb=fix)
                            myodr.set_job(fit_type=2)
                            fit = myodr.run()
                        else:
                            outlier_success = False
                            print_error('h5storage.py - updateData(): Filtered out too much data for re-fit by sigma filter.', 'warning')
                        self.update_outlier_number(outliers, n, fit_success=outlier_success)
                        updated_outlier_number = True

                    if not updated_outlier_number:
                        updated_outlier_number = True
                        self.update_outlier_number([], len(Tx))

                    self.fitresult = fit.beta
                    self.parameterBox.setFitResult(fit.beta, fit.sd_beta)
                    xcurve = np.linspace(min(data[0]), max(data[0]), 10000)
                    self.data = np.array([xcurve, self.current_function(fit.beta, xcurve)])
                    #    print "%0.3f" % fit.beta[i], "%0.3f" % fit.sd_beta[i]

            # if  self.binMode.currentIndex() == 0:
            #    self.summarizeData(data)
        else:
            self.data = np.array([])
        self.plotData()

        if not updated_outlier_number:
            self.update_outlier_number([], 0)

    def plotmpl(self, ax):
        ''' plot into matplotlib pyplot axis'''
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            ax.plot(self.data[0], self.data[1])
        if self.nextDataProcess is not None:
            self.nextDataProcess.plotmpl(ax)

    def _close(self):
        pass


class XAxisWidget(QFrame):
    ''' Allows for selection of x axis, serves as basis for yaxis'''

    def __init__(self, label, props, parent=None):
        """
        :label:  (str) label dobles as identifier in the properties
        :props:  (Properties) handler to a bali properties object
        :parent: (QObject)   Qt parent
        """
        super().__init__(parent)
        self._props = props
        self._label = label
        self._layout = QVBoxLayout()
        self._layout.setSpacing(3)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        self.cc = SearchComboBox()
        current = self._props.get(self._label + '/current_selection', '--None--')
        if current != '':
            self.cc.find_or_add(current)
        self.cc.add_if_new('--None--')
        self.cc.currentTextChanged.connect(self.updateProps)
        self._layout.addWidget(QLabel(label))
        self._layout.addWidget(self.cc)
        self.setStyleSheet("SingleProcess\
                {background-color: rgb(210,230,240);color:blue; margin:0px; border:0px solid rgb(220,240,255); }\
                        QPushButton {background-color: rgb(210,230,240);"
                           "color:#000000; margin:0px; border:0px solid rgb(20, 240, 255);} ")
        self.setStyleSheet(
            "XAxisWidget {background-color: rgb(210,230,240);"
            "color:blue; margin:1px; border:2px solid rgb(220, 240, 255); } ")

    def updateKeylist(self, keylist):
        """ update the list of keys in the combo box
        :keylist: (iterable) list of entries to be displayed in Combo box
        """
        combo = self.cc
        current = combo.currentText()
        combo.clear()
        if current != '':
            combo.find_or_add(current)
        current_scan = self._props.get('/BaliBrowser/ExperimentQ/current_scan', [])
        for par in current_scan:
            if par.casefold() != '--NONE--'.casefold() and par != '':  # sometimes it's None and sometimes NONE
                combo.add_if_new(par)

        for key in sorted(keylist):
            if key not in ('--NONE--', '--None--', '', current, *current_scan):
                combo.addItem(key)
        if '_filtercombos' in self.__dict__:
            for filterCombo in self._filtercombos:
                filterCombo.updateKeylist(keylist)

    def updateProps(self, text):
        """update properties when combo box is changed"""
        self._props.set(self._label + '/current_selection', text)


class FilterWidget(QWidget):
    filterchange = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channel = SearchComboBox()
        self.condition = QLineEdit()
        hidebutton = QPushButton('')
        hidebutton.setMaximumWidth(20)
        hidebutton.setIcon(QtGui.QIcon(icon_path + 'terminate.png'))

        self.setToolTip('Condition e.g. >0')
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.channel)
        layout.addWidget(self.condition)
        layout.addWidget(hidebutton)
        self.setLayout(layout)

        hidebutton.clicked.connect(self.disable)
        self.channel.currentTextChanged.connect(self.updated)
        self.condition.textChanged.connect(self.updated)

    def disable(self):
        self.condition.setText('')
        self.hide()

    def updated(self):
        # print('updated')
        self.filterchange.emit(1)

    def updateKeylist(self, keylist):
        ''' update the channels in the combo box'''
        combo = self.channel
        current = combo.currentText()
        combo.clear()
        combo.addItem(current)
        for s in sorted(keylist):
            combo.addItem(s)

    def getFilter(self, data):
        ''' applies a filterstring onto a set of data
        np.nan values will always result as true
        '''
        a = data
        if self.condition.text() != '':
            try:
                filterarr = eval('a' + self.condition.text())
            except:
                filterarr = np.ones(data.shape, dtype=bool)
            filterarr = np.logical_or(np.isnan(a), filterarr)
        else:
            # print('erlse',data.shape)
            filterarr = np.ones(data.shape, dtype=bool)

        return filterarr


class YAxisWidget(XAxisWidget):
    def __init__(self, label, plotter, props, plotitem, parent=None):
        super().__init__(label, props, parent)
        self._props = props
        self._label = label
        self._yscaleLine = QLineEdit('1')
        lay = QHBoxLayout()
        lay.addWidget(QLabel('scale factor:'))
        lay.addWidget(self._yscaleLine)
        self.initExpBox()
        self._layout.addLayout(lay)
        self._yscaleLine.textChanged.connect(self.updateYscale)

        self._filtercombos = []
        for i in range(10):
            filtercombo = FilterWidget()
            filtercombo.filterchange.connect(self.reevaluateData)
            filtercombo.hide()
            self._layout.addWidget(filtercombo)
            self._filtercombos.append(filtercombo)
        self._plotter = plotter
        self._plotitem = plotitem
        self.data = np.array([])
        self.setContextMenuPolicy(Qt.CustomContextMenu);
        self.customContextMenuRequested.connect(self.createContextMenu)
        self.cc.currentTextChanged.connect(self._plotter.update)
        self.nextDataProcess = None  # Next process that should be done with the data
        self.width = self._props.get(self._label + '/linewidth', 1)
        self.col = QColor(self._props.get(self._label + '/color', int(255 * 256 ** 3 + 255 * 256 + 255)))
        self.pen = pg.mkPen(self.col, width=self.width)
        self._initPlot()
        self._yscale = 1

    def updateYscale(self, text):
        # print(text)
        try:
            self._yscale = float(text)
            self.reevaluateData()
        except:
            pass

    def createContextMenu(self, position):
        menu = QMenu()
        filterAction = menu.addAction("Filter")
        binningAction = menu.addAction("Binning")
        fittingAction = menu.addAction("Fitting")
        configureAction = menu.addAction('Configure')
        configureAction.triggered.connect(self.configureProperties)

        action = menu.exec_(self.mapToGlobal(position))
        if action == binningAction:
            self.addBinning()
        if action == fittingAction:
            self.addFitting()
        if action == filterAction:
            for widget in self._filtercombos:
                if widget.isHidden():
                    widget.show()
                    break

    def _initPlot(self):
        ''' configure the plot of this curve

            :plotitem: (pyqtgraph.PlotItem)  plot environment this plot should be added to
        '''
        self.plot = pg.PlotDataItem(pen=None, symbol='o', size=1)
        self._plotitem.addItem(self.plot)
        self.configurePlot()

    def configurePlot(self):
        self.plot.setPen(None)
        self.plot.setSymbol(self._props.get(self._label + '/Symbol', 'o'))
        self.plot.setSymbolPen(self.pen)
        self.plot.setBrush(self.col)
        self.plot.setSymbolBrush(self.col)
        self.plot.setSymbolSize(self._props.get(self._label + '/SymbolSize', 5))
        if self.nextDataProcess is not None:
            self.nextDataProcess._configurePlot()

    def configureProperties(self):
        d = QDialog()
        d.setWindowTitle("Configure " + self._label)
        layout = QVBoxLayout()
        editor = PropEdit('/' + self._props.name + '/' + self._label + '/')
        layout.addWidget(editor)
        d.setLayout(layout)
        d.exec_()
        self.configurePlot()

    def addBinning(self):
        if self.nextDataProcess is None:
            binning = BinningWidget(self)
            self._layout.addWidget(binning)
            self.nextDataProcess = binning
            binning.updateData(self.data)
        elif type(self.nextDataProcess) == FittingWidget:
            binning = BinningWidget(self)
            self._layout.addWidget(binning)
            binning.nextDataProcess = self.nextDataProcess  # process fits after binning
            self.nextDataProcess = binning
            binning.nextDataProcess._source = binning
            binning.updateData(self.data)

    def addFitting(self):
        if self.nextDataProcess is None:
            fitting = FittingWidget(self)
            self._layout.addWidget(fitting)
            self.nextDataProcess = fitting
            fitting.updateData(self.data)
        elif type(self.nextDataProcess) == BinningWidget:
            if self.nextDataProcess.nextDataProcess is None:
                fitting = FittingWidget(self.nextDataProcess)
                self._layout.addWidget(fitting)
                self.nextDataProcess.nextDataProcess = fitting
                fitting.updateData(self.nextDataProcess.data)

    def reevaluateData(self):
        self.updateData(self._xtext)

    def initExpBox(self):
        self.expCheck = QCheckBox()
        self.expBox = QComboBox()
        self.updateExpBox()
        self.expBoxLayout = QHBoxLayout()
        self.expBoxLayout.addWidget(self.expCheck)
        self.expBoxLayout.addWidget(QLabel('Experiment:'))
        self.expBoxLayout.addWidget(self.expBox)
        self._layout.addLayout(self.expBoxLayout)

    def updateExpBox(self):
        expList = self._props.get('/BaliBrowser/OpenWindows', [])
        for exp in expList:
            if exp not in [self.expBox.itemText(i) for i in range(self.expBox.count())]:
                self.expBox.addItem(exp)

    def updateData(self, xtext):
        ytext = self.cc.currentText()
        self.updateExpBox()
        self._xtext = xtext
        newdata = []
        channels = dict([(i, combo.channel.currentText()) for i, combo in \
                         enumerate(self._filtercombos) if not
                         combo.isHidden()])
        filterdata = [[] for i in range(10)]
        # print(channels)
        for line in self._plotter._sData.completeData:
            expCheck = False
            if not self.expCheck.isChecked():
                expCheck = True
            else:
                if os.path.basename(self.expBox.currentText())[:-3] == line['_expName']:
                    expCheck = True
            if expCheck:
                if ytext in line.keys() and xtext in line.keys():
                    newdata.append([line[xtext], line[ytext] * self._yscale])
                    for i, channel in channels.items():
                        if channel in line.keys():
                            filterdata[i].append(line[channel])
                        else:
                            filterdata[i].append(np.nan)
        # generate filter
        totalfilter = np.ones(len(newdata), dtype=bool)
        for i, ch in channels.items():

            if len(filterdata[i]) == len(newdata) and len(filterdata[i]) > 0:
                filt = self._filtercombos[i].getFilter(np.array(filterdata[i]))
                totalfilter = np.logical_and(totalfilter, filt)
                # print(totalfilter,'filter',filt)

            else:
                if len(newdata) > 0:
                    print_error('h5storage.py: filterdata and data have different length', 'warning')

        self.data = np.array(newdata)[totalfilter].T
        if self.nextDataProcess is not None:
            self.nextDataProcess.updateData(self.data)

        # plot the data
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            self.plot.setData(x=self.data[0], y=self.data[1])
            # self.errplot.setData(x=w.dat[0],y=w.dat[1])
        else:
            self.plot.setData(x=[], y=[])

    def plotmpl(self, ax):
        ''' plots into a matplotlib.pyplot figure for exporting
        Args:

            ax: (matplotlib.pyplot.axis)
                figure to be plotted into

        '''
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            ax.plot(self.data[0], self.data[1], 'o')
        if self.nextDataProcess is not None:
            self.nextDataProcess.plotmpl(ax)


class DataSelectorItem(QComboBox):
    ''' select the source of the data to be plotted'''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem('--Summarized--')
        self.currentTextChanged.connect(self.updateSource)

    def updateKeylist(self, keylist):
        self.clear()
        for key in keylist:
            self.addItem(key)

    def updateSource(self, text):
        # print(text)
        pass


class H5DataPlotter(QWidget):
    ''' plot the data acquiread during the measurement process '''
    _viewer = PropertyAttribute('pdfviewer', 'evince')
    _export_file = PropertyAttribute('ExportFile', '/tmp/plot.pdf')

    data_update_checks = 0  # static variable
    got_data = False

    def __init__(self, props, storageGUI, parent=None):
        super().__init__(parent)
        self._props = props
        self._storageGUI = storageGUI
        self._sData = storageGUI.sData
        self.lastUpdate = 0
        self.keylist = ['--None--']
        self.sourcelist = ['--Combined--']

        self.qlayout = QGridLayout()

        # win = pg.GraphicsLayoutWidget(title = "Basic plotting examples")  # deprecated
        win = pg.GraphicsLayoutWidget(title="Basic plotting examples")
        win.setWindowTitle('pyqtgraph example: Plotting')
        self.plotItem = pg.PlotItem()
        win.addItem(self.plotItem)
        self.qlayout.addWidget(win, 0, 0)

        self.dataSelector = DataSelectorItem()

        self.dataSelector.currentTextChanged.connect(self.setDataSource)
        self.setAxes()

        layout_selector = QGridLayout()
        layout_selector.addWidget(self.dataSelector, 0, 1)
        layout_selector.addWidget(self.xAxisSelector, 1, 1)
        for i, widget in enumerate(self.yAxisSelectors):
            layout_selector.addWidget(widget, i + 2, 1)

        self.qlayout.addLayout(layout_selector, 0, 1)

        exportlayout = QHBoxLayout()

        # add plotting functionality
        printbutton = QPushButton('Print')
        exportlayout.addWidget(printbutton)
        printbutton.clicked.connect(self.exportpdf)

        pngbutton = QPushButton('Save as png')
        exportlayout.addWidget(pngbutton)
        pngbutton.clicked.connect(self.exportpng)

        layout_selector.addLayout(exportlayout, len(self.yAxisSelectors) + 3, 1)

        # add ability to load stored settings for the axes
        config_select_button = QPushButton('Select configuration')
        config_select_button.clicked.connect(self.storeConfiguration)
        # self.qlayout.addWidget(config_select_button, 2, 1)

        self.qlayout.setColumnStretch(0, 20)
        self.qlayout.setColumnStretch(1, 10)
        self.setLayout(self.qlayout)

        # start update timer
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.checkForNewData)
        timer.start(100)
        self.timer = timer

    def setAxes(self):
        self.xAxisSelector = XAxisWidget('x-axis', self._props)
        self.yAxisSelectors = []
        for i in range(3):
            self.yAxisSelectors.append(YAxisWidget('y-axis {}'.format(i), self, self._props, self.plotItem))

    def storeConfiguration(self, name='Default'):
        '''
        This function will store the current configuration of the plot widgets
        in its properties. In this way it is possible to load the full settings
        for a certain experiment with few clicks.
        '''
        current_config = {}

        # selectors whose values should be stored:
        # dataSelector, xAxisSelector, yAxisSelectors
        print(self.dataSelector.__dict__)
        current_config['dataSelector'] = self.dataSelector.__dict__
        print(current_config)
        for axisSelector in self.yAxisSelectors:
            print(axisSelector.__dict__)

    def setDataSource(self, text):
        ''' set or change the source of the data'''
        if text == '--Combined--':
            self._sData = self._storageGUI.sData
        else:
            for mgr in self._storageGUI.singleMgrs:
                if mgr.name == text:
                    self._sData = mgr

    def checkForNewData(self):
        ''' check if new data has been added and initiate update of the plots and data selectors'''
        H5DataPlotter.data_update_checks += 1
        if H5DataPlotter.data_update_checks % 1000 == 0:
            H5DataPlotter.data_update_checks = 0
            if not H5DataPlotter.got_data:
                print_error('h5storage.py - checkForNewData(): No data arrived within approx. 50 sec.', 'warning')
            H5DataPlotter.got_data = False
        if self.lastUpdate < self._sData.lastupdate:
            H5DataPlotter.got_data = True
            self.lastUpdate = self._sData.lastupdate
            # print_error('h5storage.py - checkForNewData(): Update available.', 'weak')
            self.updateDatasources()
            self.update()

    def updateDatasources(self):
        ''' check wether new data sources have been added'''
        sourcelist = ['--Combined--']
        for mgr in self._storageGUI.singleMgrs:
            # print(mgr.name)
            sourcelist.append(mgr.name)
        if self.sourcelist != sourcelist:
            self.sourcelist = sourcelist
            self.dataSelector.updateKeylist(sourcelist)

    def update(self):
        ''' update the data selector combo boxes'''
        # select all keys from current dataset that represent float or integer numbers
        if self._sData.completeData != []:
            dat = self._sData.completeData[-1]
            keylist = ['--None--'] + [s for s in dat.keys() if type(dat[s]) == float or type(dat[s]) == int]
        else:
            keylist = ['--None--']

        # update available keys in combo boxes
        if not set(keylist).issubset(self.keylist):
            self.keylist = list(set(self.keylist + keylist))
            for selector in [self.xAxisSelector] + self.yAxisSelectors:
                selector.updateKeylist(self.keylist)
        self._updatePlotData()

    def _updatePlotData(self):
        ''' update the plots '''
        xtext = self.xAxisSelector.cc.currentText()
        for i, w in enumerate(self.yAxisSelectors):
            w.updateData(xtext)

    def gen_export_plot(self):
        fig, ax = plt.subplots()
        for selector in self.yAxisSelectors:
            selector.plotmpl(ax)
        plt.xlabel(self.xAxisSelector.cc.currentText())

        ylabels = []
        for i in self.yAxisSelectors:
            label = i.cc.currentText()
            if label != '--None--':
                ylabels.append(label)
        plt.ylabel(', '.join(ylabels))

        ax.grid()
        return fig, ax

    def exportpng(self):
        ''' generates a nice png from the plotted data'''
        print('exporting png')
        fig, ax = self.gen_export_plot()

        now = datetime.datetime.now()
        now = now.strftime("%Y_%m_%d_%Hh%Mm%Ss")
        savefile = tweezerpath + '/images/' + now + '.png'
        print(savefile)
        fig.savefig(savefile)
        plt.show()

    def exportpdf(self):
        ''' generates a nice pdf from the plotted data'''
        print('exporting pdf')

        fig, ax = self.gen_export_plot()
        now = datetime.datetime.now()
        now = now.strftime("%Y_%m_%d_%Hh%Mm%Ss")
        savefile = tweezerpath + '/images/' + now + '.pdf'
        fig.savefig(savefile)
        call([self._viewer, savefile])


class H5StorageDataMgr(QWidget):
    _h5file = PropertyAttribute('h5file', 'Set Filename')  # filename of the h5 file
    _imagestreams = PropertyAttribute('imagestreams', [])
    _datapath = PropertyAttribute('path', tweezerpath)
    _singleDataStreams = PropertyAttribute('Singledatastreams', [])

    def __init__(self, props, parent=None):
        super().__init__(parent)
        self._props = props

        self.sData = DataSummary('Summary', self._props)  # summarizes the data for max 1 packet per measurement
        self.singleMgrs = []
        self.imageMgrs = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        # directory Button
        dirButton = QPushButton(self._h5file)
        dirButton.clicked.connect(self.selectDatadir)
        layout.addWidget(dirButton)
        self.fileButton = dirButton

        # subscription Editors
        self.hlayout = QHBoxLayout()
        self._subscription_editors = []
        self.initSubscriptionEditors('Channels grouped for each run:', 'Data', '', self.updateGroupManagerSubscriptions)
        self.initSubscriptionEditors('Channels stored individually:', 'Data', 'Single', self.updateSubscriptions)
        self.initSubscriptionEditors('Image channels:', 'Image', '', self.updateImageSubscriptions)
        layout.addLayout(self.hlayout)

        # Measurement name
        self.foldernameLEd = QLineEdit('name your measurement!!')
        # saveBtn = QPushButton('Save')
        # saveBtn.clicked.connect(self.savetoFile)
        # clearBtn = QPushButton('Clear')
        # clearBtn.clicked.connect(self.clear)

        hlayout = QHBoxLayout()
        # hlayout.addWidget(self.foldernameLEd)  Moved into main windows layout
        # hlayout.addWidget(saveBtn)
        # hlayout.addWidget(clearBtn)
        layout.addLayout(hlayout)

        # init data ant image managers
        self.initDataMgr()
        self.updateImageSubscriptions()

        # start update timer
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.updateData)
        timer.start(50)
        self.timer = timer

    def initSubscriptionEditors(self, label, name, prefix='', call=None):
        vlayout = QVBoxLayout()
        vlayout.addWidget(QLabel(label))
        editor = SubscriptionEditor(self._props, name, propprefix=prefix)
        # TODO: this call resets the image queue to 0 length, the signal was also emmitted when
        # the properties changed. Now a property change emmitts a different uncaught signal,
        # is this causing any problems?
        editor.subscriptionsChanged.connect(call)

        self._subscription_editors.append(editor)
        vlayout.addWidget(editor)
        self.hlayout.addLayout(vlayout)

    def initDataMgr(self):
        # init data managers for single stream dumps
        for stream in self._singleDataStreams:
            self.singleMgrs.append(SingleDataSummary(stream, stream, self._props))

    def savetoFile(self):
        text = self.foldernameLEd.text()
        self.sData.savetoFile(self._h5file, folder=text)
        for mgr in self.singleMgrs:
            mgr.savetoFile(self._h5file, folder=text)
        for mgr in self.imageMgrs:
            mgr.savetoFile(self._h5file, folder=text)

    def clear(self):
        self.sData.clear()
        for mgr in self.singleMgrs:
            mgr.clear()

    def updateGroupManagerSubscriptions(self):
        self.sData.initDataq()

    def updateSubscriptions(self):
        self.singleMgrs = []
        for stream in self._props.get('Singledatastreams', []):
            self.singleMgrs.append(SingleDataSummary(stream, stream, self._props))

    def updateImageSubscriptions(self):
        # print('#### Image subscription is updated! ####')
        self.imageMgrs = []
        for stream in self._imagestreams:
            self.imageMgrs.append(SingleImageSummary(stream, stream, self._props))

    def updateData(self):
        self.sData.recvData()
        for mgr in self.singleMgrs:
            mgr.recvData()

        for mgr in self.imageMgrs:
            mgr.recvData()

        ## check for property changes
        changes = self._props.changes()
        if len(changes) > 0 and '/' + self._props.name in changes:
            # print(changes)
            for editor in self._subscription_editors:
                editor.handle_property_changes(changes)

    def selectDatadir(self):
        fname = QFileDialog.getSaveFileName(self, 'Save into file',
                                            self._datapath, "Hdf5 files (*.hdf5 *.h5)")
        if fname[0] != '':
            if fname[0].endswith(('.h5', '.H5', '.hdf5', '.HDF5')):
                self._h5file = fname[0]
            else:
                self._h5file = fname[0] + '.h5'
            self.fileButton.setText(self._h5file)


def main():
    app = QApplication(sys.argv)
    Win = H5StorageGui()
    Win.show()
    app.exec_()


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
