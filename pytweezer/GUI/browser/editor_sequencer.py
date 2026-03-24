import importlib.util
import inspect

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qsci import QsciLexerPython
from PyQt5.Qsci import QsciScintilla
import pyqtgraph

from pytweezer.servers import Properties

class CodeEditor(QsciScintilla):

    """This widget implements a simple code editor with syntax highlighting and a button that
        pops up whenever the code is changed."""

    def __init__(self, parent=None, filename=None):
        QsciScintilla.__init__(self, parent)
        self.setMinimumSize(800,600)
        # attributes
        self.filename = filename

        # set the editor font
        font = QtGui.QFont()
        font.setFamily('Inconsolata-g for Powerline')
        font.setPointSize(12)
        font.setFixedPitch(True)
        self.setMarginsFont(font)
        self.setFont(font)

        # indentation settings
        self.setTabWidth(4)
        self.setIndentationsUseTabs(False)
        self.setAutoIndent(True)

        # autocomplete settings
        self.setAutoCompletionSource(QsciScintilla.AcsAll)
        self.setAutoCompletionThreshold(3)

        # margin to show line numbers
        self.setMarginWidth(0, '000')
        self.setMarginType(0, QsciScintilla.NumberMargin)

        # set and configure python lexer
        lexer = QsciLexerPython()
        lexer.setFont(font)
        self.setLexer(lexer)

        # load the file, if given
        if filename is not None:
            with open(filename) as f:
                self.setText(f.read())

    def saveFile(self):
        """Saves the current text to file.

        """
        text = self.text()
        with open(self.filename, mode='w') as f:
            f.write(text)


class ExperimentSequencer(QtWidgets.QWidget):

    """The `ExperimentSequencer` widget visualizes an experimental sequence by plotting the state of each Adwin output
       in use over time."""

    def __init__(self, parent=None, filename=None):
        QtWidgets.QWidget.__init__(self, parent)
        # subscribe to properties
        self.props=Properties('Sequencer')

        # create a graphics window and put it as a wdiget into a gridlayout
        self.graphicsWin = pyqtgraph.GraphicsLayoutWidget()
        self.graphicsWin.setBackground('w')
        self.qlayout = QtWidgets.QGridLayout()
        self.setLayout(self.qlayout)
        self.qlayout.addWidget(self.graphicsWin)

    def plotSequence(self, sequence):
        """Plots an experimental sequence. The input format is defined by the output of `ExperimentParser`.

        :sequence: dict
            Keys are channel names. Contains subdicts 'time' and 'value'.

        """
        #reset graphics window
        self.graphicsWin.clear()
        plots = []
        tmin = 10e3
        tmax = 0
        # create a line plot for each channel
        for ch_name in sorted(sequence.keys()):
            time = sequence[ch_name]['time']
            value = sequence[ch_name]['value']
            if max(time) > tmax:
                tmax = max(time)
            if min(time) < tmin:
                tmin = min(time)
            # convert boolean values to numerals for plotting
            for i in range(len(value)):
                v = value[i]
                if type(v) == bool:
                    if v:
                        value[i] = 1
                    else:
                        value[i] = 0

        # the second loop is necessary to be separated, because we want to access tmin and tmax
        for ch_name in sorted(sequence.keys()):
            time = sequence[ch_name]['time']
            value = sequence[ch_name]['value']

            lplot = self.graphicsWin.addPlot(title=ch_name)

            # stepMode allows correct plotting of logic steps (no interpolation)
            time.append(tmax)
            # TODO: find a better way to handle undefined values at beginning and end!
            if min(time) > tmin:
                time.insert(0, tmin)
                value.insert(0, 0)
            lc=self.props.get('linecolors',[[200,0,0],[0,0,200]])
            qualitative_colors=[pyqtgraph.mkColor(c[0],c[1],c[2]) for c in lc]
            if 'TTL' in ch_name:
                c = qualitative_colors[0]
            else:
                c = qualitative_colors[1]
            pen0 = pyqtgraph.mkPen(c, width=1)
            lplot.plot(time, value, stepMode=True, pen=pen0)

            # link all plots X axis to the first one, see stackoverflow.com/questions/44612914
            # this allows for synchronized zooming and scrolling
            if len(plots) > 0:
                lplot.setXLink(plots[0])
            
            self.graphicsWin.nextRow()
            plots.append(lplot)

class ExperimentParser(object):

    """Parser for a `pytweezer.experiment`. It simulates the run of the experiment and outputs a table that
    contains channels, times and the channels states."""

    def __init__(self, filename=None):
        self.filename = filename
        
        # build the experiment and make it accessible as self.experiment
        self.buildExperiment()

    def buildExperiment(self):
        """Builds the experiment that lives in `self.filename`. This method can be used every time the file
           changed.

        """
        fn = self.filename

        # taken from https://stackoverflow.com/questions/67631
        spec = importlib.util.spec_from_file_location(fn, fn)
        experiment_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment_module)

        name, experiment = inspect.getmembers(experiment_module,inspect.isclass)[-1]
        experiment = experiment(simulate_cmd_stack=True)
        experiment.build()
        self.experiment = experiment

    def simulateExperiment(self):
        """ Simulates a run of `self.experiment`,  and returns the command stack. The command stack is
        reorganised by channels in a dict."""
        self.experiment.start_run()
        cmd_stack = self.experiment.simulated_cmd_stack
        self.experiment.simulated_cmd_stack = []
        
        # transfer to dictionary, with channelnames as keys
        cmd_dict = {}
        for cmd in cmd_stack:
            t, ch, val = cmd
            if ch not in cmd_dict:
                cmd_dict[ch] = {'time':[], 'value':[]}
            cmd_dict[ch]['time'].append(t)
            cmd_dict[ch]['value'].append(val)
        return cmd_dict

class CodeEditorParser(QtWidgets.QMainWindow):

    """This main window unites a code editor, parser and a visualisation of the experiment code."""

    def __init__(self, parent=None, filename=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        # attribute definitions
        self.filename = filename
        self.parser = ExperimentParser(filename)

        # create/name/configure the main window
        self.setWindowTitle("BaLi Experiment Editor")
        self.setGeometry(300,300,1280,800)

        # add a DockWidget that contains the editor
        self.createDocks()
        self.addButtons()
        
        # initially simulate the sequence and plot it
        #self.simulateExperiment()

    def addButtons(self):
        """
           Buttons implemented:
               - save and simulate
        """

        # save and simulate button
        self.simulateBtn = QtWidgets.QPushButton('Simulate Experiment')
        self.codeLayout = QtWidgets.QGridLayout()
        vspace = [18,2,1]
        hspace = [20,2,1]
        for i in range(3):
            self.codeLayout.setColumnStretch(i, vspace[i])
            self.codeLayout.setRowStretch(i, hspace[i])

        self.CodeEditor.setLayout(self.codeLayout)
        self.codeLayout.addWidget(self.simulateBtn, 2, 1)
        #self.simulateBtn.hide()
        self.simulateBtn.clicked.connect(self.simulateExperiment)

        # only save button
        self.simulateBtn = QtWidgets.QPushButton('Save')
        self.codeLayout.addWidget(self.simulateBtn, 1, 1)
        self.simulateBtn.hide()
        self.simulateBtn.clicked.connect(self.CodeEditor.saveFile)
        self.CodeEditor.textChanged.connect(self.simulateBtn.show)

    def createDocks(self):
        """This method creates a DockArea that contains the `CodeEditor` on the left and the visualizing
           `ExperimentSequencer` on the right."""
        # dock the editor to the left
        self.CodeEditor = CodeEditor(self, self.filename)
        editorDock = QtWidgets.QDockWidget('Editor',self)
        editorDock.setWidget(self.CodeEditor)
        editorDock.show()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,editorDock)

        # dock the sequencer to the right
        self.ExperimentSequencer = ExperimentSequencer(self, self.filename)
        sequencerDock = QtWidgets.QDockWidget('Real Time Sequencer',self)
        sequencerDock.setWidget(self.ExperimentSequencer)
        sequencerDock.show()
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,sequencerDock)

    def simulateExperiment(self):
        """ Simulates the current version of the experiment and passes the command stack to the
            ExperimentSequencer for visualization.
        """
        # save file, rebuild experiment and simulate it
        self.CodeEditor.saveFile()
        self.parser.buildExperiment()
        simulated_stack = self.parser.simulateExperiment()
        
        # pass the simulated command stack to the sequencer for plotting
        self.ExperimentSequencer.plotSequence(simulated_stack)

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Win = CodeEditorParser(filename=sys.argv[1])
    Win.show()
    app.exec_()

if __name__ == '__main__':
    main()
