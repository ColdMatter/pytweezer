from pytweezer.GUI.simple_defaults import FloatManager, BoolManager, ComboManager


class FloatBox(FloatManager):
    ''' Box for handling the attribute argument

    Args:
        argument:(NumberValue)
            points to the argument in the experiment

        unit:(str)
            unit string

        display_multiplier:(float)
            multiplication factor for the unit

    '''
    def __init__(self, props, value, parent=None, **kwargs):
        super().__init__(props, parent, parName=kwargs['name'], **kwargs)

    def updateValue(self, value):
        '''Update the corresponding argument in BaliExperiment

        Args:
            value:(type needs to correspont the to the argument type it represents
        '''
        self.value = value*self.display_multiplier
        if self._props:
            self._props.set(self.parName, value*self.display_multiplier)
            self._props.set('last_set', self.parName+': '+str(value))


class BoolBox(BoolManager):
    ''' GUI for boolean values

    KWArgs:
        parName:(str)
            name of the parameter
        value:(BoolValue)
            reference to the argument class


    '''

    def updateValue(self, value):
        '''Update the corresponding argument in BaliExperiment

        Args:
            value:(type needs to correspont the to the argument type it represents
        '''
        self.value = bool(value)
        if self._props:
            self._props.set(self.parName, bool(value))
            self._props.set('last_set', self.parName+': '+str(value))

    def updateCheckbox(self):
        ''' updates the spin box if properties have changed '''
        val=self.value     #the values in properties are in SI units. Non SI only on disp
        if val != self.isChecked():
            self.setChecked(val)


class ComboBox(ComboManager):
    ''' GUI for boolean values

    KWArgs:
        parName:(str)
            name of the parameter
        argument:(BoolValue)
            reference to the argument class
    '''

    def updateValue(self, value):
        '''Update the corresponding argument in BaliExperiment

        Args:
            value:(type needs to correspont the to the argument type it represents
        '''
        value = self.stringlist[value]

        self.value = value
        if self._props:
            self._props.set(self.parName, value)
            self._props.set('last_set', self.parName+': '+str(value))