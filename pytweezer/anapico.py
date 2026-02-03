import pyvisa

class AnapicoSYN420:
    def __init__(self, resource_address, resource_manager=None):
        """
        Initializes the Anapico SYN420 driver.
        
        Args:
            resource_address (str): The VISA address (e.g., 'USB0::0x03EB::...').
            resource_manager (pyvisa.ResourceManager, optional): Existing RM instance.
        """
        if resource_manager is None:
            self.rm = pyvisa.ResourceManager()
        else:
            self.rm = resource_manager
            
        self.inst = self.rm.open_resource(resource_address)
        
        # Configure standard termination characters if necessary (often '\n')
        self.inst.write_termination = '\n'
        self.inst.read_termination = '\n'
        

    def write(self, command):
        """Wrapper for SCPI write."""
        self.inst.write(command)

    def read(self):
        """Wrapper for SCPI read."""
        return self.inst.read()

    def query(self, command):
        """Wrapper for SCPI query."""
        return self.inst.query(command)

    @property
    def enabled(self):
        """
        Gets or sets the output status.
        True to Enable (ON), False to Disable (OFF).
        """
        # Reading status isn't in original C#, but good practice to include
        resp = self.query(":OUTP:STAT?")
        return bool(int(resp))

    @enabled.setter
    def enabled(self, value):
        if value:
            self.write(":OUTP:STAT ON")
        else:
            self.write(":OUTP:STAT OFF")

    @property
    def modulation_enabled(self):
        """Gets or sets the Modulation status."""
        resp = self.query(":SOUR:FM:STAT?")
        return bool(int(resp))
    
    @modulation_enabled.setter
    def modulation_enabled(self, value):
        if value:
            self.write(":SOUR:FM:STAT ON")
        else:
            self.write(":SOUR:FM:STAT OFF")

    @property
    def cw_frequency(self):
        """Gets or sets the CW Frequency."""
        return float(self.query(":SOUR:FREQ:CW?"))

    @cw_frequency.setter
    def cw_frequency(self, value):
        # Assumes debug check is handled externally or ignored here
        self.write(f":SOUR:FREQ:CW {value}")

    def close(self):
        """Closes the VISA resource."""
        self.inst.close()


    
    ## WARNING: THE FOLLOWING METHODS WERE WRITTEN BY GEMINI AI BASED ON A C# DRIVER.
    ## THEY HAVE NOT BEEN TESTED WITH THE ACTUAL HARDWARE.
    ## USE WITH CAUTION AND VERIFY FUNCTIONALITY BEFORE DEPLOYMENT.

    # @property
    # def list_sweep_enabled(self):
    #     """
    #     Enables List Sweep mode on Channel 1 (and Fix on others), 
    #     or disables it (Fix on all).
    #     """
    #     # Basic check to see if Channel 1 is in List mode
    #     resp = self.query(":SOUR1:FREQ:MODE?")
    #     return "LIST" in resp.upper()

    # @list_sweep_enabled.setter
    # def list_sweep_enabled(self, value):
    #     if value:
    #         # Logic Correction: The C# loop `i > 0` prevented the `i==0` block from running.
    #         # Implemented here: Channel 1 -> LIST, Others -> FIX.
    #         for i in range(self.number_of_channels):
    #             # Channel index is 1-based in SCPI
    #             channel_str = f":SOUR{i+1}"
                
    #             if i == 0:
    #                 # Primary Channel: Set to List Mode
    #                 self.write(f"{channel_str}:FREQ:MODE LIST")
    #                 self.write(f"{channel_str}:POW:MODE LIST")
    #                 self.write(f"{channel_str}:LIST:COUN 1")
    #             else:
    #                 # Other Channels: Set to Fix Mode
    #                 self.write(f"{channel_str}:FREQ:MODE FIX")
            
    #         # Trigger setup
    #         self.write(":INIT:CONT ON")
    #         self.write(":TRIG:TYPE NORM")
    #         self.write(":TRIG:SOUR EXT")
    #         self.write(":TRIG:SLOP POS")
    #         self.write(":TRIG:OUTP:MODE NORM")
    #     else:
    #         # Disable: Set all channels to FIX
    #         for i in range(self.number_of_channels):
    #             channel_str = f":SOUR{i+1}"
    #             self.write(f"{channel_str}:FREQ:MODE FIX")
    #             self.write(f"{channel_str}:POW:MODE FIX")

    # def write_list(self, data):
    #     """
    #     Writes list data to the device.
        
    #     Args:
    #         data: Can be a single string (writes to current channel) 
    #               or a list of strings (writes to specific channels).
    #     """
    #     if isinstance(data, str):
    #         # Matches C# public void WriteList(string list)
    #         self.write(f":MEM:FILE:LIST:DATA {data}")
            
    #     elif isinstance(data, list):
    #         # Matches C# public void WriteList(string[] chList)
    #         for i in range(self.number_of_channels):
    #             if i < len(data):
    #                 list_str = data[i]
    #                 num_bytes = len(list_str)
    #                 num_digits = len(str(num_bytes))
                    
    #                 # Select channel
    #                 self.write(f":SOUR:SEL {i+1}")
                    
    #                 # Construct SCPI Block Header: #N + Length + Data
    #                 # Example: #3100...data...
    #                 header = f"#{num_digits}{num_bytes}"
    #                 self.write(f":MEM:FILE:LIST:DATA {header}{list_str}")

    # def read_list(self):
    #     """Reads list data from the currently selected channel."""
    #     self.write(":MEM:FILE:LIST:DATA?")
    #     return self.read()

    # def read_channel_list(self):
    #     """Reads list data from all channels."""
    #     ch_list = []
    #     for i in range(self.number_of_channels):
    #         self.write(f":MEM{i+1}:FILE:LIST:DATA?")
    #         ch_list.append(self.read())
    #     return ch_list


# Example Usage
if __name__ == "__main__":
    # Replace with your actual VISA address
    address = "USB0::0x03EB::0xAFFF::321-028100000-0168::INSTR" 
    try:
        driver = AnapicoSYN420(address)
        
        # Enable Output
        driver.enabled = True
        
        # Set CW Frequency
        driver.cw_frequency = 10e9 # 10 GHz
        
    except Exception as e:
        print(f"Error: {e}")