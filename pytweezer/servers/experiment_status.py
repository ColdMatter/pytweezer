import time
from datetime import datetime

from pytweezer.servers import DataClient

name = "experiment_status_reporter"
stream = DataClient(name)
stream.subscribe("Experiment.start")

icons = {"00_Interaction_optical_dipole_trap": ":boom:",
         "triggercam": ":eyes:",
         "00_loading_scheme": ":sparkler:",
         "01_optical_trapping": ":mouse_trap:",
         "02_compensation_helper": ":mag:",
         "03_pumpi_the_pump": ":fuelpump:",
         "odt_helper": ":dart:",
         "SetPiezoMirrorVoltage": ":zap:",
         "compensation_helper": ":battery:",
         "rf_trap_defaults": ":black_circle:",
         "coils_warm_up": ":hotsprings:",
         "tickle_ion": ":joy:"}

experiment_counts = {}
n_tot = 0
msgmm = r"**Summary of the last 100 experiments** "
msgmm += "\n \n"

while True:
    while stream.has_new_data():
        msg = stream.recv()
        if msg != None:
            n_tot += 1
            metadata = msg[1]

            name = metadata["_expName"]
            if type(name) is list:
                name = name[0]

            icon = "▒"
            if name in icons.keys():
                icon = icons[name]

            msgmm += icon

            if name in experiment_counts.keys():
                experiment_counts[name] += 1
            else:
                experiment_counts[name] = 1

            ## POSTING FOR EVERY EXPERIMENT MIGHT BE TOO VERBOSE
            #msgmm = "{} Currently running `{}`".format(icon, metadata["_expName"])
            #attachmentmm = {}
            #text = ""
            #for key, value in sorted(metadata.items()):
            #    text += "    {}: {} \n".format(key, value)
            #attachmentmm["text"] = text
            #attachmentmm["title"] = "Experiment configuration"
            #attachmentmm["color"] = "ADD8E6"

            if n_tot == 100:
                sorted_experiment_counts = sorted(experiment_counts.items(), key=lambda x:x[1])
                attachmentmm = {}
                text = ""
                for exp, counts in sorted_experiment_counts[::-1]:
                    text += "    {}: {} \n".format(exp, counts)
                attachmentmm["text"] = text
                attachmentmm["title"] = "Summarized experiment counts"
                attachmentmm["color"] = "ADD8E6"

                # reset things
                msgmm = r"**Summary of the last 100 experiments** "
                msgmm += "\n \n"

                n_tot = 0
                experiment_counts = {}


        time.sleep(0.1)