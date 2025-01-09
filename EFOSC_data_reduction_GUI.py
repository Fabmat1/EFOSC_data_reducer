import json
import os
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.ttk as ttk
from data_reduction import data_reduction
from PIL import ImageTk, Image
import sv_ttk

class DataReductionGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        if os.path.isfile("./.previousselection.json"):
            with open("./.previousselection.json", "r") as jsonfile:
                self.variabledict = json.load(jsonfile)
        else:
            self.variabledict = {
                "unshiftedflats": [],
                "shiftedflats": [],
                "biases": [],
                "scienceframes": [],
                "comparisonframes": [],
                "compdivision": 3,
                "sciencedivision": 3,
                "output_dict": "./output",
                "outputfile_path": "./EFOSC.csv",
                "comparisonparams": [],
                "debugimages": False,
                "c_cov": 100,
                "s_cov": 0.05,
                "q_cov": 2e-5,
                "cub_cov": 2.5e-10,
                "c_size": 100,
                "s_size": 50,
                "q_size": 100,
                "cub_size": 100,
                "sampleamt": "500000",
                "accept_param": "1.1"
            }

        super().__init__()
        self.title("EFOSC Data Reducer")
        try:
            if os.name == "nt":
                self.iconbitmap("favicon.ico")
            else:
                imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
                self.tk.call('wm', 'iconphoto', self._w, imgicon)
        except Exception as e:
            print(e)

        # Set scaling factor for high-DPI displays
        # scale_factor = 2.0
        # self.tk.call("tk", "scaling", scale_factor)

        self.geometry("650x200+250+250")
        # if os.name == 'nt':
        #     self.state('zoomed')
        # elif os.name == "posix":
        #     self.attributes('-zoomed', True)

        mainframe = ttk.Frame(self)
        superframe = ttk.Frame(mainframe)
        # Shifted, not shifted
        frameframe = ttk.Frame(superframe)
        scienceframe = ttk.Frame(frameframe)
        sciencelabel = ttk.Label(scienceframe, text="Select the science frames:")
        sciencelabel.pack()
        scienceselectedfilelabel = ttk.Label(scienceframe, text="", foreground="green")
        sciencebtn = ttk.Button(scienceframe, text="Select Files", command=lambda: self.fileselection(scienceframe, scienceselectedfilelabel, "scienceframes"))
        sciencebtn.pack()
        scienceframe.grid(row=2, column=0, padx=10, pady=10)
        frameframe.grid(row=0, column=0)
        controlframe = ttk.Frame(superframe)
        outputpathframe = ttk.Frame(controlframe)
        outputlabel = ttk.Label(outputpathframe, text="Output path: ")
        output_path_var = tk.StringVar(value=self.variabledict["output_dict"])
        output_path_var.trace_add("write", lambda a, b, c: self.set_entry("output_dict", output_path_var.get()))
        output_path = ttk.Entry(outputpathframe, validate="focusout", textvariable=output_path_var)
        outputlabel.grid(row=0, column=0, sticky="w")
        output_path.grid(row=0, column=1, sticky="e")
        outputpathframe.grid(row=0, column=0, sticky="w")

        outputfileframe = ttk.Frame(controlframe)
        outputfilelabel = ttk.Label(outputfileframe, text="Output File: ")
        output_file_var = tk.StringVar(value=self.variabledict["outputfile_path"])
        output_file_var.trace_add("write", lambda a, b, c: self.set_entry("outputfile_path", output_file_var.get()))
        output_file = ttk.Entry(outputfileframe, validate="focusout", textvariable=output_file_var)
        outputfilelabel.grid(row=0, column=0, sticky="w")
        output_file.grid(row=0, column=1, sticky="e")
        outputfileframe.grid(row=1, column=0, sticky="w")

        divframe = ttk.Frame(controlframe)
        plotvar = tk.IntVar(value=0)
        coaddcheck = ttk.Checkbutton(divframe, text="Show Debug Plots", variable=plotvar)
        coaddcheck.grid(row=3, column=0, columnspan=1)
        divframe.grid(row=4, column=0, sticky="w")

        testbtn = ttk.Button(mainframe, text="Reduce Data", command=lambda:
        data_reduction(
            self.variabledict["scienceframes"],
            self.variabledict["outputfile_path"],
            self.variabledict["output_dict"],
            show_debug_plot=True if plotvar.get() == 1 else False,
        ))

        controlframe.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

        # ttk.Separator(mainframe, orient=tk.VERTICAL).grid(column=1, row=0, rowspan=2, sticky='ns')
        # ttk.Separator(mainframe, orient=tk.HORIZONTAL).grid(column=0, row=0, columnspan=2, sticky='ew')
        superframe.grid(row=0, column=0)
        testbtn.grid(row=1, column=0, sticky="se")
        mainframe.pack(fill=tk.BOTH, expand=1)

    def printstuff(self):
        print(f"""unshiftedflats:{self.variabledict["unshiftedflats"]}
shiftedflats:{self.variabledict["shiftedflats"]}
biases:{self.variabledict["biases"]}
scienceframes:{self.variabledict["scienceframes"]}
comparisonframes:{self.variabledict["comparisonframes"]}
comparisonparams:{self.variabledict["comparisonparams"]}""")

    def fileselection(self, wintitle, label, key):
        storevar = fd.askopenfilenames(parent=self, title=wintitle)
        self.variabledict[key] = storevar
        with open(".previousselection.json", "w") as jsonfile:
            json.dump(self.variabledict, jsonfile)
        label.config(text=f"Found {len(storevar)} Files!")

    def set_entry(self, key, value):
        self.variabledict[key] = value
        with open(".previousselection.json", "w") as jsonfile:
            json.dump(self.variabledict, jsonfile)


if __name__ == "__main__":
    root = DataReductionGUI()
    sv_ttk.set_theme("light")
    root.mainloop()
