import numpy as np
import pdb


class SummaryWriter ():
    def __init__ (self):
        self.data_list = None
        self.data      = None
        return
    def UpdateData (self, data, label=None):
        self.data   = data
        self.len    = len(data)
        self.arange = list(range(self.len))
        self.label  = label
    def ResetData (self):
        self.data      = None
        self.data_list = None
    def UpdateMultipleData (self, data, label):
        if self.data_list == None:
            self.data_list   = [] if self.data == None else [self.data  ]
            self.len_list    = [] if self.data == None else [self.len   ]
            self.arange_list = [] if self.data == None else [self.arange]
            self.label_list  = [] if self.data == None else [self.label ]
        self.UpdateData (data, label)
        self.data_list.append   (self.data  )
        self.len_list.append    (self.len   )
        self.arange_list.append (self.arange)
        self.label_list.append  (self.label )
    def UpdateDataList(self, data_list):
        self.data_list = data_list

    #def CsvWrite (self, file_path, data):
    def CsvWrite (self, file_path):
        # data : type of list of list
        import csv
        with open (file_path, 'w') as f:
            writer = csv.writer(f)
            #pdb.set_trace ()
            writer.writerows ([self.arange, self.data])
        f.close()
        print (f'CSV File Write to path {file_path}')
        return
    def CsvWriteMultiple (self, file_path):
        # data_list : type of list of list of list
        import csv
        self.arange.insert (0, '')
        idx = 0
        #import pdb
        #pdb.set_trace ()
        for data in self.data_list:
            if isinstance (data, np.ndarray):
                data = data.tolist()
            data.insert (0, self.label_list[idx])
            idx+=1
        with open (file_path, 'w') as f:
            writer = csv.writer (f)
            writer.writerows ([self.arange] + self.data_list)
        f.close ()
        print (f'CSV File Write to path {file_path}')
        return
    #def PltImage (self, file_path, title, xlabel, ylabel, data):
    def PngPlot (self, file_path, title, xlabel, ylabel):
        # data : type of
        import matplotlib.pyplot as plt
        plt.plot (self.arange, self.data)
        plt.title (title    )
        plt.xlabel(xlabel   )
        plt.ylabel(ylabel   )
        plt.savefig (file_path)
        plt.clf ()
        print (f"PNG File Write to path {file_path}")
        return
    def PngPlotMultiple (self, file_path, title, xlabel, ylabel):
        if self.data_list == None:
            self.data_list = [self.data]
            self.len_list  = [self.len ]
            self.arange_list = [self.arange]
        import matplotlib.pyplot as plt
        for idx in range (len(self.data_list)):
            #import pdb
            #pdb.set_trace ()
            plt.plot(self.arange_list[idx], self.data_list[idx], label=self.label_list[idx])
        plt.legend()
        plt.title (title    )
        plt.xlabel(xlabel   )
        plt.ylabel(ylabel   )
        plt.savefig (file_path)
        plt.clf ()
        print (f"PNG File Write to path {file_path}")

    def PngHistPlot (self, file_path, title, xlabel, ylabel, bins):
        import matplotlib.pyplot as plt

        n, bins, patches = plt.hist( self.data, bins=bins, edgecolor='black', linewidth=1.2 )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(file_path)
        plt.clf()
        print(f'PNG Histogram File Write to Path {file_path}')

    def PngMultipleHistPlot (self, file_path, title, xlabel, ylabel, bins, labels):
        import matplotlib.pyplot as plt

        #palette = ['green', 'red', 'blue', 'gray', 'cyan']

        #n, bins, patches = plt.hist( self.data, bins=bins, edgecolor='black', linewidth=1.2 )
        for iters in range (len(self.data_list)):
            #plt.hist( self.data_list[iters], bins=bins, color=palette[iters], label=labels[iters], alpha=0.2, density=True)
            plt.hist( self.data_list[iters], bins=bins, label=labels[iters], alpha=0.2, density=True)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(file_path)
        plt.clf()
        print(f'PNG Histogram File Write to Path {file_path}')
