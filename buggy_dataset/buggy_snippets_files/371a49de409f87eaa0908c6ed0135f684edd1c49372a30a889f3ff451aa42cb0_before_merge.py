    def __init__(self):
        super(AboutDialogInfo, self).__init__()

        self.Copyright = "Copyright © 2003 - 2016 Broad Institute, Inc.\nAll rights reserved."

        self.Name = "CellProfiler"

        self.Version = pkg_resources.get_distribution("cellprofiler").version