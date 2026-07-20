import h5py 

class Result:
    # def __init__(self, rid):
    #     self.rid = rid
    #     self.scan_param = None
    #     self.scan_vals = []
    #     self.images = []
    #     self.analysis_results = []

    def __init__(
        self,
        rid,
        scan_param,
        scan_vals,
        images,
        analysis_results,
        zips,
        create_hdf5=True,
    ):
        self.rid = rid
        self.scan_param = scan_param
        self.scan_vals = scan_vals
        self.images = images
        self.analysis_results = analysis_results
        self.zips = zips
        if create_hdf5:
            self.create_hdf5(f"experiment_result_{rid}.h5")

    def create_hdf5(self, filename):
        with h5py.File(filename, "w") as f:
            f.attrs["rid"] = self.rid
            f.attrs["scan_param"] = self.scan_param
            f.create_dataset("scan_vals", data=self.scan_vals)
            f.create_dataset("images", data=self.images)
            f.create_dataset("zips", data=self.zips)
            # Assuming analysis_results is a list lists
            for i, result in enumerate(self.analysis_results):
                f.create_dataset(f"analysis_result_{i}", data=result, dtype=float)
