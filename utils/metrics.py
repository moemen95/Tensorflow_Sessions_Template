class FPSMeter:
    """
    Class to measure frame per second in our networks
    """

    def __init__(self, batch_size):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0
        self.milliseconds = 0.0

        self.batch_size = batch_size

    def reset(self):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0

    def update(self, seconds):
        self.milliseconds += seconds * 1000
        self.frame_count += self.batch_size

        self.frame_per_second = self.frame_count / (self.milliseconds / 1000.0)
        self.f_in_milliseconds = self.milliseconds / self.frame_count

    @property
    def mspf(self):
        return self.f_in_milliseconds

    @property
    def fps(self):
        return self.frame_per_second

    def print_statistics(self):
        print("""
Statistics of the FPSMeter
Frame per second: {:.2f} fps
Milliseconds per frame: {:.2f} ms in one frame
These statistics are calculated based on
{:d} Frames and the whole taken time is {:.4f} Seconds
        """.format(self.frame_per_second, self.f_in_milliseconds, self.frame_count, self.milliseconds / 1000.0))
