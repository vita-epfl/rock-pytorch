import os
import sys


class HiddenPrints(object):
    """ Hides print called inside this class
    Used to suppress COCOeval printing when not verbose

    From: https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
