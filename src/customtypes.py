'''
This class features some helper functions for when creating custom
types for STGP.
'''
def param_init(self, val):
    self.val = val
    # Flag for optimising
    self.hyper_parameter = True


def param_str(self):

    if type(self.val) is str:
        val_str = "'"+self.val+"'"  # Append quotes
    else:
        val_str = str(self.val)  # Otherwise if its a number, dont add quotes

    return self.__class__.__name__ + "(" + val_str + ")"
