
# == Function node output types == #

# A pipeline is actually just a special type of list
PipelineStump = type('PipelineStump', (list,), {})


# == Arg types == #

def param_init(self, val):
    self.val = val


def param_str(self):

    if type(self.val) is str:
        val_str = "'"+self.val+"'" # Append quotes
    else:
        val_str = str(self.val) # Otherwise if its a number, dont add quotes

    return self.__class__.__name__ + "("+ val_str + ")"


# Specify all the types we want as name, parameter name pairs
types = [
    ("CType_Selection", "C"),
    ("PercentileType", "percentile"),
    ("ThresholdType", "threshold"),
    ("KType", "n_neighbors"),
    ("CType", "C"),
    ("PenaltyType", "penalty"),
    ("KernelType", "kernel"),
    ("GammaType", "gamma"),
    ("NumEstimatorsType", "n_estimators"),
    ("LossType", "loss"),
    ("AlphaType", "alpha"),
    ("BoosterType", "booster"),
    ("DepthType", "max_depth"),
    ("LRType", "learning_rate"),
    ("IterType", "max_iter"),
]


# Make all the defined types dynamically
for name, param_name in types:
    code = "{0} = type('{0}', (), {{'name':'{1}', '__init__':param_init, '__str__':param_str, '__repr__': param_str}})"\
        .format(name, param_name)
    exec(code)