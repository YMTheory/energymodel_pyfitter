def _init():
    """
    initialize dicts for:
    1. fitting parameters
    2. gamma sim npe data
    3. gamma sim primary elec/posi data
    """
    global _global_fitpar_dict
    _global_fitpar_dict = {}

    global _global_npe_dict
    _global_npe_dict = {}

    global _global_prm_dict
    _global_prm_dict = {}

def set_fitpar_value(key, val):
    """
    Define a global fitting par dict
    """
    _global_fitpar_dict[key] = val

def set_npe_value(key, val):
    """
    Define a global npe dict
    """
    _global_npe_dict[key] = val


def set_prm_value(key, val):
    """
    Define a global prm dict
    """
    _global_prm_dict[key] = val

    
def get_fitpar_value(key):
    """
    get values from fitpar dict, if the var dose not exist, return default values.
    """
    defValue = 0
    try:
        return _global_fitpar_dict[key]
    except:
        return defValue

def get_npe_value(key):
    """
    get values from fitpar dict, if the var dose not exist, return default values.
    """
    defValue = 0
    try:
        return _global_npe_dict[key]
    except:
        return defValue


def get_prm_value(key):
    """
    get values from fitpar dict, if the var dose not exist, return default values.
    """
    defValue = 0
    try:
        return _global_prm_dict[key]
    except:
        return defValue
