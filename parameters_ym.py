def _init():
    """
    initialize fitting parameters in the global scope
    """
    global _global_fitpar_dict
    _global_fitpar_dict = {}

    global _global_npe_dict 
    _global_npe_dict = {}

    global _global_prm_dict
    _global_prm_dict = {}

    global _global_kB_dict
    _global_kB_dict = {}

    global _quenchE


def get_fitpar_value(key):
    """
    get value of fitpar with key word key
    """
    defValue = 0
    try:
        return _global_fitpar_dict[key]
    except:
        return defValue

def get_npe_value(key):
    """
    get value of npe with key word key
    """
    defValue = 0
    try:
        return _global_npe_dict[key]
    except:
        return defValue



def get_prm_value(key):
    """
    get value of prm with key word key
    """
    defValue = 0
    try:
        return _global_prm_dict[key]
    except:
        return defValue



def get_kB_value(key):
    """
    get value of kB with key word key
    """
    defValue = 0
    try:
        return _global_kB_dict[key]
    except:
        return defValue



def get_quenchE():
    """
    get quenchE 
    """
    return _quenchE



def set_fitpar_value(key:str, val:float):
    """
    set value of fitpar dict
    """
    _global_fitpar_dict[key] = val

def set_npe_value(key, val):
    """
    set value of npe dict
    """
    _global_npe_dict[key] = val

def set_prm_value(key, val):
    """
    set value of prm dict
    """
    _global_prm_dict[key] = val


def set_kB_value(key, val):
    """
    set value of kB dict
    """
    _global_kB_dict[key] = val


def set_quenchE(val):
    """
    set quenching curves E
    """
    _quenchE = val



def set_fitpar_value_ingroup(keys, vals):
    for k, v in zip(keys, vals):
        set_fitpar_value(k, v)


def set_npe_value_ingroup(keys, vals):
    for k, v in zip(keys, vals):
        set_npe_value(k, v)


def set_prm_value_ingroup(keys, vals):
    for k, v in zip(keys, vals):
        set_prm_value(k, v)



def _print():
    for item in _global_fitpar_dict.items():
        print(item)
