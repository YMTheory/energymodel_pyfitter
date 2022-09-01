#def _init():
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

global _run_mode
_run_mode = "cpu"

global _fit_gam_flag
_fit_gam_flag = True

global _fit_gam_method
_fit_gam_method = "lsq"

global _fit_B12_flag
_fit_B12_flag = True

global _server_name
_server_name = "local"


def get_fitpar_value(key):
    """
    get value of fitpar with key word key
    """
    defValue = 0
    try:
        return _global_fitpar_dict[key]
    except KeyError:
        return defValue


def get_npe_value(key):
    """
    get value of npe with key word key
    """
    defValue = 0
    try:
        return _global_npe_dict[key]
    except KeyError:
        return defValue


def get_prm_value(key):
    """
    get value of prm with key word key
    """
    defValue = 0
    try:
        return _global_prm_dict[key]
    except KeyError:
        return defValue


def get_kB_value(key):
    """
    get value of kB with key word key
    """
    defValue = 0
    try:
        return _global_kB_dict[key]
    except KeyError:
        return defValue


def get_quenchE():
    """
    get quenchE 
    """
    return _quenchE


def set_fitpar_value(key: str, val: float):
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
    global _quenchE
    _quenchE = val


def set_run_mode(mode):
    """
    Set run mode (options: normal, vec, cuda)
    """
    assert (mode == "cpu" or mode
            == "cuda"), "Wrong run_mode inputs, options are only cpu and cuda"
    global _run_mode
    _run_mode = mode


def get_run_mode():
    """
    return the _run_mode
    """
    return _run_mode


def set_fit_gam_flag(val):
    """
    Set fit_gam_flag.
    """
    assert isinstance(val, bool), "Wrong flag type, must be boolean."
    global _fit_gam_flag
    _fit_gam_flag = val


def get_fit_gam_flag():
    return _fit_gam_flag


def set_fit_B12_flag(val):
    """
    Set fit_B12_flag.
    """
    assert isinstance(val, bool), "Wrong flag type, must be boolean."
    global _fit_B12_flag
    _fit_B12_flag = val


def get_fit_B12_flag():
    return _fit_B12_flag


def set_server_name(val):
    """
    Set server name (file directory).
    """
    assert (val == "local" or val == "ihep"), "Wrong server name argument, options are local or ihep"
    global _server_name
    _server_name = val


def get_server_name():
    """
    return server name.
    """
    return _server_name


def set_fit_gam_method(val):
    assert (val == "unbin" or val == "binned"), "Wrong gamma fitting argument, options are unbin or binned."
    global _fit_gam_method
    _fit_gam_method = val


def get_fit_gam_method():
    return _fit_gam_method

def set_fitpar_value_ingroup(keys, vals):
    for k, v in zip(keys, vals):
        set_fitpar_value(k, v)


def set_npe_value_ingroup(keys, vals):
    for k, v in zip(keys, vals):
        set_npe_value(k, v)


def set_prm_value_ingroup(keys, vals):
    for k, v in zip(keys, vals):
        set_prm_value(k, v)


def set_fitpar_value_indict(d):
    for k, v in d.items():
        set_fitpar_value(k, v)


def _print():
    for item in _global_fitpar_dict.items():
        print(item, end=" ")
    print("\n")
