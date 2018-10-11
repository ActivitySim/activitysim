import inspect


def func_source_data(func):
    """
    Return data about a function source, including file name,
    line number, and source code.

    Parameters
    ----------
    func : object
        May be anything support by the inspect module, such as a function,
        method, or class.

    Returns
    -------
    filename : str
    lineno : int
        The line number on which the function starts.
    source : str

    """
    filename = inspect.getsourcefile(func)
    lineno = inspect.getsourcelines(func)[1]
    source = inspect.getsource(func)

    return filename, lineno, source
