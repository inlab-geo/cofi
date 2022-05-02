from warnings import warn


def warn_normal_equation():
    # TODO only possible if gtg is full rank (has no zero eigenvalues, or even small eigenvalues)
    warn(
        "You are using linear regression formula solver, please note that this is"
        " only for small scale of data"
    )
