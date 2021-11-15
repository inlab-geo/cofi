def cofi_init():
    pass


def cofi_misfit(x, y):
    A, B = 1.0, 100.0
    return (A - x) ** 2 + B * (y - x ** 2) ** 2
