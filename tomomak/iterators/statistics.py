import numpy as np


def rms(solution, real_solution, *args, **kwargs):
    """Returns a normalized root mean square error

    Args:
        solution(ndarray): supposed solution.
        real_solution(ndarray): known solution.
        *args, **kwargs: not used, but needed to be here in order to work with Solver properly.

    Returns:
        float: normalized RMS.

    """
    res = solution - real_solution
    res = np.square(res)
    res = np.sum(res)
    tmp = np.sum(np.square(solution))
    if tmp != 0:
        return np.sqrt(res / tmp)
    else:
        return 0


def rn(solution, real_solution, *args, **kwargs):
    """Residual norm.

    Args:
        solution(ndarray): supposed solution.
        real_solution(ndarray): known solution.
        *args, **kwargs: not used, but needed to be here in order to work with Solver properly.

    Returns:
        float: residual norm

    """
    norm = np.subtract(solution, real_solution)
    norm = np.square(norm)
    return np.sqrt(np.sum(norm))


def chi_sc(solution, real_solution, *args, **kwargs):
    """Chi^2 statistics.

    Note, that fo usage in feasibility method chi^2 should be divided by number of detectors.
    Args:
        solution(ndarray): supposed solution.
        real_solution(ndarray): known solution.
        *args, **kwargs: not used, but needed to be here in order to work with Solver properly.

    Returns:
        float: chi^2.

    """
    chi = np.subtract(solution, real_solution)
    chi = np.multiply(chi, chi)
    chi = np.divide(chi, real_solution, out=np.zeros_like(chi), where=real_solution!=0)
    return np.sum(chi)


def corr_coef(model, solution, old_solution, *args, **kwargs):
    """Returns correlation coefficient, used for stopping criterion

    See Craciunescu et al., Nucl. Instr. and Meth. in Phys. Res. A595 2008 623-630.

    Args:
        model(tomomak.Model): used model.
        solution(ndarray): supposed solution.
        old_solution(ndarray): supposed_solution at a previous iteration.

    Returns:
        float: correlation coefficient.
    """
    det_num = model.detector_signal.shape[0]
    det_num2 = det_num**2
    f_s = np.sum(old_solution)
    f_new_s = np.sum(solution)
    corr= det_num2 * np.sum(np.multiply(solution, old_solution))
    corr = corr - f_s * f_new_s
    divider = det_num2 * np.sum(np.multiply(solution, solution))
    tmp = f_new_s**2
    divider = np.sqrt(divider - tmp)
    corr = corr / divider
    divider = det_num2 * np.sum(np.multiply(old_solution, old_solution))
    tmp = f_s**2
    divider = np.sqrt(divider - tmp)
    return corr / divider

def convergence(solution, old_solution, *args, **kwargs):
    """Returns d(solution) / solution.

    Args:
        solution(ndarray): supposed solution.
        old_solution(ndarray): solution at previous step/

    Returns:
        float: ds/s

    """
    return np.sum(np.abs(solution - old_solution)) / np.abs(np.sum(solution))