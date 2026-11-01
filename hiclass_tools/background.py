"""Backgrounds with Omega_m0 derived by CLASS (the analytic Background lives in horndeski_template)."""
import concurrent.futures

from horndeski_template import Background


def background_from_class(w0, wa, cosmo=None):
    """Build the background with Omega_m0 derived by CLASS from `cosmo`."""
    from classy import Class
    from .spectra import class_params_gr

    params = class_params_gr(w0, wa, cosmo)
    params['output'] = ''  # background + thermodynamics only
    params.pop('P_k_max_h/Mpc', None)

    c = Class()
    c.set(params)
    try:
        c.compute()
        return Background(w0=w0, wa=wa, Omega_m0=c.Omega0_m())
    finally:
        c.struct_cleanup()
        c.empty()


def _from_class_worker(w0, wa, cosmo):
    try:
        return w0, wa, background_from_class(w0, wa, cosmo), None
    except Exception as e:
        return w0, wa, None, str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__


def backgrounds_from_class(pairs, cosmo=None, workers=None):
    """background_from_class for many (w0, wa) in parallel -> {(w0, wa): Background}. Failures are reported and skipped."""
    out = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_from_class_worker, w0, wa, cosmo) for w0, wa in pairs]
        for future in concurrent.futures.as_completed(futures):
            w0, wa, bg, error = future.result()
            if error is None:
                out[(w0, wa)] = bg
            else:
                print(f"Background failed for w0={w0:.3g}, wa={wa:.3g}: {error}")
    return out
