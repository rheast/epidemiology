import atexit, datetime, os, shutil
import numpy as np


class RHEast:
    def __init__(self) -> None:
        self.font = "Times New Roman"
        self.year = 2025  # int(datetime.datetime.now().year)
        self.path = os.path.dirname(os.path.abspath(__file__))
        pycache = os.path.join(self.path, "__pycache__")
        atexit.register(lambda: shutil.rmtree(pycache, ignore_errors=True))
        return

    # y=α*(x-κ)+β Linear
    def lin(self, x, a, b, k=0):
        return a * (x - k) + b

    # y=α*ln(x-κ)+β Logarithmic
    def log(self, x, a, b, k=0):
        return a * np.log(x - k) + b

    # y=α*e^(β*(x-κ)) Exponential
    def exp(self, x, a, b, k=0):
        return a * np.exp(b * (x - k))

    # y=α/(1+e^(β*(x-κ))) Sigmoid
    def sig(self, x, a, b, k=0):
        return a / (1 + np.exp(-b * (x - k)))

    def fit(self, data, run):
        from scipy.optimize import curve_fit  # pip install scipy
        from sklearn.metrics import r2_score  # pip install scikit-learn
        import warnings

        x, y = [np.array(i) for i in data]
        year = [2000, 2000 * 2 - self.year, self.year]
        p0, b0, b1 = [0] * 3, [-np.inf] * 3, [np.inf] * 3
        if run == self.sig:
            b0, b1 = [0, -1, -np.inf], [1, 1, np.inf]
        if x[0] < year[1] or x[0] > year[2]:
            p0, b0, b1 = p0[0:2], b0[0:2], b1[0:2]
        else:
            p0[-1], b0[-1], b1[-1] = year
        sets = {"p0": p0, "bounds": (b0, b1), "maxfev": int(1e5)}
        # print(sets)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            params, _ = curve_fit(run, x, y, **sets)
            r2 = r2_score(y, run(x, *params))

        return r2, params

    def fit_all(self, data, **_):
        run = [self.sig] if data[-1][-1] < 2 else [self.lin, self.log]
        run = run[:1] if len(data[0]) < 3 else run
        matrix = []

        for i in range(len(run)):
            r2, params = self.fit(data, run[i])
            if run[i] == self.lin or r2 > 0.8:
                matrix.append([r2, params, run[i]])

        return sorted(matrix, key=lambda x: x[0], reverse=True)


rheast = RHEast()

if __name__ == "__main__":
    print(rheast.fit_all([[1, 2, 3], [1, 2, 3]]))
