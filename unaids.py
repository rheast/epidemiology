import os, re
import numpy as np
import pandas as pd  # pip install pandas openpyxl


class UNAIDS:
    def __init__(self) -> None:
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.file = os.path.join(self.path, "file")
        self.xlsx = os.path.join(self.file, "HIV_estimates_from_1990-to-present.xlsx")
        self.data, self.time = {}, (2013, 2050)
        return

    def sheet_img(self, data, model, sets):
        image, (a, b), (c, d) = [], (data[0][0], data[0][-1]), self.time
        n = self.num_5(b + 1)
        matrix = [
            {**sets, "time": (a, b), "line": "-"},
            {**sets, "time": (b, d), "line": "--"},
            {**sets, "time": (n, d), "space": 5, "label": sets["name"]},
        ]
        m = list(self.sheet_zip(data, lambda x: x % 5 == 0))
        m = m or [data[0][0:1], data[1][0:1]]
        image.append({**sets, "space": 5, "data": m})
        image.append({**sets, "alpha": 0.5, "s": 20, "data": data})
        for img in matrix:
            if model:
                image.append({**img, "data": self.sheet_num(model, **img)})
        return image

    def sheet_div(self, a, b):
        return [a[0], np.array(a[1]) / np.array(b[1])]

    def sheet_com(self, a, b):
        m = sorted(set(a[0]) & set(b[0]))
        n = lambda x: [m, [v for t, v in zip(x[0], x[1]) if t in m]]
        return [n(a), n(b)]

    def sheet_sum(self, data, time, sets):
        data = data[0][0], sum(y for _, y in data)
        return self.sheet_cut(data, time, sets)

    def sheet_cut(self, data, time, sets):
        a = self.sheet_zip(data, lambda x: x < time)
        b = self.sheet_zip(data, lambda x: x >= time)
        c = self.sheet_zip(data, lambda x: x % 5 == 0)
        matrix = [
            {**sets, "line": "-", "data": a},
            {**sets, "line": "--", "data": b},
            {**sets, "space": 1, "label": sets["name"], "data": c},
        ]
        return matrix

    def sheet_zip(self, data, check):
        return zip(*[[i, j] for i, j in zip(*data) if check(i) and j])

    def sheet_num(self, model, time=[], line=False, space=1, **_):
        _, params, run = model[0]
        start, end = time or self.time
        x = np.arange(start, end + 1, space)
        x = np.linspace(start, end, 100) if line else x
        y = run(x, *params)
        return [x, y]

    def sheet_get(self, name, page=[], time=[], **_):
        sets = {"name": name, "title": "Number"}
        sets["start"] = (time or self.time)[0]
        sets["sheet"], sets["index"] = page or (1, 30)
        return self.sheet(**sets)

    def sheet(self, sheet, name="", index=0, title="", start=0, every=False):
        if not sheet in self.data:
            self.data[sheet] = pd.read_excel(self.xlsx, sheet_name=sheet)
        data = self.data[sheet].copy()
        data = data[data.index >= 7]
        if every:
            return data
        data = data[data.iloc[:, 2] == name]
        data = data.iloc[:, [0, index]]
        data.columns = ["Time", title]
        a, b = data.columns
        data[a] = pd.to_numeric(data[a])
        data = data[data[a] >= start]
        data = data[~data[b].isin(["...", "<500", "<200", "<100"])]
        data = self.num_all(data)
        data = [np.array(data[i]) for i in data.columns]
        return data

    def num(self, value):
        if type(value) == type(""):
            value = re.sub("[, <>]", "", value)
        if isinstance(value, str) and "m" in value.lower():
            return float(value.replace("m", "")) * 1e6
        return float(value)

    def num_lim(self, value, stp=""):
        if value == 0:
            return 0
        units = [(1e6, "m"), (1e3, "k"), (1, ""), (0.01, "%")]
        num, unit = next((n, u) for n, u in units if value >= n or u == "%")
        stp = str(stp) if type(stp) == type(1) else ""
        return f"{value/num:.2f}".rstrip(stp).rstrip(".") + unit

    def num_all(self, data):
        for i in data.columns[1::]:
            data[i] = data[i].apply(self.num)
        return data

    def num_per(self, num, *_):
        return f"{num:.0f}%"

    def num_5(self, num):
        return num + (5 - num % 5) if num % 5 != 0 else num

unaids = UNAIDS()