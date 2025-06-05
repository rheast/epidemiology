import os
import numpy as np
import pandas as pd  # pip install pandas openpyxl
import geopandas as gpd  # pip install geopandas matplotlib
import matplotlib.pyplot as plt  # pip install matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
from rheast import rheast
from unaids import unaids


class Predict:
    def __init__(self) -> None:
        self.border = "#e7f1ff"
        self.color = [
            *["#f93a4a", "#1677ff", "#00bfd0", "#00b578", "#b136ff"],
            *["#ff36c4", "#ff8f1f", "#00d4a0", "#7c868d", "#996b59"],
        ]
        self.vlim = {"vmin": -10, "vmax": 10}
        self.cmap = ["#1677ff", "#5dfeb7", "#fff3d9", "#ff8f1f", "#f93a4a"]
        self.cmap = LinearSegmentedColormap.from_list("custom_cmap", self.cmap, N=256)
        plt.rcParams["font.sans-serif"] = "Times New Roman"
        self.run_area()
        return

    def run_area(self, index=0):
        run = {"0": self.run_target, "1": self.run_global, "2": self.run_world}
        if not str(index) in run:
            return False
        matrix, ctrl = run[str(index)]()
        image = self.all(matrix)
        if index > 1:
            data = self.growth(image)
            self.map(data)
            self.bar(data)
        else:
            self.line(image, ctrl)
        return self.run_area(index + 1)

    def run_world(self):
        matrix = set(unaids.sheet(1, every=True)["Unnamed: 2"])
        matrix = [[i, 0, {}] for i in matrix]
        return matrix, {}

    def run_global(self):
        ctrl = {"path": "_global", "y0": (15e6, 50e6), "y3": (1e5, 5e5)}
        matrix = [0, 1, 3, 0, 2, 2, 3, 1, 2]
        matrix = [[unaids.world[i], e] for i, e in enumerate(matrix)]
        matrix = [[*i, {"rule": "sum", "txt": "Global (Sum result)"}] for i in matrix]
        return matrix, ctrl

    def run_target(self):
        ctrl = {"path": "_target", "y1": (0.65, 1)}
        matrix = [
            [(1, 30), ("Estimated people living with HIV", "")],
            [(3, 78), ("People who know their HIV status", "First 95 target")],
            [(3, 83), ("People on antiretroviral treatment", "Second 95 target")],
            [(3, 88), ("People with suppressed viral load", "Third 95 target")],
        ]
        for i, (a, b) in enumerate(matrix):
            matrix[i] = ["Global", 0, {"rule": 95, "page": a, "text": b}]
        return matrix, ctrl

    def growth(self, matrix):
        data = []
        for (_, value), name in matrix:
            number, growth = value, False
            for i in range(len(value)):
                if len(set(value[-(i + 1) :])) > 2:
                    number = value[-(i + 1) :]
                    break
            if len(number) < 1 or name in unaids.world:
                continue
            start, end, year = number[0], number[-1], len(number) - 1
            if start and end and year:
                if abs(end - start) / (1 / year) < 100:
                    continue
                growth = (end / start) ** (1 / year) - 1
            if growth and not np.isnan(growth):
                data.append([name, float(f"{growth*100:.3f}")])
        data = sorted(data, key=lambda i: (i[1], i[0]))
        return data

    def map(self, data):
        countries, values = zip(*data)
        df = pd.DataFrame({"name": countries, "value": values})

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        sm = ScalarMappable(cmap=self.cmap, norm=plt.Normalize(**self.vlim))
        sm.set_array([])
        axp = ax.get_position()
        cbar = fig.add_axes([axp.x1 + 0.02, axp.y0, 0.02, axp.height])
        cbar = fig.colorbar(sm, cax=cbar)
        cbar.set_label("Growth Rate")
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(unaids.num_per))

        sets = {"column": "value", "ax": ax, **self.vlim}
        sets = {**sets, "cmap": self.cmap, "missing_kwds": {"color": "#d1d5dd"}}
        world = os.path.join(rheast.file, "ne_110m_admin_0_countries.zip")
        world = gpd.read_file(world)
        world = world.merge(df, how="left", left_on="NAME", right_on="name")
        world.boundary.plot(ax=ax, linewidth=1, color="black")
        world.plot(**sets)

        path = os.path.join(rheast.image, "fig__map.svg")
        fig.savefig(path, bbox_inches="tight", format="svg")
        return

    def bar(self, data):
        countries, values = zip(*data[-65:][::-1])
        norm = mcolors.Normalize(**self.vlim)
        cmap = [self.cmap(norm(v)) for v in values]

        plt.figure(figsize=(15, 5))
        plt.grid(True, color=self.border, linestyle="--")
        plt.bar(countries, values, color=cmap, zorder=4)
        ax = plt.gca()
        ax.set_yticks(ax.get_yticks())
        [ax.spines[e].set_color(self.border) for e in ["top", "right"]]
        [unaids.num_lim(i, 0) for i in ax.get_yticks()]
        plt.ylabel("Growth Rate")
        plt.xticks(rotation=35, ha="right")
        plt.xlim(-0.5, len(countries) - 0.5)
        plt.tight_layout()
        plt.subplots_adjust()

        path = os.path.join(rheast.image, "fig__bar.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
        path = os.path.join(rheast.image, "fig__bar.svg")
        plt.savefig(path, bbox_inches="tight", format="svg")
        return

    def line(self, image, ctrl={}):
        ax, axes = max([i["ax"] for i in image]), []
        fig = [(15, 5), (15, 11), (15, 17)][ax // 2]
        fig = plt.figure(figsize=fig)
        for i in range(ax + 1):
            i = int(f"{[11,12,22,22,32,32][ax]}{i+1}")
            axes.append(fig.add_subplot(i))
        self.line_draw(image, axes, ctrl)
        self.line_grid(axes, ctrl)
        path = os.path.join(rheast.image, f"fig_{ctrl['path']}.svg")
        fig.savefig(path, bbox_inches="tight", format="svg")
        return

    def line_draw(self, image, axes, ctrl):
        occupy = ctrl.get("oc", {})
        for img in image:
            ax, sets = axes[img["ax"]], {"zorder": 4}
            data = list(img["data"])
            if len(data) == 0:
                continue
            x, y = data
            for i in img:
                if i in ["color", "alpha", "s"]:
                    sets[i] = img[i]
            if "line" in img:
                ax.plot(x, y, linestyle=img["line"], **sets)
            else:
                label = img.get("label", "")
                ax.scatter(x, y, linewidths=0, label=label, **sets)
            if not "space" in img:
                continue
            for a, b in zip(x, y):
                o, j = f"{img['ax']}_{a:.0f}", img.get("range", 0.01)
                occupy[o] = occupy.get(o, [])
                sets = {"zorder": 5, "ha": "center", "textcoords": "offset points"}
                sets = {**sets, "va": "bottom", "xytext": (1, 0), "fontsize": 9}
                if any(i * (1 - j) < b < i * (1 + j) for i in occupy[o]):
                    continue
                ax.annotate(unaids.num_lim(b), (a, b), **sets)
                occupy[o].append(b)
        return

    def line_grid(self, axes, ctrl):
        for i, ax in enumerate(axes):
            ax.grid(True, color=self.border, linestyle="--")
            x, y = unaids.time, ctrl.get(f"y{i}", False)
            for e in ["top", "right"]:
                ax.spines[e].set_color(self.border)
            ax.set_xlim(*x), ax.set_yticks(ax.get_yticks())
            ax.spines["left"].set_position(("data", x[0]))
            y = ax.set_ylim(*y) if y else False
            y = [unaids.num_lim(i, 0) for i in ax.get_yticks()]
            n = "Percentage" if ax.get_yticks()[-1] < 2 else "Number"
            ax.set_yticklabels(y), ax.set_ylabel(f"{n} of people")
            sets = {"facecolor": "white", "shadow": False, "framealpha": 1}
            ax.legend(**sets)
        return

    def all(self, matrix):
        image, robot = [], []
        for i, (name, ax, info) in enumerate(matrix):
            sets = {"name": name, "ax": ax, "range": info.get("range", 0.025)}
            data = self.all_data(sets, info)
            if len(matrix) > len(self.color):
                if len(data) and len(data[0]):
                    image.append([data, name])
                continue
            matrix[i] = data
            sets, data, image = self.all_before(i, sets, info, data, matrix, image)
            image, robot = self.all_after(i, sets, info, data, matrix, image, robot)
        return image

    def all_data(self, sets, info):
        data = unaids.sheet_get(**{**sets, **info})
        for i in [(1, 27), (3, 78)]:
            if len(data) > 1 and len(set(data[1])) > 1:
                break
            data = unaids.sheet_get(**{**sets, **info, "page": i})
        return data

    def all_before(self, i, sets, info, data, matrix, image):
        rule = info.get("rule")
        sets["color"] = info.get("color", self.color[i])
        if rule == 95 and i == 0:
            sets["name"] = info["text"][0]
            line = {"color": self.border, "line": "-", "ax": 1}
            image.append({**line, "data": [unaids.time, [0.95, 0.95]]})
        if rule == 95 and i > 0:
            image += unaids.sheet_img(data, [], sets)
            data = unaids.sheet_div(*unaids.sheet_com(data, matrix[i - 1]))
            sets = {**sets, "ax": 1, "range": 0.01, "name": info["text"][1]}
        if rule == "sum" and i == 0:
            sets = {**sets, "color": self.color[0], "name": "Global (Fit result)"}
            line = list(unaids.sheet_zip(data, lambda x: x % 5 == 0))
            image += unaids.sheet_img(line, (), {**sets, "alpha": 1})
        return sets, data, image

    def all_after(self, i, sets, info, data, matrix, image, robot):
        rule = info.get("rule", False)
        time = (data[0][0], data[0][-1])
        model = rheast.fit_all(data)
        image += unaids.sheet_img(data, model, sets)
        if rule == "sum" and i > 0:
            sets = {"ax": 0, "color": "#4f5eff"}
            robot.append(unaids.sheet_num(model, space=0.25, **sets))
        if rule == "sum" and i == len(matrix) - 1:
            sets = {**sets, "name": info["txt"]}
            data = unaids.sheet_sum(robot, time[1] + 0.5, sets)
            line = list(data[-1]["data"])
            data[-1]["data"] = unaids.sheet_zip(line, lambda x: x >= time[1])
            image = data + image
        if rule == 95:
            sets = {**sets, "ax": 0, "range": 0.025, "name": info["text"][0]}
            data = unaids.sheet_num(model, space=0.25)
            data[1] = data[1] * robot[-1][1] if i > 0 else data[1]
            robot.append(data)
        if rule == 95 and i > 0:
            data = list(unaids.sheet_zip(data, lambda x: x >= time[0]))
            image += unaids.sheet_cut(data, time[1] + 0.75, sets)
        return image, robot


predict = Predict()
