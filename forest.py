import json, os
import numpy as np
import pandas as pd  # pip install pandas openpyxl xlrd
import matplotlib.pyplot as plt  # pip install matplotlib
import matplotlib.lines as mlines
from sklearn.ensemble import RandomForestRegressor  # pip install scikit-learn
from rheast import rheast
from unaids import unaids


class Forest:
    def __init__(self) -> None:
        self.border = "#e7f1ff"
        self.color = [
            *["#1677ff", "#00bfd0", "#00b578", "#ff8f1f"],
            *["#f93a4a", "#ff36c4", "#b136ff", "#7c868d", "#996b59"],
        ]
        self.download = [
            "http://www.unaids.org/sites/default/files/media_asset/HIV_estimates_from_1990-to-present.xlsx",
            "https://www.equaldex.com/api/mapdata.php?info_id=999",
            "https://cdn.amcharts.com/lib/5/geodata/data/countries2.js",
            "https://cdn.who.int/media/docs/default-source/hrp/379607.pdf",
            "https://api.worldbank.org/v2/en/indicator/SP.URB.TOTL.IN.ZS?downloadformat=excel",
            "https://hivfinancial.unaids.org/GARPR16-GAM2024ProgrammeExpenditures.xlsx",
        ]
        plt.rcParams["font.sans-serif"] = "Times New Roman"
        self.run()
        return

    def run(self, matrix={}):
        run = [self.val, self.lgb, self.sex, self.urb, self.fun, self.tar]
        for i in run:
            matrix = i(matrix)
        path = os.path.join(rheast.image, "output.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(matrix))
        for i in ["", "New infection"]:
            if i in unaids.info:
                unaids.info.remove(i)
            self.forest(matrix)
        return

    def forest(self, matrix):
        data, robot, color = [], [], self.color
        info = {i: None for i in unaids.info}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4))
        # print({i: sum(i in matrix[e] for e in matrix) for i in unaids.info})

        for i in matrix:
            arr = {**info, **matrix[i]}
            if len(str(arr).split("None")) == 1:
                data.append(arr)

        x = [[d[i] for i in unaids.info[2:]] for d in data]
        y = [d["Growth rate"] for d in data]
        z = [d["Country"] for d in data]
        model = {"n_estimators": 1000, "random_state": 0}
        if "New infection" in unaids.info:
            model["max_features"] = int(np.sqrt(len(unaids.info)))
        rf_model = RandomForestRegressor(**model)
        rf_model.fit(x, y)

        for i, e in enumerate(unaids.info[2:]):
            a = np.corrcoef([d[e] for d in data], y)[0, 1]
            robot.append("+" if a > 0 else "-")

        imp = [round(i * 100, 2) for i in rf_model.feature_importances_]
        sets = {"Clue": unaids.info[2:], "Importance": imp, "Correlation": robot}
        sets = pd.DataFrame(sets)
        sets = sets.sort_values(by="Importance", ascending=False)
        path = os.path.join(rheast.image, "output.xlsx")
        sets.to_excel(path, index=False)

        mark = {"labels": sets["Clue"], "colors": color[: len(sets)]}
        mark = {**mark, "autopct": lambda i: f"{i:.2f}%" if i > 0 else ""}
        mark = {**mark, "textprops": {"color": "white"}, "labeldistance": 1.1}
        mark = {**mark, "wedgeprops": {"linewidth": 1, "edgecolor": "white"}}
        mark = {**mark, "pctdistance": 0.8, "startangle": 0}
        ax1.pie(sets["Importance"], **mark)
        for text in ax1.texts:
            if text.get_text() in sets["Clue"].values:
                text.set_color("black")
        ax1.axis("equal")

        colors = [color[4] if i == "+" else color[0] for i in sets["Correlation"]]
        ax2.bar(sets["Clue"], sets["Importance"], color=colors, width=0.5, zorder=5)
        ax2.set_ylabel("Importance"), ax2.set_yticks(ax2.get_yticks())
        ax2.set_yticklabels([f"{i:.0f}%" for i in ax2.get_yticks()])
        ax2.tick_params(axis="x", rotation=0)
        ax2.grid(True, color=self.border, linestyle="--")
        for e in ["top", "right"]:
            ax2.spines[e].set_color(self.border)
        for x, y in zip(sets["Clue"], sets["Importance"]):
            ax2.text(x, y + 0.5, f"{y:.2f}%", ha="center", va="bottom", fontsize=9)
        mark = [("Negative correlation", 0), ("Positive correlation", 4)]
        mark = [{"label": a, "markerfacecolor": color[b]} for (a, b) in mark]
        mark = [{"marker": "o", "color": "w", "markersize": 7.5, **i} for i in mark]
        mark = [mlines.Line2D([], [], **i) for i in mark]
        ax2.legend(handles=mark, loc="upper right")
        plt.xticks(rotation=20, ha="right")

        fig.tight_layout()
        path = os.path.join(rheast.image, f"output.{len(unaids.info)}.svg")
        fig.savefig(path, bbox_inches="tight", format="svg")
        return

    def val(self, matrix):
        path = os.path.join(rheast.image, "fig__bar.txt")
        with open(path, "r", encoding="utf-8") as f:
            data = json.loads(f.read().replace("'", '"'))
            for a, b in data:
                a = self.name(a)
                matrix[a] = {"Country": a, "Growth rate": b}
        return matrix

    def lgb(self, matrix):
        data, trans = {}, {}
        path = os.path.join(rheast.file, "LGBT score.json.txt")
        with open(path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())["regions"]
        path = os.path.join(rheast.file, "LGBT score.js.txt")
        with open(path, "r", encoding="utf-8") as f:
            trans = json.loads(f.read().split(" = ")[-1].split(";")[0])
        for i in trans:
            if i in data:
                a, b = trans[i]["country"], data[i]["ei"]
                a = self.name(a)
                if a in matrix:
                    matrix[a]["LGBT score"] = b
        return matrix

    def sex(self, matrix):
        path = os.path.join(rheast.file, "Sex education.txt")
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().split("\n\n")
            for i, e in enumerate(data):
                arr = [n.strip(" ") for n in e.split("\n")[1].split(",")]
                for a in arr:
                    if a in matrix:
                        matrix[a]["Sex education"] = len(data) - i - 1
        return matrix

    def urb(self, matrix):
        path = os.path.join(rheast.file, "Urban population.xls")
        data = pd.read_excel(path, sheet_name=0, engine="xlrd")
        data = data[data.index > 2].iloc[:, [0, -1]]
        for _, (a, b) in data.iterrows():
            a = self.name(a)
            b = float(b)
            if a in matrix and b > 0:
                matrix[a]["Urban population"] = b
        return matrix

    def fun(self, matrix):
        path = os.path.join(rheast.file, "Funding.xlsx")
        data = pd.read_excel(path, sheet_name=0)
        data = data[data.index > 2].iloc[:, [1, 3, 4, -2]]
        robot = {}
        for _, (name, year, sub, val) in data.iterrows():
            if len(sub.split("TOTAL GRAND")) > 1:
                if not name in robot:
                    robot[name] = []
                robot[name].append([year, val])
        for a in robot:
            e = sorted(robot[a], key=lambda i: (i[1], i[0]))
            a = self.name(a)
            if a in matrix:
                matrix[a]["Funding"] = e[-1][-1]
        return matrix

    def tar(self, matrix):
        data = unaids.data[3].iloc[6:, [2, 3, 10, 33, 39, 63, 69]]
        robot, world, before = {}, [], ""
        for _, arr in data.iterrows():
            name, a0, a1, b0, b1, c0, c1 = arr
            if name in unaids.world:
                if name != before:
                    world, before = [], name
                if isinstance(c0, (int, float)) and float(c0) > 0:
                    world = self.num(a0, b0, c0)
                continue
            if not name in robot:
                robot[name] = world
            for e in [[a1, b1, c1], [a0, b0, c0]]:
                a, b, c = e
                if isinstance(a, (int, float)) and float(a) > 0:
                    if isinstance(c, (int, float)) and float(c) > 0:
                        a, b, c = [float(str(x).replace(">", "")) for x in e]
                        robot[name] = self.num(a, b, c)

        data = unaids.data[1].iloc[6:, [2, 48]]
        country, after = [], ""
        for _, arr in data.iterrows():
            name, n0 = arr
            n0 = unaids.num(n0)
            if name in unaids.world:
                if name != before:
                    world, before = [], name
                world.append(n0)
                continue
            else:
                if name != after:
                    country, after = [False] * 5, name
                if n0 is not False:
                    country.append(n0)
                robot[after]["New infection"] = [world[-5:], country[-5:]]

        for name in robot:
            arr = robot[name]["New infection"]
            if type(arr) != type([]):
                continue
            a, b = arr
            b = [x for x in b if x is not False]
            if len(b) < 2:
                b = a
            b = (b[-1] / b[0]) ** (1 / (len(b) - 1)) - 1
            robot[name]["New infection"] = round(b, 5)

        for i in robot:
            if i in matrix:
                matrix[i].update(robot[i])
        return matrix

    def num(self, *number):
        sets, target = {}, ["First", "Second", "Third"]
        for i, e in enumerate(number):
            sets[f"{target[i]} 95 target"] = e
        return sets

    def name(self, name):
        name = name.split(",")[0].split("(")[0].strip(" ").replace("'s ", " ")
        data = [
            ["Cape Verde", "Cabo Verde"],
            ["Cote dIvoire", "Cote d'Ivoire", "Côte d'Ivoire"],
            ["Czech Republic", "Czechia"],
            ["Dominican Republic", "Dominica"],
            ["Lao People Democratic Republic", "Lao PDR"],
            ["Slovakia", "Slovak Republic"],
        ]
        for e in data:
            if name in e:
                name = e[0]
        return name


forest = Forest()
