#!/usr/bin/env python

import os
import os.path
import pickle

import rdkit.Chem as rkc
import rdkit.Chem.AllChem as rkac

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from faerun import Faerun
import tmap as tm


def prepare_tree_data(df, path, force_minhash=False):
    lf = tm.LSHForest(2048, 128)

    if not os.path.isfile(path) or force_minhash:
        print("Creating TMAP")
        enc = tm.Minhash(d=2048)
        fps = [
            tm.VectorUint(
                rkac.GetMorganFingerprintAsBitVect(
                    rkc.MolFromSmiles(smi), radius=3, nBits=2048
                ).GetOnBits()
            )
            for smi in df["smiles"]
        ]
        lf.batch_add(enc.batch_from_sparse_binary_array(fps))
        lf.index()
        lf.store(path)
    else:
        print("Loading TMAP")
        lf.restore(path)
    return lf


DEFAULT_FIELD_INFO = {
    "is_hidden": False,
    "hidden_inverted": False,
    "label_field": "smiles",
    "point_scale": 2.5,
    "max_point_size": 25,
    "categorical": False,
    "has_legend": True,
    "selected_labels": "SMILES",
    "legend_labels": None,
    "fields": [],
}

TEXT_TEMPLATE = """
<a href="http://tmap.gdb.tools" target="_blank">TMAP</a> of the molecules created with a {} SMILES-based scaffold decorator generative model (<a href="">see manuscript</a>).<br>
The decorator model was trained {} and the five scaffolds are {}.<br>
{}
"""

TEXT_TEMPLATES = {
    "validation": "from the validation set",
    "non-dataset": "not from the training set (at most ECFP6 Tanimoto "
    "<= 0.7 to any scaffold in the training set)",
    "multi-step": "multi-step",
    "single-step": "single-step",
    "drd2_first": "with a set of active DRD2 modulators obtained from "
    "<a href='https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0203-5' "
    "target='_blank'>ExCAPE DB</a>",
    "chembl_first": "with a drug-like subset of ChEMBL",
    "drd2_last": "Activity is determined by a Random Forest target prediction "
    "model trained on active and inactive modulators to the same target.",
    "chembl_last": "The decorator learns synthetic chemistry knowledge, as "
    "it tends to create decorations that are both easily synthesizable\
                <br>and these are joined to the scaffold with a bond that "
    "complies with the <a href='https://pubs.acs.org/doi/10.1021/ci970429i' target='_blank'>"
    "RECAP</a> rules.",
}


def create_tmap(name, lf, df, tree_info):

    layout_path = f"./layouts/{name}.pkl"

    if os.path.exists(layout_path):
        print("Loading TMAP layout")
        with open(layout_path, "rb") as fd:
            layout_data = pickle.load(fd)
    else:
        print("Creating TMAP layout")
        cfg = tm.LayoutConfiguration()
        cfg.k = tree_info["k"]
        cfg.node_size = tree_info["node_size"]
        if "sl_extra_scaling_steps" in tree_info:
            cfg.sl_extra_scaling_steps = tree_info["sl_extra_scaling_steps"]
        layout_data = [
            np.array(v) for v in tm.layout_from_lsh_forest(lf, config=cfg)[:4]
        ]

        with open(layout_path, "wb") as fd:
            pickle.dump(layout_data, fd)

    x, y, s, t = layout_data

    print("Coloring TMAP")

    faerun = Faerun(
        clear_color="#000",
        coords=False,
        view="front",
        impress=TEXT_TEMPLATE.format(*tree_info["legend"]),
    )

    for field, info in tree_info["fields"].items():
        info = {**DEFAULT_FIELD_INFO, **info}
        if len(info["fields"]) > 1:
            c = [df[f] for f in info["fields"]]
            colormap = info["colormap"]
            categorical = info["categorical"]
            selected_labels = info["selected_labels"]
            series_title = info["series_title"]
        else:
            c = [df[field].tolist()]
            colormap = [info["colormap"]]
            categorical = [info["categorical"]]
            selected_labels = [info["selected_labels"]]
            series_title = [info["series_title"]]
        coords = {"x": x, "y": y, "c": c, "labels": df[info["label_field"]].tolist()}
        if info["is_hidden"]:
            if len(info["fields"]) < 1:
                dff = df[field]
                if info["hidden_inverted"]:
                    dff = ~dff
                coords["s"] = [int(v) for v in dff]

        faerun.add_scatter(
            field,
            coords,
            shader="smoothCircle",
            colormap=colormap,
            point_scale=info["point_scale"],
            max_point_size=info["max_point_size"],
            categorical=categorical,
            has_legend=info["has_legend"],
            selected_labels=selected_labels,
            series_title=series_title,
            legend_title=info["legend_title"],
            legend_labels=info["legend_labels"],
        )
    faerun.add_tree(
        tree_info["name"], {"from": s, "to": t}, point_helper=tree_info["point_helper"]
    )
    faerun.plot(template="template_smiles.j2", path=f"./html/{name}")


COLORMAPS = {
    "scaffold": ListedColormap(
        ["#1BAFB1", "#F0588C", "#5C415D", "#FF7824", "#FFFC43"], name="scaffold"
    ),
    "p": LinearSegmentedColormap.from_list("", ["red", "yellow", "green"]),
    "in_validation": ListedColormap(["black", "white"], name="validation"),
    "invalid": ListedColormap(["red", "black"], name="invalid"),
}


def create_drd2_tmap(name, is_validation=False, is_multistep=False, node_size=1 / 50):
    data = pd.read_csv("./datasets/drd2_{}.csv".format(name))
    lsh_forest = prepare_tree_data(data, "./trees/drd2_{}.tmap".format(name))
    os.makedirs("./html/drd2_{}".format(name), exist_ok=True)

    scaffolds = data["scaffold"].drop_duplicates().sort_values().tolist()
    data["scaffold_id"] = 0
    for i, scaffold in enumerate(scaffolds):
        data.loc[data["scaffold"] == scaffold, "scaffold_id"] = i

    scaffold_labels = [
        tuple(sl)
        for sl in data[["scaffold_id", "scaffold"]]
        .drop_duplicates()
        .sort_values("scaffold_id")
        .to_numpy()
        .tolist()
    ]

    fields = {}
    if is_validation:
        fields["in_validation"] = {
            "is_hidden": True,
            "colormap": COLORMAPS["in_validation"],
            "point_scale": 5,
            "max_point_size": 50,
            "categorical": True,
            "series_title": "In validation",
            "legend_title": "In validation",
        }

    fields["scaffold_id"] = {
        "colormap": COLORMAPS["scaffold"],
        "point_scale": 3,
        "max_point_size": 30,
        "categorical": True,
        "series_title": "Scaffolds",
        "legend_title": "Scaffolds",
        "legend_labels": scaffold_labels,
    }
    fields["p"] = {
        "colormap": COLORMAPS["p"],
        "point_scale": 1.5,
        "max_point_size": 15,
        "series_title": "Activities",
        "legend_title": "Activities",
    }

    legend_args = []
    if is_multistep:
        legend_args.append(TEXT_TEMPLATES["multi-step"])
    else:
        legend_args.append(TEXT_TEMPLATES["single-step"])
    legend_args.append(TEXT_TEMPLATES["drd2_first"])
    if is_validation:
        legend_args.append(TEXT_TEMPLATES["validation"])
    else:
        legend_args.append(TEXT_TEMPLATES["non-dataset"])
    legend_args.append(TEXT_TEMPLATES["drd2_last"])

    create_tmap(
        "drd2_{}".format(name),
        lsh_forest,
        data,
        {
            "k": 256,
            "node_size": node_size,
            "name": "drd2_{}".format(name),
            "point_helper": "scaffold_id",
            "legend": legend_args,
            "fields": fields,
        },
    )


def create_chembl_tmap(name, is_validation=False, is_multistep=False):
    data = pd.read_csv("./datasets/chembl_{}.csv".format(name)).sample(10000)
    lsh_forest = prepare_tree_data(
        data, "./trees/chembl_{}.tmap".format(name), force_minhash=True
    )
    os.makedirs("./html/chembl_{}".format(name), exist_ok=True)

    scaffold_labels = [
        tuple(sl)
        for sl in data[["scaffold_id", "scaffold"]]
        .drop_duplicates()
        .sort_values("scaffold_id")
        .to_numpy()
        .tolist()
    ]

    fields = {}
    fields["synthesis_filters"] = {
        "fields": [
            "valid_recap",
            "chembl_allowed_0",
            "chembl_allowed_1",
            "zinc_allowed_0",
            "zinc_allowed_1",
        ],
        "categorical": [True] * 5,
        "selected_labels": ["SMILES"] * 5,
        "colormap": [COLORMAPS["invalid"]] * 5,
        "point_scale": 5,
        "max_point_size": 50,
        "series_title": [
            "RECAP valid",
            "All decs. in TS",
            "At most one dec. out TS",
            "All decs. in ZINC",
            "At most one dec. out ZINC",
        ],
        "legend_title": ["Synthetic information"] * 5,
    }

    if is_validation:
        fields["in_validation"] = {
            "is_hidden": True,
            "colormap": COLORMAPS["in_validation"],
            "point_scale": 5,
            "max_point_size": 50,
            "categorical": True,
            "series_title": "In validation",
            "legend_title": "In validation",
        }

    fields["scaffold_id"] = {
        "colormap": COLORMAPS["scaffold"],
        "point_scale": 2,
        "max_point_size": 20,
        "categorical": True,
        "series_title": "Scaffolds",
        "legend_title": "Scaffolds",
        "legend_labels": scaffold_labels,
    }

    legend_args = []
    if is_multistep:
        legend_args.append(TEXT_TEMPLATES["multi-step"])
    else:
        legend_args.append(TEXT_TEMPLATES["single-step"])
    legend_args.append(TEXT_TEMPLATES["chembl_first"])
    if is_validation:
        legend_args.append(TEXT_TEMPLATES["validation"])
    else:
        legend_args.append(TEXT_TEMPLATES["non-dataset"])
    legend_args.append(TEXT_TEMPLATES["chembl_last"])

    create_tmap(
        lsh_forest,
        data,
        {
            "k": 256,
            "node_size": 1 / 100,
            "name": "chembl_{}".format(name),
            "point_helper": "scaffold_id",
            "legend": legend_args,
            "fields": fields,
        },
        "./html/chembl_{}".format(name),
    )


if __name__ == "__main__":
    create_drd2_tmap("ss_vs", is_validation=True)
    create_drd2_tmap("ms_vs", is_validation=True, is_multistep=True, node_size=1 / 75)
    create_drd2_tmap("ss_nd", node_size=1 / 60)
    create_drd2_tmap("ms_nd", is_multistep=True, node_size=1 / 80)

    # create_chembl_tmap("ss_vs", is_validation=True)
    # create_chembl_tmap("ms_vs", is_validation=True, is_multistep=True)
    # create_chembl_tmap("ss_nd")
    # create_chembl_tmap("ms_nd", is_multistep=True)
