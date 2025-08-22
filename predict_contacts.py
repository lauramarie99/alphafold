#!/usr/bin/env python
import os
import numpy as np
import scipy
import sys
import json
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

# loading model params in memory (only model_3_ptm for now)
# load ptm models to get pairwise alignment error estimation just in case
def load_models(num_recycle, params):
    model_runner = {}
    for model_name in ["model_3_ptm"]:
        if model_name not in model_runner:
            model_config = config.model_config(model_name)
            model_config.data.eval.num_ensemble = 1
            model_config.model.num_recycle = num_recycle
            model_config.data.common.num_recycle = num_recycle
            model_params = data.get_model_haiku_params(
                model_name=model_name, data_dir=params
            )
            model_runner[model_name] = model.RunModel(model_config, model_params)
    return model_runner


# generate fake template features
def mk_mock_template(query_sequence):
    # mock template features
    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []
    for _ in query_sequence:
        templates_all_atom_positions.append(
            np.zeros((templates.residue_constants.atom_type_num, 3))
        )
        templates_all_atom_masks.append(
            np.zeros(templates.residue_constants.atom_type_num)
        )
        output_templates_sequence.append("-")
        output_confidence_scores.append(-1)
    output_templates_sequence = "".join(output_templates_sequence)
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.array(templates_all_atom_positions)[None],
        "template_all_atom_masks": np.array(templates_all_atom_masks)[None],
        "template_sequence": [f"none".encode()],
        "template_aatype": np.array(templates_aatype)[None],
        "template_confidence_scores": np.array(output_confidence_scores)[None],
        "template_domain_names": [f"none".encode()],
        "template_release_date": [f"none".encode()],
    }
    return template_features

# predict structure
def predict_structure(
    model_name, prefix, processed_feature_dict, model_runner, Ls, outdir, random_seed=0
):
    prediction_result = model_runner.predict(
        processed_feature_dict, random_seed=random_seed
    )
    unrelaxed_protein = protein.from_prediction(
        processed_feature_dict, prediction_result
    )
    unrelaxed_pdb_lines = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(outdir, f"{prefix}_unrelaxed_{model_name}.pdb")
    with open(unrelaxed_pdb_path, "wt") as fp:
        fp.write(unrelaxed_pdb_lines)
    # write distogram / error estimation for future use
    pred_saving = {}
    pdist = prediction_result["distogram"]["logits"]
    pdist = scipy.special.softmax(pdist, axis=-1)
    prob12 = np.sum(pdist[:Ls[0],Ls[0]:, :32], axis=-1) # array of shape N_res, N_res, N_bins (64 by default, 24 A range)
    pred_saving["contact_prob"] = prob12.astype(np.float16)
    pred_saving["predicted_aligned_error"] = prediction_result[
        "predicted_aligned_error"
    ].astype(np.float16)
    pred_saving["plddt"] = prediction_result["plddt"]
    npz_path = os.path.join(outdir, f"{prefix}_scores_{model_name}.npz")
    np.savez_compressed(npz_path, **pred_saving)
    return pred_saving["plddt"].mean(), pred_saving["predicted_aligned_error"].mean()


def process_features(a3m_fn, Ls):
    # Read query sequence and number of residues for each chain
    query_sequence = open(a3m_fn, "r").readlines()[1].strip()
    a3m_lines = "".join(open(a3m_fn, "r").readlines())
    msa_obj = pipeline.parsers.parse_a3m(a3m_lines)
    # gather features
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=query_sequence, description="none", num_res=len(query_sequence)
        ),
        **pipeline.make_msa_features(msas=[msa_obj]),
        **mk_mock_template(query_sequence),
    }
    # add big enough number to residue index to indicate chain breaks
    idx_res = feature_dict["residue_index"]
    L_prev = 0
    # Ls: number of residues in each chain
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i :] += 200
        L_prev += L_i
    feature_dict["residue_index"] = idx_res
    return feature_dict


def get_config():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_json", required=True,
        help="Path to a JSON file describing all runs."
    )
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        cfg = json.load(f)

    # Validate & normalize
    if "items" not in cfg or not isinstance(cfg["items"], list) or len(cfg["items"]) == 0:
        raise SystemExit("config_json must contain a non-empty 'items' list.")

    # Defaults (all optional)
    cfg.setdefault("n_recycle", 3)
    cfg.setdefault("prefix_template", "{stem}")
    cfg.setdefault("outdir", ".")   # global default = cwd
    cfg.setdefault("params", ".")

    # Normalize and ensure outdir exists
    cfg["outdir"] = os.path.expanduser(str(cfg["outdir"]))
    os.makedirs(cfg["outdir"], exist_ok=True)

    for i, it in enumerate(cfg["items"]):
        if "a3m" not in it:
            raise SystemExit(f"items[{i}] missing required key 'a3m'.")
        if "Ls" not in it or not isinstance(it["Ls"], list) or not all(isinstance(x, int) for x in it["Ls"]):
            raise SystemExit(f"items[{i}] must include 'Ls' as a list of integers.")
        # Normalize absolute path early (optional)
        it["a3m"] = os.path.expanduser(str(it["a3m"]))
        if not os.path.exists(it["a3m"]):
            raise SystemExit(f"A3M not found: {it['a3m']}")
        # Optional per-item prefix; fall back to template
        it.setdefault("prefix", None)

    return cfg


def main():
    cfg = get_config()
    # Load alphafold models once
    model_runner = load_models(cfg["n_recycle"], cfg["params"])
    outdir = cfg["outdir"]

    for it in cfg["items"]:
        a3m = it["a3m"]
        Ls_this = it["Ls"]
        stem = os.path.splitext(os.path.basename(a3m))[0]

        # Compute per-item prefix
        base_prefix = it["prefix"] if it["prefix"] is not None else cfg["prefix_template"].format(stem=stem)

        # Build features and run models
        feat = process_features(a3m, Ls_this)
        for model_name, AF_model in model_runner.items():
            inputs = AF_model.process_features(feat, random_seed=0)
            lddt, pae = predict_structure(model_name, base_prefix, inputs, AF_model, Ls_this, outdir)
            print(f"{stem} | {model_name}: pLDDT={lddt:.2f}, mean PAE={pae:.2f}")


if __name__ == "__main__":
    main()