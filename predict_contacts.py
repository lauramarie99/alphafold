#!/usr/bin/env python
import os
import numpy as np
import scipy
import sys
import json
from alphafold.common import protein, residue_constants
from alphafold.data import pipeline, pipeline_multimer, msa_pairing
from alphafold.data import templates, feature_processing
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model


def load_models(num_recycle, params, use_multimer):
    if use_multimer:
        models = ["model_3_multimer_v3"]
    else:
        models = ["model_3_ptm"]
    model_runner = {}
    for model_name in models:
        if model_name not in model_runner:
            model_config = config.model_config(model_name)
            if not use_multimer:
                model_config.data.eval.num_ensemble = 1
                model_config.data.common.num_recycle = num_recycle
            else:
                model_config.model.num_ensemble_eval = 1
            model_config.model.num_recycle = num_recycle
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
        "template_sequence": np.array(["none".encode()], dtype=object),
        "template_aatype": np.array(templates_aatype)[None],
        "template_confidence_scores": np.array(output_confidence_scores)[None],
        "template_domain_names": np.array(["none".encode()], dtype=object),
        "template_release_date": np.array(["none".encode()], dtype=object),
    }
    return template_features


# predict structure
def predict_structure(
    model_name,
    prefix,
    processed_feature_dict,
    model_runner,
    Ls,
    outdir,
    random_seed=0,
    use_multimer=False,
):
    prediction_result = model_runner.predict(
        processed_feature_dict, random_seed=random_seed
    )
    unrelaxed_protein = protein.from_prediction(
        processed_feature_dict,
        prediction_result,
        remove_leading_feature_dimension=not use_multimer,
    )
    unrelaxed_pdb_lines = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(outdir, f"{prefix}_unrelaxed_{model_name}.pdb")
    with open(unrelaxed_pdb_path, "wt") as fp:
        fp.write(unrelaxed_pdb_lines)
    # write distogram / error estimation for future use
    pred_saving = {}
    pdist = prediction_result["distogram"]["logits"]
    pdist = scipy.special.softmax(pdist, axis=-1)
    prob12 = np.sum(
        pdist[: Ls[0], Ls[0] :, :32], axis=-1
    )  # array of shape N_res, N_res, N_bins (64 by default, 24 A range)
    pred_saving["contact_prob"] = prob12.astype(np.float16)
    pred_saving["predicted_aligned_error"] = prediction_result[
        "predicted_aligned_error"
    ].astype(np.float16)
    pred_saving["plddt"] = prediction_result["plddt"]
    npz_path = os.path.join(outdir, f"{prefix}_scores_{model_name}.npz")
    np.savez_compressed(npz_path, **pred_saving)
    return pred_saving["plddt"].mean(), pred_saving["predicted_aligned_error"].mean()


def process_features_default(a3m_fn, Ls):
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

def process_features_multimer(a3m_fn, Ls, outdir):
    """Builds multimer features from a paired A3M alignment."""

    from alphafold.common import residue_constants

    # --- Query sequence ---
    query_sequence = open(a3m_fn, "r").readlines()[1].strip()
    total_len = sum(Ls)
    assert (
        len(query_sequence) == total_len
    ), f"Query length {len(query_sequence)} != sum(Ls)={total_len}"

    # --- Parse paired MSA ---
    a3m_lines = "".join(open(a3m_fn, "r").readlines())
    msa_obj = pipeline.parsers.parse_a3m(a3m_lines)
    msa_features = pipeline.make_msa_features([msa_obj])

    nrow, Ltot = msa_features["msa"].shape
    # Replace species identifiers with dummy ints (not used here)
    msa_features["msa_species_identifiers"] = np.zeros((nrow,), dtype=np.int32)

    # --- Sequence & template features per chain ---
    all_features_dict = {}
    offset = 0
    for i, L in enumerate(Ls):
        chain_seq = query_sequence[offset : offset + L]
        offset += L
        feats = {
            **pipeline.make_sequence_features(chain_seq, "none", L),
            **mk_mock_template(chain_seq),
        }
        chain_id = chr(65 + i)  # A, B, ...
        feats = pipeline_multimer.convert_monomer_features(feats, chain_id)
        all_features_dict[chain_id] = feats

    # Add assembly features
    all_features_dict = pipeline_multimer.add_assembly_features(all_features_dict)

    # Dummy per-chain MSAs so process_unmerged_features doesn’t crash
    for feats in all_features_dict.values():
        Lc = feats["aatype"].shape[0]
        feats["msa"] = np.zeros((1, Lc), dtype=np.int32)
        feats["deletion_matrix_int"] = np.zeros((1, Lc), dtype=np.int32)
        feats["num_alignments"] = np.array(1, dtype=np.int32)

    feature_processing.process_unmerged_features(all_features_dict)

    # Merge chain features
    np_chains_list = list(all_features_dict.values())
    np_example = msa_pairing.merge_chain_features(
        np_chains_list, pair_msa_sequences=False, max_templates=1
    )

    # Inject the real paired MSA features
    np_example.update(msa_features)

    # --- Finalize first: adds msa_mask, bert_mask, etc. ---
    np_example = feature_processing.process_final(np_example)

    # --- Pad MSA after finalization ---
    np_example = pipeline_multimer.pad_msa(np_example, 764)

    # --- Stub in extra MSA fields (required by multimer model) ---
    nrow, Ltot = np_example["msa"].shape
    num_extra = 1
    np_example["extra_msa"] = np.zeros((num_extra, Ltot), dtype=np.int32)
    np_example["extra_deletion_matrix"] = np.zeros((num_extra, Ltot), dtype=np.int32)
    np_example["extra_msa_mask"] = np.zeros((num_extra, Ltot), dtype=np.float32)
    np_example["extra_cluster_bias_mask"] = np.zeros((num_extra,), dtype=np.float32)

    # Overwrite cluster_bias_mask so it matches padded rows
    np_example["cluster_bias_mask"] = np.zeros((nrow,), dtype=np.float32)
    np_example["num_alignments"] = np.array(nrow, dtype=np.int32)

    # --- Save final MSA for debugging ---
    msa = np_example["msa"]
    with open(f"{outdir}/final_msa.fasta", "w") as f:
        for i, row in enumerate(msa):
            seq = "".join(
                residue_constants.restypes_with_x_and_gap[int(x)] for x in row
            )
            f.write(f">seq{i}\n{seq}\n")

    return np_example


def process_features(a3m_fn, Ls, outdir, use_multimer=False):
    if use_multimer:
        print("Using multimer feature processing")
        return process_features_multimer(a3m_fn, Ls, outdir)
    else:
        return process_features_default(a3m_fn, Ls)


def get_config():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_json", required=True, help="Path to a JSON file describing all runs."
    )
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        cfg = json.load(f)

    # Validate & normalize
    if (
        "items" not in cfg
        or not isinstance(cfg["items"], list)
        or len(cfg["items"]) == 0
    ):
        raise SystemExit("config_json must contain a non-empty 'items' list.")

    # Defaults (all optional)
    cfg.setdefault("n_recycle", 3)
    cfg.setdefault("prefix_template", "{stem}")
    cfg.setdefault("outdir", ".")  # global default = cwd
    cfg.setdefault("params", ".")
    cfg.setdefault("use_multimer", False)

    # Normalize and ensure outdir exists
    cfg["outdir"] = os.path.expanduser(str(cfg["outdir"]))
    os.makedirs(cfg["outdir"], exist_ok=True)

    for i, it in enumerate(cfg["items"]):
        if "a3m" not in it:
            raise SystemExit(f"items[{i}] missing required key 'a3m'.")
        if (
            "Ls" not in it
            or not isinstance(it["Ls"], list)
            or not all(isinstance(x, int) for x in it["Ls"])
        ):
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
    model_runner = load_models(cfg["n_recycle"], cfg["params"], cfg["use_multimer"])
    outdir = cfg["outdir"]

    for it in cfg["items"]:
        a3m = it["a3m"]
        Ls_this = it["Ls"]
        stem = os.path.splitext(os.path.basename(a3m))[0]

        # Compute per-item prefix
        base_prefix = (
            it["prefix"]
            if it["prefix"] is not None
            else cfg["prefix_template"].format(stem=stem)
        )

        # Build features and run models
        feat = process_features(a3m, Ls_this, outdir, cfg["use_multimer"])

        for model_name, AF_model in model_runner.items():
            inputs = AF_model.process_features(feat, random_seed=0)
            lddt, pae = predict_structure(
                model_name,
                base_prefix,
                inputs,
                AF_model,
                Ls_this,
                outdir,
                use_multimer=cfg["use_multimer"],
            )
            print(f"{stem} | {model_name}: pLDDT={lddt:.2f}, mean PAE={pae:.2f}")


if __name__ == "__main__":
    main()