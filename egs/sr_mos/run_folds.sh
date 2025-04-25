#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=2      # stage to start
stop_stage=4 # stage to stop (changed to include merging)
# fold=0 # This will now be set in the loop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=8      # number of parallel jobs in feature extraction
seed=1337

conf=conf/ssl-mos-wav2vec2-dualcrit_freeze.yaml

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
test_sets="dev" # Keep this as a single string for the find command later
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
model_averaging="False"
use_stacking="False"
meta_model_checkpoint=""

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

mkdir -p "data"

# Calculate the base experiment name (without fold suffix)
if [ -z "${tag}" ]; then
    base_expname="$(basename "${conf%.*}")-${seed}"
else
    base_expname="${tag}-${seed}"
fi
echo "Base experiment name: ${base_expname}"


# Define the folds to iterate through
folds=(0 1 2 3 4)

# Loop through each fold for training and inference stages
for fold in "${folds[@]}"; do
    # Skip this loop if stages 2 or 3 are not in the requested range
    if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 2 ]; then
        echo "====================================="
        echo "Processing fold: $fold"
        echo "====================================="

        # --- Update expname and expdir to include the current fold using the base name ---
        expname="${base_expname}-fold${fold}"
        expdir=exp/${expname}
        # --- End of expname and expdir update ---

        mkdir -p "${expdir}" # Ensure fold-specific expdir exists

        if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
            echo "Stage 2: Network training for fold $fold"
            if [ "${n_gpus}" -gt 1 ]; then
                echo "Not Implemented yet."
                # train="python -m seq2seq_vc.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
            else
                train="train.py"
            fi
            echo "Training start. See the progress via ${expdir}/train.log."
            # Use the fold-specific data paths
            ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
                ${train} \
                    --config "${conf}" \
                    --train-csv-path "data/fold${fold}/train.csv" \
                    --dev-csv-path "data/fold${fold}/dev.csv" \
                    --outdir "${expdir}" \
                    --resume "${resume}" \
                    --verbose "${verbose}" \
                    --seed "${seed}"
            echo "Successfully finished training for fold $fold."
        fi

        if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
            echo "Stage 3: Inference for fold $fold"
            # shellcheck disable=SC2012

            # Determine the output directory for this fold's inference
            current_outdir="${expdir}/results"
            if [ "${use_stacking}" = "True" ]; then
                # Update meta_model_checkpoint to be fold-specific if not explicitly set
                # If meta_model_checkpoint is provided as an absolute path, keep it
                if [[ -z "${meta_model_checkpoint}" || "${meta_model_checkpoint}" != /* ]]; then
                     meta_model_checkpoint="${expdir}/meta_model.pkl"
                fi
                current_outdir+="/stacking-model"
            elif [ "${model_averaging}" = "True" ]; then
                 current_outdir+="/model-averaging"
            else
                # Update checkpoint to be fold-specific if not explicitly set
                 # If checkpoint is provided as an absolute path, keep it
                if [[ -z "${checkpoint}" || "${checkpoint}" != /* ]]; then
                    checkpoint="${expdir}/checkpoint-best.pkl"
                fi
                 current_outdir+="/$(basename "${checkpoint}" .pkl)"
            fi

            for name in ${test_sets}; do
                # Ensure the output directory for this test set and fold exists
                test_set_outdir="${current_outdir}/${name}"
                [ ! -e "${test_set_outdir}" ] && mkdir -p "${test_set_outdir}"

                [ "${n_gpus}" -gt 1 ] && n_gpus=1
                echo "Inference start. See the progress via ${test_set_outdir}/inference.log."
                ${cuda_cmd} --gpu "${n_gpus}" "${test_set_outdir}/inference.log" \
                    inference.py \
                        --config "${expdir}/config.yml" \
                        --csv-path "data/fold${fold}/${name}.csv" \
                        --checkpoint "${checkpoint}" \
                        --outdir "${test_set_outdir}" \
                        --model-averaging "${model_averaging}" \
                        --use-stacking "${use_stacking}" \
                        --meta-model-checkpoint "${meta_model_checkpoint}" \
                        --verbose "${verbose}"
                echo "Successfully finished inference of ${name} set for fold $fold."
                # Output results.csv path after inference
                echo "Results for fold $fold, test_set $name saved to ${test_set_outdir}/results.csv"
                grep "UTT" "${test_set_outdir}/inference.log" || true # Use true to prevent pipefail if grep finds nothing
            done
            echo "Successfully finished inference for fold $fold."
        fi
    fi # End check for stages 2 or 3
done # End of fold loop


# Stage 4: Merging results
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "====================================="
    # Use the base_expname to identify the specific experiment
    echo "Stage 4: Merging results from all folds for experiment: ${base_expname}"
    echo "====================================="

    # Construct the pattern to find all results.csv files for THIS specific experiment
    # Use the calculated base_expname to ensure only relevant directories are matched
    results_pattern="exp/${base_expname}-fold*/results/${test_sets}/results.csv"
    echo "Searching for results files matching: ${results_pattern}"
    results_files=$(find exp/ -maxdepth 4 -path "${results_pattern}" -type f | sort)

    # Determine the output filename for the merged results
    # Use the base_expname which already includes the seed and tag/conf_base
    merged_results_file="exp/${base_expname}_$(echo "${test_sets}" | tr ' ' '_')_merged.csv"
    echo "Merged results will be saved to: ${merged_results_file}"


    if [ -z "$results_files" ]; then
        echo "No results.csv files found to merge matching pattern: ${results_pattern}"
        echo "Please ensure stages 2 and 3 ran successfully for the relevant folds and check the paths."
    else
        echo "Found the following results files:"
        echo "$results_files"
        echo ""

        # Get the header from the first file and write to the merged file
        first_file=$(echo "$results_files" | head -n 1)
        if [ -f "$first_file" ]; then
            echo "Writing header from $first_file to $merged_results_file"
            head -n 1 "$first_file" > "$merged_results_file"

            # Append the content (excluding header) from all files
            echo "$results_files" | while read -r result_file; do
                if [ -f "$result_file" ]; then # Double-check file exists
                    echo "Appending data from $result_file"
                    tail -n +2 "$result_file" >> "$merged_results_file"
                else
                    echo "Warning: File not found during append: $result_file"
                fi
            done

            echo "Successfully merged results from all folds to $merged_results_file"
        else
            echo "Error: First results file not found: $first_file"
            exit 1 # Exit if the first file isn't found, as we can't get the header
        fi
    fi

    echo "Successfully finished merging stage."
fi