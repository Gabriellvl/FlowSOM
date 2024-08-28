import csv
import timeit
import tracemalloc

import numpy as np
from sklearn.metrics.cluster import v_measure_score

from flowsom import FlowSOM
from flowsom.io.read_fcs import read_csv_dataset, read_FCS_numpy

# List of datasets with corresponding columns to use and optional label column index
datasets = [
    ["./data/Nilsson_rare.fcs", range(4, 18)],
    ["./data/Levine_13dim.fcs", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], -1],
    ["./data/Levine_32dim.fcs", range(4, 36), 39],
    ["./data/Samusik_01.fcs", range(8, 47), -1],
    ["./data/Mosmann_rare.fcs", range(6, 21)],
    ["./data/ZYX9/spleen_panelT_annotation.csv", range(6, 18)],
    ["./data/Samusik_all.fcs", range(8, 47), -1],
    ["./data/FlowCAP_WNV.fcs", range(2, 8), -2],
    ["./data/FlowCAP_ND.fcs", range(2, 12), -2],
]

# Different algorithm versions to test
algorithm_versions = [ #"original", "squared", "numba",
                        #"batch",
                        #"batch2_sq", "batch2_sq2","batch2_eucl" ,"batch2",
                        "batch2" ]


def run_benchmark():
    """Runs benchmarks for all datasets and versions, saving the results to CSV."""
    for version in algorithm_versions:
        print(f"Running benchmarks for version: {version}\n")

        results = []
        total_time = 0.0
        total_memory = 0.0

        for dataset_info in datasets:
            dataset_path = dataset_info[0]
            columns_to_use = dataset_info[1]
            label_column = dataset_info[2] if len(dataset_info) > 2 else None

            print(f"Benchmarking file: {dataset_path} with columns {columns_to_use} and label column: {label_column}")

            # Time the execution of the benchmark
            execution_time = timeit.timeit(
                lambda: benchmark_single_dataset(dataset_path, version, columns_to_use, seed=42,
                                                 label_column=label_column),
                number=2
            )

            # Measure peak memory usage
            tracemalloc.start()
            benchmark_single_dataset(dataset_path, version, columns_to_use, seed=42, label_column=label_column)
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Update total time and memory for summation at the end
            total_time += execution_time
            total_memory += peak_memory / 1024 / 1024  # Convert to MiB

            # Run the benchmark to get the clustering results and v-measure score
            v_measure, num_clusters = benchmark_single_dataset(dataset_path, version, columns_to_use, seed=42,
                                                               label_column=label_column)

            # Log the results for this dataset
            results.append([
                dataset_path,
                f"{execution_time:.2f}",
                f"{peak_memory / 1024 / 1024:.2f}",  # Convert to MiB
                num_clusters,
                f"{v_measure:.2f}" if v_measure is not None else "N/A"
            ])

        # Add a row for the total sum at the end
        results.append(["Total", f"{total_time:.2f}", f"{total_memory:.2f}", 0, 0])

        # Save the results to a CSV file
        with open(f"./benchmarks/{version}/benchmark_results.csv", "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Filename", "Time (s)", "Memory (MiB)", "Clusters", "V-Measure"])
            csvwriter.writerows(results)

        print(f"Benchmarking completed for version: {version}\n")


def benchmark_single_dataset(file_path, version, columns_to_use, seed, label_column=None):
    """
    Runs the FlowSOM algorithm on a single dataset and returns v-measure and number of clusters.

    Args:
        file_path (str): Path to the dataset file.
        version (str): The version of the algorithm to run.
        columns_to_use (list | range): Columns to use for clustering.
        seed (int): Seed for reproducibility.
        label_column (int, optional): Index of the true labels column.

    Returns
    -------
        v_measure (float): The v-measure score of the clustering, or None if no label column.
        num_clusters (int): The number of clusters used.
    """
    # Load the dataset based on the file format
    if file_path.endswith(".csv"):
        data_array = read_csv_dataset(file_path)
    elif file_path.endswith(".fcs"):
        data_array = read_FCS_numpy(file_path)

    if label_column is not None:
        # Extract labels and remove NaNs
        true_labels = data_array[:, label_column]
        valid_mask = ~np.isnan(true_labels)
        filtered_data = data_array[valid_mask, :]
        true_labels = data_array[valid_mask, label_column]

        # Adjust label values if necessary
        if 0 not in true_labels:
            true_labels = true_labels - 1
        true_labels = true_labels.astype(np.int32)

        # Determine the number of clusters
        num_clusters = np.unique(true_labels).shape[0]

        # Run FlowSOM with the extracted data
        flowsom_instance = FlowSOM(filtered_data, cols_to_use=columns_to_use, n_clusters=num_clusters, seed=seed,
                                   version=version)
        predicted_labels = flowsom_instance.metacluster_labels

        # Calculate v-measure score
        v_measure = v_measure_score(true_labels, predicted_labels)
        print(f"V-measure score: {v_measure}")
        print(f"Number of clusters: {num_clusters}")

        return v_measure, num_clusters

    else:
        # If no label column, just run FlowSOM and return default cluster count
        flowsom_instance = FlowSOM(data_array, cols_to_use=columns_to_use, n_clusters=10, seed=seed, version=version)
        return None, 10


# Execute the benchmark runner
run_benchmark()
