use ndarray::{array, s, Array1, Array2, ArrayBase, Axis, Data, Dim, Ix1, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Enum representing the distance metric to be used.
#[derive(Clone)]
pub enum DistanceMetric {
    Euclidean,
    // Other metrics can be added as needed.
}

/// Struct containing parameters for the balanced K-Means algorithm.
pub struct BalancedKMeans {
    pub n_clusters: usize,
    pub n_iters: usize,
    pub metric: DistanceMetric,
}

/// Adjusts the centers of small clusters towards data points from large clusters.
///
/// Returns `true` if any centers were adjusted.
pub fn adjust_centers(
    centroids: &mut Array2<f32>,
    labels: &Array1<usize>,
    cluster_sizes: &Array1<usize>,
    dataset: &Array2<f32>,
    threshold: f32,
    average_size: usize,
) -> bool {
    // Identify large cluster data points.
    let large_clusters: Vec<usize> = cluster_sizes
        .iter()
        .enumerate()
        .filter_map(|(i, &size)| {
            if size > (average_size as f32 * threshold).ceil() as usize {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if large_clusters.is_empty() {
        // No large clusters to adjust towards.
        return false;
    }

    // Collect indices of data points in large clusters.
    let large_cluster_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(idx, &label)| {
            if large_clusters.contains(&label) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    if large_cluster_indices.is_empty() {
        // No data points in large clusters.
        return false;
    }

    let num_large = large_cluster_indices.len();

    // Atomic flag to indicate if any center was adjusted.
    let adjusted = AtomicBool::new(false);

    // Iterate over all clusters in parallel.
    centroids
        .axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut centroid)| {
            let size = cluster_sizes[i];
            if size > (average_size as f32 * threshold).ceil() as usize {
                // Not a small cluster; no adjustment needed.
                return;
            }

            // Create a thread-local RNG using a seed (for reproducibility or random seeding).
            let mut rng = StdRng::from_entropy(); // or use a fixed seed if reproducibility is required

            // Select a random data point from large clusters.
            let selected_idx = large_cluster_indices[rng.gen_range(0..num_large)];
            let data_point = dataset.row(selected_idx);

            // Compute the weighted average.
            // Weight of the current center for the weighted average.
            let wc = size as f32;
            let wc = if wc > 7.0 { 7.0 } else { wc };
            let wd = 1.0;

            // Update centroid: (wc * centroid + wd * data_point) / (wc + wd)
            let new_centroid = (&centroid * wc + &data_point * wd) / (wc + wd);
            if !new_centroid.iter().all(|&v| v.is_nan()) {
                centroid.assign(&new_centroid); // Use assign to update the view
                adjusted.store(true, Ordering::Relaxed);
            }
        });

    adjusted.load(Ordering::Relaxed)
}

impl BalancedKMeans {
    /// Creates a new BalancedKMeans instance with the given number of clusters and iterations.
    pub fn new(n_clusters: usize, n_iters: usize) -> Self {
        BalancedKMeans {
            n_clusters,
            n_iters,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Fits the model to the data and returns the cluster centroids.
    pub fn fit(&self, X: &Array2<f32>) -> Array2<f32> {
        let n_samples = X.len_of(Axis(0));
        let n_features = X.len_of(Axis(1));

        // Determine the number of mesoclusters.
        let n_mesoclusters = (self.n_clusters as f32).sqrt().round() as usize;

        // Step 1: Initialize mesocluster centroids randomly.
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..n_samples).choose_multiple(&mut rng, n_mesoclusters);
        let mut meso_centroids = Array2::<f32>::zeros((n_mesoclusters, n_features));
        for (i, &idx) in indices.iter().enumerate() {
            meso_centroids.row_mut(i).assign(&X.row(idx));
        }

        // Step 2: Assign points to mesoclusters.
        let meso_labels = self.assign_labels(X, &meso_centroids);

        // Step 3: Calculate the number of fine clusters per mesocluster.
        let meso_counts = self.count_labels(&meso_labels, n_mesoclusters);
        let fine_clusters_per_meso = self.arrange_fine_clusters(&meso_counts);

        // Step 4: For each mesocluster, cluster into fine clusters.
        let mut fine_centroids_list = Vec::new();
        let mut fine_labels = Array1::<usize>::zeros(n_samples);
        let mut cluster_start = 0;
        for i in 0..n_mesoclusters {
            let fine_clusters = fine_clusters_per_meso[i];
            if fine_clusters == 0 {
                continue;
            }

            // Extract points belonging to the current mesocluster.
            let indices: Vec<usize> = meso_labels
                .iter()
                .enumerate()
                .filter(|&(_, &label)| label == i)
                .map(|(idx, _)| idx)
                .collect();
            let meso_data = X.select(Axis(0), &indices);

            // Cluster the mesocluster data into fine clusters.
            let kmeans = KMeans {
                n_clusters: fine_clusters,
                n_iters: self.n_iters,
                metric: self.metric.clone(),
            };
            let (fine_centroids, labels) = kmeans.fit_predict(&meso_data);

            // Update labels to reflect global cluster indices.
            for (j, &idx) in indices.iter().enumerate() {
                fine_labels[idx] = labels[j] + cluster_start;
            }

            // Collect fine centroids as Array1<f32>
            for centroid in fine_centroids.axis_iter(Axis(0)) {
                fine_centroids_list.push(centroid.to_owned());
            }

            cluster_start += fine_clusters;
        }

        // Combine all fine centroids into a 2D array (clusters x features).
        let fine_centroids = ndarray::stack(
            Axis(0),
            &fine_centroids_list
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // Step 5: Run a few K-Means iterations on the whole dataset with all centroids.
        let (mut final_centroids, _) = self.kmeans_with_initial_centroids(
            X,
            self.n_clusters,
            &fine_centroids,
            std::cmp::max(self.n_iters / 10, 2),
        );

        // Step 6: Perform balancing EM iterations to ensure cluster balance.
        // Initialize cluster sizes based on the final labels.
        let mut final_labels = self.predict(X, &final_centroids);
        let mut cluster_sizes = self.count_labels(&final_labels, self.n_clusters);
        self.balancing_em_iters(
            X,
            &mut final_centroids,
            &mut final_labels,
            &mut cluster_sizes,
            self.n_iters,
            0.25, // balancing_threshold (as per C++ default)
            2,    // balancing_pullback (as per C++ default)
        );

        final_centroids
    }

    /// Predicts the closest cluster each sample in X belongs to.
    pub fn predict(&self, X: &Array2<f32>, centroids: &Array2<f32>) -> Array1<usize> {
        self.assign_labels(X, centroids)
    }

    /// Computes clustering and predicts cluster indices for each sample.
    pub fn fit_predict(&self, X: &Array2<f32>) -> (Array2<f32>, Array1<usize>) {
        let centroids = self.fit(X);
        let labels = self.predict(X, &centroids);
        (centroids, labels)
    }

    /// Assigns labels to each sample based on the closest centroid.
    fn assign_labels<S1, S2>(
        &self,
        X: &ArrayBase<S1, Ix2>,
        centroids: &ArrayBase<S2, Ix2>,
    ) -> Array1<usize>
    where
        S1: Data<Elem = f32> + Sync,
        S2: Data<Elem = f32> + Sync,
    {
        let n_samples = X.len_of(Axis(0));
        let n_centroids = centroids.len_of(Axis(0));
        let labels: Vec<usize> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let x = X.row(i);
                let mut min_dist = std::f32::INFINITY;
                let mut min_j = 0;
                for j in 0..n_centroids {
                    let c = centroids.row(j);
                    let dist = self.compute_distance(&x, &c);
                    if dist < min_dist {
                        min_dist = dist;
                        min_j = j;
                    }
                }
                min_j
            })
            .collect();

        Array1::from(labels)
    }

    /// Computes the distance between two points based on the selected metric.
    fn compute_distance<S1, S2>(&self, x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> f32
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = f32>,
    {
        match self.metric {
            DistanceMetric::Euclidean => (&*x - &*y).mapv(|v| v * v).sum(),
        }
    }

    /// Runs K-Means with initial centroids provided.
    fn kmeans_with_initial_centroids(
        &self,
        X: &Array2<f32>,
        n_clusters: usize,
        initial_centroids: &Array2<f32>,
        n_iters: usize,
    ) -> (Array2<f32>, Array1<usize>) {
        let n_samples = X.len_of(Axis(0));
        let n_features = X.len_of(Axis(1));
        let mut centroids = initial_centroids.clone();
        let mut labels = Array1::<usize>::zeros(n_samples);

        for _ in 0..n_iters {
            // Assign labels.
            labels = self.assign_labels(X, &centroids);

            // Compute new centroids.
            let mut new_centroids = Array2::<f32>::zeros((n_clusters, n_features));
            let mut counts = vec![0; n_clusters];
            X.axis_iter(Axis(0))
                .zip(labels.iter())
                .for_each(|(x, &label)| {
                    new_centroids
                        .row_mut(label)
                        .zip_mut_with(&x, |a, &b| *a += b);
                    counts[label] += 1;
                });
            new_centroids
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(i, mut c)| {
                    if counts[i] > 0 {
                        c.mapv_inplace(|v| v / counts[i] as f32);
                    }
                });
            centroids = new_centroids;
        }

        (centroids, labels)
    }

    /// Counts the number of occurrences of each label.
    fn count_labels(&self, labels: &Array1<usize>, n_labels: usize) -> Vec<usize> {
        let mut counts = vec![0; n_labels];
        labels.iter().for_each(|&label| {
            counts[label] += 1;
        });
        counts
    }

    /// Arranges the number of fine clusters per mesocluster proportionally.
    fn arrange_fine_clusters(&self, meso_counts: &[usize]) -> Vec<usize> {
        let n_mesoclusters = meso_counts.len();
        let total_points: usize = meso_counts.iter().sum();
        let mut fine_clusters_per_meso = vec![0; n_mesoclusters];
        let mut clusters_remaining = self.n_clusters;
        let mut points_remaining = total_points;
        let mut nonempty_mesos_remaining = meso_counts.iter().filter(|&&c| c > 0).count();

        for i in 0..n_mesoclusters {
            if i < n_mesoclusters - 1 {
                if meso_counts[i] == 0 {
                    fine_clusters_per_meso[i] = 0;
                } else {
                    nonempty_mesos_remaining -= 1;
                    let s = ((clusters_remaining as f32) * (meso_counts[i] as f32)
                        / (points_remaining as f32))
                        .round() as usize;
                    fine_clusters_per_meso[i] =
                        s.min(clusters_remaining - nonempty_mesos_remaining).max(1);
                }
            } else {
                fine_clusters_per_meso[i] = clusters_remaining;
            }
            clusters_remaining -= fine_clusters_per_meso[i];
            points_remaining -= meso_counts[i];
        }

        fine_clusters_per_meso
    }

    /// Performs balancing EM iterations to ensure cluster balance.
    ///
    /// This function iteratively adjusts cluster centers to balance cluster sizes.
    fn balancing_em_iters(
        &self,
        X: &Array2<f32>,
        centroids: &mut Array2<f32>,
        labels: &mut Array1<usize>,
        cluster_sizes: &mut Vec<usize>,
        n_iters: usize,
        balancing_threshold: f32,
        balancing_pullback: usize,
    ) {
        let mut balancing_counter = balancing_pullback;
        let mut total_iters = 0;

        while total_iters < n_iters {
            // Balancing step - move the centers around to equalize cluster sizes
            if total_iters > 0 {
                let adjusted = adjust_centers(
                    centroids,
                    labels,
                    &Array1::from(cluster_sizes.clone()),
                    X,
                    balancing_threshold,
                    cluster_sizes.iter().sum::<usize>() / self.n_clusters,
                );

                if adjusted {
                    if balancing_counter >= balancing_pullback {
                        balancing_counter -= balancing_pullback;
                        // If centers were adjusted, increment total_iters to allow more iterations
                        // This mimics the "pullback" logic in the C++ code
                        if total_iters < n_iters {
                            total_iters += 1;
                        }
                    } else {
                        balancing_counter += 1;
                    }
                }
            }

            // E-step: Assign labels based on current centroids
            *labels = self.assign_labels(X, centroids);

            // M-step: Recompute cluster centers and sizes
            let new_cluster_sizes = self.count_labels(labels, self.n_clusters);
            let mut new_centroids = Array2::<f32>::zeros((self.n_clusters, X.len_of(Axis(1))));
            X.axis_iter(Axis(0))
                .zip(labels.iter())
                .for_each(|(x, &label)| {
                    new_centroids
                        .row_mut(label)
                        .zip_mut_with(&x, |a, &b| *a += b);
                });
            new_centroids
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(i, mut c)| {
                    if new_cluster_sizes[i] > 0 {
                        c.mapv_inplace(|v| v / new_cluster_sizes[i] as f32);
                    }
                });
            *centroids = new_centroids;
            *cluster_sizes = new_cluster_sizes;

            total_iters += 1;
        }
    }
}

/// Standard K-Means clustering algorithm.
pub struct KMeans {
    pub n_clusters: usize,
    pub n_iters: usize,
    pub metric: DistanceMetric,
}

impl KMeans {
    /// Creates a new KMeans instance with the given number of clusters and iterations.
    pub fn new(n_clusters: usize, n_iters: usize) -> Self {
        KMeans {
            n_clusters,
            n_iters,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Fits the model to the data and returns the cluster centroids and labels.
    pub fn fit_predict(&self, X: &Array2<f32>) -> (Array2<f32>, Array1<usize>) {
        let n_samples = X.len_of(Axis(0));
        let n_features = X.len_of(Axis(1));
        let mut rng = thread_rng();

        // Initialize centroids randomly.
        let indices: Vec<usize> = (0..n_samples).choose_multiple(&mut rng, self.n_clusters);
        let mut centroids = Array2::<f32>::zeros((self.n_clusters, n_features));
        for (i, &idx) in indices.iter().enumerate() {
            centroids.row_mut(i).assign(&X.row(idx));
        }

        let mut labels = Array1::<usize>::zeros(n_samples);

        for _ in 0..self.n_iters {
            // Assign labels.
            labels = self.assign_labels(X, &centroids);

            // Compute new centroids.
            let mut new_centroids = Array2::<f32>::zeros((self.n_clusters, n_features));
            let mut counts = vec![0; self.n_clusters];
            X.axis_iter(Axis(0))
                .zip(labels.iter())
                .for_each(|(x, &label)| {
                    new_centroids
                        .row_mut(label)
                        .zip_mut_with(&x, |a, &b| *a += b);
                    counts[label] += 1;
                });
            new_centroids
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(i, mut c)| {
                    if counts[i] > 0 {
                        c.mapv_inplace(|v| v / counts[i] as f32);
                    }
                });
            centroids = new_centroids;
        }

        (centroids, labels)
    }

    /// Assigns labels to each sample based on the closest centroid.
    fn assign_labels<S1, S2>(
        &self,
        X: &ArrayBase<S1, Ix2>,
        centroids: &ArrayBase<S2, Ix2>,
    ) -> Array1<usize>
    where
        S1: Data<Elem = f32> + Sync,
        S2: Data<Elem = f32> + Sync,
    {
        let n_samples = X.len_of(Axis(0));
        let n_centroids = centroids.len_of(Axis(0));
        let labels: Vec<usize> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let x = X.row(i);
                let mut min_dist = std::f32::INFINITY;
                let mut min_j = 0;
                for j in 0..n_centroids {
                    let c = centroids.row(j);
                    let dist = self.compute_distance(&x, &c);
                    if dist < min_dist {
                        min_dist = dist;
                        min_j = j;
                    }
                }
                min_j
            })
            .collect();

        Array1::from(labels)
    }

    /// Computes the distance between two points based on the selected metric.
    fn compute_distance<S1, S2>(&self, x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> f32
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = f32>,
    {
        match self.metric {
            DistanceMetric::Euclidean => (&*x - &*y).mapv(|v| v * v).sum(),
        }
    }
}

// fn main() {
//     // Parameters for data generation
//     let n_samples = 100000; // Number of data points
//     let n_features = 10;    // Number of dimensions
//     let n_clusters = 5;     // Number of desired clusters
//     let n_iters = 25;      // Number of iterations for K-Means

//     println!("Generating random dataset...");
//     println!(
//         "Number of Samples: {}, Number of Features: {}, Number of Clusters: {}",
//         n_samples, n_features, n_clusters
//     );

//     // Step 1: Generate cluster centers
//     let mut rng = thread_rng();
//     let cluster_center_distribution = Normal::new(50.0, 10.0).unwrap(); // Mean=50, StdDev=10
//     let mut cluster_centers = Vec::new();
//     for _ in 0..n_clusters {
//         let center: Vec<f32> = (0..n_features)
//             .map(|_| cluster_center_distribution.sample(&mut rng))
//             .collect();
//         cluster_centers.push(center);
//     }

//     // Step 2: Assign each sample to a cluster and generate data points around the cluster centers
//     let mut data = Vec::with_capacity(n_samples * n_features);
//     let cluster_assignment_distribution = Uniform::new(0, n_clusters);
//     let point_distribution = Normal::new(0.0, 5.0).unwrap(); // Spread within clusters

//     for _ in 0..n_samples {
//         let cluster_idx = cluster_assignment_distribution.sample(&mut rng);
//         let center = &cluster_centers[cluster_idx];
//         for &feature in center.iter() {
//             let point = feature + point_distribution.sample(&mut rng);
//             data.push(point);
//         }
//     }

//     // Convert the data vector into a 2D ndarray array
//     let X = Array2::<f32>::from_shape_vec((n_samples, n_features), data).expect("Error creating array");

//     println!("Dataset generated.");

//     // Create a BalancedKMeans instance.
//     let bkmeans = BalancedKMeans::new(n_clusters, n_iters);

//     println!("Starting balanced K-Means clustering...");
//     // Fit the model to the data.
//     let centroids = bkmeans.fit(&X);
//     println!("Clustering completed.");

//     // Predict cluster labels for the data.
//     let labels = bkmeans.predict(&X, &centroids);

//     // Output summary statistics
//     println!("Final Centroids:\n{}", centroids);
//     println!("Cluster Assignment Counts:");
//     for cluster_id in 0..n_clusters {
//         let count = labels.iter().filter(|&&x| x == cluster_id).count();
//         println!("Cluster {}: {} points", cluster_id, count);
//     }
// }
// use ndarray::{Array1, Array2};
// use ndarray_rand::RandomExt;

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-50.0, 50.0);

    Array2::from_shape_fn((rows, cols), |_| dist.sample(&mut rng) as f32)
}

fn verify_assignments(X: &Array2<f32>, centroids: &Array2<f32>, labels: &Array1<usize>) -> bool {
    let n_samples = X.len_of(Axis(0));
    let n_centroids = centroids.len_of(Axis(0));

    // Initialize a flag to track verification status
    let mut all_correct = true;

    // Process in parallel for efficiency
    let incorrect_assignments = (0..n_samples)
        .into_par_iter()
        .filter(|&i| {
            let x = X.row(i);
            let assigned_centroid = centroids.row(labels[i]);
            let assigned_distance = (&x - &assigned_centroid).mapv(|v| v * v).sum();

            // Find the minimum distance to any centroid
            let mut min_distance = std::f32::INFINITY;
            for j in 0..n_centroids {
                let c = centroids.row(j);
                let dist = (&x - &c).mapv(|v| v * v).sum();
                if dist < min_distance {
                    min_distance = dist;
                }
            }

            // Check if the assigned distance matches the minimum distance
            if (assigned_distance - min_distance).abs() > 1e-6 {
                // Found an incorrect assignment
                true
            } else {
                false
            }
        })
        .count();

    if incorrect_assignments > 0 {
        println!(
            "Verification failed: {} points are not assigned to their nearest centroids.",
            incorrect_assignments
        );
        all_correct = false;
    }

    all_correct
}

/// Verifies that each centroid is the mean of all points assigned to its cluster.
///
/// # Arguments
///
/// * `X` - The dataset as a 2D array.
/// * `centroids` - The centroids as a 2D array.
/// * `labels` - The cluster labels for each point.
///
/// # Returns
///
/// * `bool` - `true` if all centroids are correctly computed, `false` otherwise.
fn verify_centroid_means(X: &Array2<f32>, centroids: &Array2<f32>, labels: &Array1<usize>) -> bool {
    let n_centroids = centroids.len_of(Axis(0));
    let n_features = centroids.len_of(Axis(1));

    // Initialize a flag to track verification status
    let mut all_correct = true;

    // Process each centroid in parallel
    let incorrect_centroids = (0..n_centroids)
        .into_par_iter()
        .filter(|&j| {
            // Collect indices of points assigned to centroid j
            let assigned_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(idx, &label)| if label == j { Some(idx) } else { None })
                .collect();

            let count = assigned_indices.len();
            if count == 0 {
                // No points assigned to this centroid; depending on implementation, this might be acceptable
                // or might indicate a problem. Here, we'll consider it a failure.
                println!("Centroid {} has no points assigned.", j);
                return true; // Incorrect
            }

            // Sum all points assigned to centroid j
            let mut sum = Array1::<f32>::zeros(n_features);
            for &idx in &assigned_indices {
                sum += &X.row(idx);
            }

            // Compute mean
            let mean = &sum / (count as f32);

            // Compare mean with the centroid
            let centroid = centroids.row(j);
            let difference = (&mean - &centroid).mapv(|v| v.abs()).sum();
            let centroid_norm = (&centroid).mapv(|v| v.abs()).sum();

            // Define a small tolerance for floating-point comparisons
            println!(
                "Centroid {}: Difference = {} Norm = {} Ratio = {}",
                j,
                difference,
                (&centroid).mapv(|v| v * v).sum(),
                difference / centroid_norm
            );
            difference / centroid_norm > 1e-2
        })
        .count();

    if incorrect_centroids > 0 {
        println!(
            "Centroid Mean Verification failed: {} centroids are not correctly computed as the mean of their assigned points.",
            incorrect_centroids
        );
        all_correct = false;
    }

    all_correct
}

fn main() {
    // Parameters for data generation
    let n_samples = 100000; // Number of data points
    let n_features = 10; // Number of dimensions
    let n_clusters = 256; // Number of desired clusters
    let n_iters = 100; // Number of iterations for K-Means

    println!("Generating random dataset...");
    println!(
        "Number of Samples: {}, Number of Features: {}, Number of Clusters: {}",
        n_samples, n_features, n_clusters
    );

    // Generate random data uniformly distributed
    let X = generate_random_matrix(n_samples, n_features);

    println!("Dataset generated.");

    // Create a BalancedKMeans instance.
    let bkmeans = BalancedKMeans::new(n_clusters, n_iters);

    println!("Starting balanced K-Means clustering...");
    // Fit the model to the data.
    let centroids = bkmeans.fit(&X);
    println!("Clustering completed.");

    // Predict cluster labels for the data.
    let labels = bkmeans.predict(&X, &centroids);

    // Output summary statistics
    println!("Final Centroids:\n{}", centroids);
    println!("Cluster Assignment Counts:");
    for cluster_id in 0..n_clusters {
        let count = labels.iter().filter(|&&x| x == cluster_id).count();
        println!("Cluster {}: {} points", cluster_id, count);
    }

    // Verification: Ensure all points are assigned to the nearest centroid
    println!("Starting verification of cluster assignments...");
    let verification_passed = verify_assignments(&X, &centroids, &labels);
    if verification_passed {
        println!("Verification passed: All points are correctly assigned to the nearest centroid.");
    } else {
        println!("Verification failed: Some points are not assigned to the nearest centroid.");
    }

    // Verification: Ensure each centroid is the mean of its assigned points
    println!("Starting verification of centroid means...");
    let centroid_verification_passed = verify_centroid_means(&X, &centroids, &labels);
    if centroid_verification_passed {
        println!("Centroid Mean Verification passed: All centroids are correctly computed as the mean of their assigned points.");
    } else {
        println!("Centroid Mean Verification failed: Some centroids are not correctly computed.");
    }
}
