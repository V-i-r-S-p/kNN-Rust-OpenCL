use std::error::Error;
use std::time::Instant;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use clap::Parser;
use csv::ReaderBuilder;
use ocl::{Buffer, ProQue, Device, Platform};
use std::path::Path;
use std::fs::File;

#[derive(Debug, Clone)]
struct Point {
    features: Vec<f64>,
    label: i32,
}

#[derive(Debug)]
struct Neighbor {
    distance: f64,
    label: i32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn euclidean_distance_cpu(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn knn(train: &[Point], test: &Point, k: usize) -> i32 {
    let mut heap = BinaryHeap::with_capacity(k);

    for point in train {
        let distance = euclidean_distance_cpu(&point.features, &test.features);
        heap.push(Neighbor {
            distance,
            label: point.label,
        });

        if heap.len() > k {
            heap.pop();
        }
    }

    let mut label_counts = std::collections::HashMap::new();
    for neighbor in heap.into_iter() {
        *label_counts.entry(neighbor.label).or_insert(0) += 1;
    }

    label_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
        .unwrap_or(0)
}

fn knn_opencl(train: &[Point], test: &Point, k: usize, work_size: Option<usize>) -> Result<i32, Box<dyn Error>> {
    let platform = Platform::default();
    let device = Device::first(platform)?;
    
    let pro_que = ProQue::builder()
        .platform(platform)
        .device(device)
        .src(r#"
            __kernel void calculate_distances(
                __global const double* train_features,
                __global const double* test_point,
                __global double* distances,
                const uint num_points,
                const uint num_features
            ) {
                size_t gid = get_global_id(0);
                if (gid >= num_points) return;
                
                double sum = 0.0;
                for (uint i = 0; i < num_features; i++) {
                    double diff = train_features[gid * num_features + i] - test_point[i];
                    sum += diff * diff;
                }
                distances[gid] = sqrt(sum);
            }
        "#)
        .build()?;

    let num_points = train.len();
    let num_features = test.features.len();
    
    // Prepare data - flatten 2D array to 1D
    let train_features: Vec<f64> = train.iter()
        .flat_map(|p| p.features.iter().copied())
        .collect();
    let test_features = test.features.clone();
    
    // Create buffers with correct sizes
    let train_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .len(train_features.len())
        .copy_host_slice(&train_features)
        .build()?;
    
    let test_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .len(test_features.len())
        .copy_host_slice(&test_features)
        .build()?;
    
    let distances_buffer = Buffer::<f64>::builder()
        .queue(pro_que.queue().clone())
        .len(num_points)
        .build()?;
    
    // Build kernel with proper argument sizes
    let kernel = pro_que.kernel_builder("calculate_distances")
        .arg(&train_buffer)
        .arg(&test_buffer)
        .arg(&distances_buffer)
        .arg(num_points as u32)
        .arg(num_features as u32)
        .build()?;
    
    // Determine work size
    let global_work_size = work_size.unwrap_or_else(|| {
        // Default to next multiple of 64 that's >= num_points
        ((num_points + 63) / 64) * 64
    });

    // Execute kernel with proper global work size
    unsafe { 
        kernel.cmd()
            .global_work_size(global_work_size)
            .enq()?;
    }
    
    // Read results
    let mut distances = vec![0.0; num_points];
    distances_buffer.read(&mut distances).enq()?;
    
    // Rest of the kNN logic remains the same
    let mut heap = BinaryHeap::with_capacity(k);
    
    for (i, &distance) in distances.iter().enumerate() {
        heap.push(Neighbor {
            distance,
            label: train[i].label,
        });
        
        if heap.len() > k {
            heap.pop();
        }
    }
    
    let mut label_counts = std::collections::HashMap::new();
    for neighbor in heap.into_iter() {
        *label_counts.entry(neighbor.label).or_insert(0) += 1;
    }

    Ok(label_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
        .unwrap_or(0))
}

fn read_points_from_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(file);
    
    let mut points = Vec::new();

    for result in reader.records() {
        let record = result?;
        let len = record.len();
        
        if len < 2 {
            return Err("Each row must contain at least one feature and a label".into());
        }

        let features: Vec<f64> = record
            .iter()
            .take(len - 1)
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;

        let label = record[len - 1].parse::<i32>()?;

        points.push(Point { features, label });
    }

    Ok(points)
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to the training data CSV file
    #[clap(short = 'f', long)]
    file: String,

    /// Number of neighbors to consider (k)
    #[clap(short = 'k', long, default_value_t = 3)]
    k: usize,

    /// Test point coordinates separated by commas (e.g., "1.0,2.0,3.0")
    #[clap(
        short = 'p',
        long = "test-point",
        // Добавляем эти настройки:
        allow_hyphen_values = true,  // Разрешаем отрицательные числа
        value_delimiter = ',',      // Разделитель для значений
        value_parser = parse_f64     // Валидатор значений
    )]
    test_point: Vec<f64>,

    /// Label for the test point (optional, default is 0)
    #[clap(short = 'l', long = "test-label", default_value_t = 0)]
    test_label: i32,

    /// Use OpenCL acceleration
    #[clap(short = 'o', long)]
    opencl: bool,

    /// OpenCL work size (optional, defaults to next multiple of 64 >= number of points)
    #[clap(short = 'w', long = "work-size")]
    work_size: Option<usize>,
}

// Функция для парсинга f64
fn parse_f64(s: &str) -> Result<f64, String> {
    s.parse::<f64>().map_err(|e| format!("'{s}' is not a valid number: {e}"))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Read training data
    let train_data = read_points_from_csv(&args.file)?;
    if train_data.is_empty() {
        return Err("Training data is empty".into());
    }

    // Parse test point
    let test_features: Vec<f64> = args.test_point;

    // Check dimensions
    let expected_dim = train_data[0].features.len();
    if test_features.len() != expected_dim {
        return Err(format!(
            "Test point has {} features, but training data has {}",
            test_features.len(),
            expected_dim
        ).into());
    }

    let test_point = Point {
        features: test_features,
        label: args.test_label,
    };

    println!("Running kNN with:");
    println!("- k: {}", args.k);
    println!("- Training data points: {}", train_data.len());
    println!("- Feature dimensions: {}", expected_dim);
    println!("- Test point: {:?} (label: {})", test_point.features, test_point.label);
    println!("- Using OpenCL: {}", args.opencl);
    if args.opencl {
        println!("- Work size: {:?}", args.work_size);
    }

    // Run kNN and measure time
    let start = Instant::now();
    
    let predicted_label = if args.opencl {
        println!("Using OpenCL acceleration");
        knn_opencl(&train_data, &test_point, args.k, args.work_size)?
    } else {
        println!("Using CPU implementation");
        knn(&train_data, &test_point, args.k)
    };
    
    let duration = start.elapsed();

    println!("\nPredicted label: {}", predicted_label);
    println!("Time elapsed: {:?}", duration);

    Ok(())
}