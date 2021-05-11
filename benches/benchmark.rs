use criterion::BenchmarkId;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use rand::distributions::{Bernoulli, Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Beta;
use reservoir_sampling::unweighted::core::r as reservoir_sample;
use sslap::{AuctionSolver, Float};

type UInt = u32;

fn gen_symmetric_input(
    seed: u64,
    size: UInt,
    density: Float,
    min_value: Float,
    max_value: Float,
) -> (Vec<UInt>, Vec<UInt>, Vec<Float>) {
    let mut row_indices = Vec::with_capacity((size.pow(2) as Float * density) as usize * 2);
    let mut column_indices = Vec::with_capacity((size.pow(2) as Float * density) as usize * 2);
    let mut values = Vec::with_capacity((size.pow(2) as Float * density) as usize * 2);
    let mut val_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut filter_rng = ChaCha8Rng::seed_from_u64(seed + 1);

    let between = Uniform::from(min_value..max_value);
    let num_of_arcs_fully_dense = (size as u32).pow(2);
    let target_elements_from_prng = ((num_of_arcs_fully_dense as Float) * density) as u32;
    let whether_to_add = Bernoulli::from_ratio(target_elements_from_prng, num_of_arcs_fully_dense)
        .expect("unexpected error");
    let mut ensured_i_to_j = (0..size).collect::<Vec<UInt>>();
    ensured_i_to_j.as_mut_slice().shuffle(&mut filter_rng);
    (0..size)
        .flat_map(|i| (0..size).map(move |j| (i, j)))
        .for_each(|(i, j)| {
            if whether_to_add.sample(&mut filter_rng) || (ensured_i_to_j[i as usize] == j) {
                let v = between.sample(&mut val_rng);
                if ensured_i_to_j[i as usize] != UInt::MAX {
                    ensured_i_to_j[i as usize] = UInt::MAX;
                }
                row_indices.push(i);
                column_indices.push(j);
                values.push(v);
            }
        });
    assert!(
        row_indices.len() >= size as usize,
        "row_indices.len() & size: {} < {}",
        row_indices.len(),
        size
    );
    assert!(
        column_indices.len() >= size as usize,
        "column_indices.len() & size: {} < {}",
        column_indices.len(),
        size
    );
    assert!(
        values.len() >= size as usize,
        "values.len() & size: {} < {}",
        values.len(),
        size
    );
    (row_indices, column_indices, values)
}

fn gen_asymmetric_input(
    seed: u64,
    num_of_people: UInt,
    num_of_objects: UInt,
    arcs_per_person: UInt,
    min_value: Float,
    range_width: Float,
) -> (Vec<UInt>, Vec<UInt>, Vec<Float>) {
    let mut row_indices = Vec::with_capacity((num_of_people * arcs_per_person) as usize);
    let mut column_indices = Vec::with_capacity((num_of_people * arcs_per_person) as usize);
    let mut values = Vec::with_capacity((num_of_people * arcs_per_person) as usize);
    let mut val_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut filter_rng = ChaCha8Rng::seed_from_u64(seed + 1);
    let beta = Beta::new(3.0, 3.0).unwrap();
    (0..num_of_people)
        .map(|i| {
            let mut j_samples = vec![0; arcs_per_person as usize];
            reservoir_sample(0..num_of_objects, j_samples.as_mut_slice(), &mut filter_rng);
            j_samples.sort_unstable();
            (i, j_samples)
        })
        .for_each(|(i, j_samples)| {
            row_indices.extend(std::iter::repeat(i).take(j_samples.len()));
            column_indices.extend(j_samples.iter());
            let j_values = j_samples
                .iter()
                .map(|_| (range_width * beta.sample(&mut val_rng) + min_value).floor());
            values.extend(j_values);
        });
    assert_eq!(
        row_indices.len(),
        (num_of_people * arcs_per_person) as usize
    );
    assert_eq!(
        column_indices.len(),
        (num_of_people * arcs_per_person) as usize
    );
    assert_eq!(values.len(), (num_of_people * arcs_per_person) as usize);
    (row_indices, column_indices, values)
}

fn bench_symmetric_density_and_size(c: &mut Criterion, max_density_percent: UInt, max_size: UInt) {
    let mut group = c.benchmark_group("symmetric");
    for density in (1..=max_density_percent).map(|i| i as Float * 0.01) {
        for size in (1000..=max_size).step_by(1000) {
            let (row_indices, column_indices, values) =
                gen_symmetric_input(size as u64, size, density, 500.0, 1000.0);
            group.throughput(Throughput::Elements(row_indices.len() as u64));
            group.sample_size(10);
            group.sampling_mode(SamplingMode::Flat);
            let benchmark_id =
                BenchmarkId::new(format!("density {}", density), format!("size {}", size));
            group.bench_with_input(
                benchmark_id,
                &(size, row_indices, column_indices, values),
                |b, input| {
                    b.iter_batched(
                        || input.clone(),
                        |(size, r, c, v)| {
                            let solution = AuctionSolver::new(size, size, r.as_slice(), c, v)
                                .unwrap()
                                .solve();
                            if !solution.optimal_soln_found {
                                println!(
                                    "not optimal: nits {}, nreductions {}, num_unassigned {}, {}",
                                    solution.nits,
                                    solution.nreductions,
                                    solution.num_unassigned,
                                    solution.num_unassigned + solution.num_assigned == size as UInt
                                )
                            }
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn bench_asymmetric_num_of_people_and_arcs_per_person(
    c: &mut Criterion,
    max_num_of_people: UInt,
    max_arcs_per_person: UInt,
) {
    let mut group = c.benchmark_group("asymmetric");
    for num_of_people in (100..=max_num_of_people).step_by(200) {
        for arcs_per_person in (32..=max_arcs_per_person).step_by(8) {
            // let num_of_objects = num_of_people * arcs_per_person;
            let num_of_objects = 60000;
            let (row_indices, column_indices, values) = gen_asymmetric_input(
                num_of_people as u64,
                num_of_people,
                num_of_objects,
                arcs_per_person,
                300.0,
                700.0,
            );
            group.throughput(Throughput::Elements(row_indices.len() as u64));
            group.sample_size(10);
            group.sampling_mode(SamplingMode::Flat);
            let benchmark_id = BenchmarkId::new(
                format!("num_of_people {}", num_of_people),
                format!(
                    "num_of_objects {}, arcs_per_person {}",
                    num_of_objects, arcs_per_person
                ),
            );
            group.bench_with_input(
                benchmark_id,
                &(
                    num_of_people,
                    num_of_objects,
                    row_indices,
                    column_indices,
                    values,
                ),
                |b, input| {
                    b.iter_batched(
                        || input.clone(),
                        |(num_of_people, num_of_objects, r, c, v)| {
                            let solution = AuctionSolver::new(
                                num_of_people,
                                num_of_objects,
                                r.as_slice(),
                                c,
                                v,
                            )
                            .unwrap()
                            .solve();
                            if !solution.optimal_soln_found {
                                println!(
                                    "not optimal: nits {}, nreductions {}, num_unassigned {}, {}",
                                    solution.nits,
                                    solution.nreductions,
                                    solution.num_unassigned,
                                    solution.num_unassigned + solution.num_assigned
                                        == num_of_people as UInt
                                )
                            }
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn bench_symmetric_density_1_size_10000(c: &mut Criterion) {
    bench_symmetric_density_and_size(c, 1, 10000)
}

fn bench_asymmetric_num_of_people_2000_arcs_per_person_16(c: &mut Criterion) {
    bench_asymmetric_num_of_people_and_arcs_per_person(c, 2000, 32)
}

criterion_group!(
    benches,
    bench_symmetric_density_1_size_10000,
    bench_asymmetric_num_of_people_2000_arcs_per_person_16
);
criterion_main!(benches);
