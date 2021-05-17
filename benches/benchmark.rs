use criterion::BenchmarkId;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use rand::distributions::{Bernoulli, Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Beta;
use reservoir_sampling::unweighted::core::r as reservoir_sample;
use sparse_linear_assignment::{AuctionSolver, Float};

type UInt = u32;

fn gen_symmetric_input(
    solver: &mut AuctionSolver<UInt>,
    seed: u64,
    size: UInt,
    density: Float,
    min_value: Float,
    max_value: Float,
) {
    let mut val_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut filter_rng = ChaCha8Rng::seed_from_u64(seed + 1);

    let between = Uniform::from(min_value..max_value);
    let num_of_arcs_fully_dense = (size as u32).pow(2);
    let target_elements_from_prng = ((num_of_arcs_fully_dense as Float) * density) as u32;
    let whether_to_add = Bernoulli::from_ratio(target_elements_from_prng, num_of_arcs_fully_dense)
        .expect("unexpected error");
    let mut ensured_i_to_j = (0..size).collect::<Vec<UInt>>();
    ensured_i_to_j.as_mut_slice().shuffle(&mut filter_rng);

    solver.init(size, size, None, None).unwrap();
    (0..size)
        .flat_map(|i| (0..size).map(move |j| (i, j)))
        .for_each(|(i, j)| {
            if whether_to_add.sample(&mut filter_rng) || (ensured_i_to_j[i as usize] == j) {
                let v = between.sample(&mut val_rng);
                if ensured_i_to_j[i as usize] != UInt::MAX {
                    ensured_i_to_j[i as usize] = UInt::MAX;
                }
                solver.add_value(i, j, v).unwrap();
            }
        });
}

fn gen_asymmetric_input(
    solver: &mut AuctionSolver<UInt>,
    seed: u64,
    num_of_people: UInt,
    num_of_objects: UInt,
    arcs_per_person: UInt,
    min_value: Float,
    range_width: Float,
) {
    let mut val_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut filter_rng = ChaCha8Rng::seed_from_u64(seed + 1);
    let beta = Beta::new(3.0, 3.0).unwrap();

    solver
        .init(num_of_people, num_of_objects, None, None)
        .unwrap();
    (0..num_of_people)
        .map(|i| {
            let mut j_samples = vec![0; arcs_per_person as usize];
            reservoir_sample(0..num_of_objects, j_samples.as_mut_slice(), &mut filter_rng);
            j_samples.sort_unstable();
            (i, j_samples)
        })
        .for_each(|(i, j_samples)| {
            let j_values = j_samples
                .iter()
                .map(|_| (range_width * beta.sample(&mut val_rng) + min_value).floor())
                .collect::<Vec<_>>();
            solver
                .extend_from_values(i, j_samples.as_slice(), j_values.as_slice())
                .unwrap();
        });
}

fn bench_symmetric_density_and_size(c: &mut Criterion, max_density_percent: UInt, max_size: UInt) {
    let mut group = c.benchmark_group("symmetric_random_degree");
    let (mut solver, solution) = AuctionSolver::new(
        max_size as usize,
        max_size as usize,
        (max_size as usize).pow(2) * (max_density_percent as usize) / 100,
    );
    for density in (1..=max_density_percent).map(|i| i as Float * 0.01) {
        for size in (1000..=max_size).step_by(1000) {
            gen_symmetric_input(&mut solver, size as u64, size, density, 500.0, 1000.0);
            group.throughput(Throughput::Elements(solver.num_of_arcs() as u64));
            group.sample_size(10);
            group.sampling_mode(SamplingMode::Flat);
            let benchmark_id =
                BenchmarkId::new("forward", format!("density {} size {}", density, size));
            let input = (solver.clone(), solution.clone());

            group.bench_with_input(benchmark_id, &input, |b, input| {
                b.iter_batched(
                    || input.clone(),
                    |(mut solver, mut solution)| {
                        solver.solve(&mut solution, false, None).unwrap();
                        if solution.num_unassigned != 0 {
                            println!(
                                "not optimal: nits {}, nreductions {}, num_unassigned {}",
                                solution.nits, solution.nreductions, solution.num_unassigned,
                            )
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
            let benchmark_id =
                BenchmarkId::new("khosla", format!("density {} size {}", density, size));
            let input = (solver.clone(), solution.clone());

            group.bench_with_input(benchmark_id, &input, |b, input| {
                b.iter_batched(
                    || input.clone(),
                    |(mut solver, mut solution)| {
                        solver.solve_approx(&mut solution, false, None).unwrap();
                        if solution.num_unassigned != 0 {
                            println!(
                                "not optimal: nits {}, nreductions {}, num_unassigned {}",
                                solution.nits, solution.nreductions, solution.num_unassigned,
                            )
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_asymmetric_num_of_people_and_arcs_per_person(
    c: &mut Criterion,
    max_num_of_people: UInt,
    max_arcs_per_person: UInt,
) {
    let mut group = c.benchmark_group("asymmetric_ksparse");
    let num_of_objects = 60000;
    let (mut solver, solution) = AuctionSolver::new(
        max_num_of_people as usize,
        num_of_objects as usize,
        (max_num_of_people * max_arcs_per_person) as usize,
    );
    for num_of_people in (100..=max_num_of_people).step_by(200) {
        for arcs_per_person in (32..=max_arcs_per_person).step_by(8) {
            // let num_of_objects = num_of_people * arcs_per_person;
            gen_asymmetric_input(
                &mut solver,
                num_of_people as u64,
                num_of_people,
                num_of_objects,
                arcs_per_person,
                300.0,
                700.0,
            );
            group.throughput(Throughput::Elements(solver.num_of_arcs() as u64));
            group.sampling_mode(SamplingMode::Flat);
            let benchmark_id = BenchmarkId::new(
                "classic",
                format!(
                    "num_of_people {}, num_of_objects {}, arcs_per_person {}",
                    num_of_people, num_of_objects, arcs_per_person
                ),
            );
            let input = (solver.clone(), solution.clone());
            group.bench_with_input(benchmark_id, &input, |b, input| {
                b.iter_batched(
                    || input.clone(),
                    |(mut solver, mut solution)| {
                        solver.solve(&mut solution, false, None).unwrap();
                        if !solution.optimal_soln_found {
                            println!(
                                "not optimal: nits {}, nreductions {}, num_unassigned {}",
                                solution.nits, solution.nreductions, solution.num_unassigned,
                            )
                        }
                    },
                    BatchSize::SmallInput,
                );
            });
            let input = (solver.clone(), solution.clone());
            let benchmark_id_imporoved = BenchmarkId::new(
                "improved",
                format!(
                    "num_of_people {}, num_of_objects {}, arcs_per_person {}",
                    num_of_people, num_of_objects, arcs_per_person
                ),
            );
            group.bench_with_input(benchmark_id_imporoved, &input, |b, input| {
                b.iter_batched(
                    || input.clone(),
                    |(mut solver, mut solution)| {
                        solver.solve_approx(&mut solution, false, None).unwrap();
                        if solution.num_unassigned != 0 {
                            println!(
                                "not optimal: nits {}, nreductions {}, num_unassigned {}",
                                solution.nits, solution.nreductions, solution.num_unassigned,
                            )
                        }
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_symmetric_density_1_size_10000(c: &mut Criterion) {
    bench_symmetric_density_and_size(c, 1, 10000)
}

fn bench_asymmetric_num_of_people_2000_arcs_per_person_32(c: &mut Criterion) {
    bench_asymmetric_num_of_people_and_arcs_per_person(c, 2000, 32)
}

criterion_group!(
    benches,
    bench_symmetric_density_1_size_10000,
    bench_asymmetric_num_of_people_2000_arcs_per_person_32
);
criterion_main!(benches);
