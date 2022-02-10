use criterion::{black_box, criterion_group, criterion_main, Criterion};

use parallel_vec::ParallelVec;

#[derive(Clone, Copy, Default)]
struct Small(u32);
#[derive(Clone, Copy, Default)]
struct Big([u64; 32]);

trait Inc {
    fn inc(&mut self);
}

impl Inc for Small {
    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl Inc for Big {
    fn inc(&mut self) {
        for i in 0..32 {
            self.0[i] += 1;
        }
    }
}

fn bench_iter_2(c: &mut Criterion, size: usize) {
    let small = (Small(0), Small(1));
    let mut vec = Vec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_vec_small_2x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_small_2x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
            }
        })
    });
    let mixed = (Big::default(), Small(1));
    let mut vec = Vec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_vec_mixed_2x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_mixed_2x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
            }
        })
    });
    let big = (Big::default(), Big::default());
    let mut vec = Vec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_vec_big_2x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_big_2x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
            }
        })
    });
}

fn bench_iter_3(c: &mut Criterion, size: usize) {
    let small = (Small(0), Small(1), Small(2));
    let mut vec = Vec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_vec_small_3x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_small_3x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
            }
        })
    });
    let mixed = (Big::default(), Small(1), Big::default());
    let mut vec = Vec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_vec_mixed_3x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_mixed_3x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
            }
        })
    });
    let big = (Big::default(), Big::default(), Big::default());
    let mut vec = Vec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_vec_big_3x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_big_3x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
            }
        })
    });
}

fn bench_iter_4(c: &mut Criterion, size: usize) {
    let small = (Small(0), Small(1), Small(2), Small(3));
    let mut vec = Vec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_vec_small_4x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_small_4x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
            }
        })
    });
    let mixed = (Big::default(), Small(1), Big::default(), Small(2));
    let mut vec = Vec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_vec_mixed_4x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_mixed_4x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
            }
        })
    });
    let big = (Big::default(), Big::default(), Big::default(), Big::default());
    let mut vec = Vec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_vec_big_4x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_big_4x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
            }
        })
    });
}

fn bench_iter_5(c: &mut Criterion, size: usize) {
    let small = (Small(0), Small(1), Small(2), Small(3), Small(4));
    let mut vec = Vec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_vec_small_5x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4, item_5) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
                black_box(item_5).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![small]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_small_5x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4, item_5) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
                black_box(item_5).inc();
            }
        })
    });
    let mixed = (Big::default(), Small(1), Big::default(), Small(2), Big::default());
    let mut vec = Vec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_vec_mixed_5x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4, item_5) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
                black_box(item_5).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![mixed]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_mixed_5x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4, item_5) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
                black_box(item_5).inc();
            }
        })
    });
    let big = (Big::default(), Big::default(), Big::default(), Big::default(), Big::default());
    let mut vec = Vec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_vec_big_5x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4, item_5) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
                black_box(item_5).inc();
            }
        })
    });
    let mut vec = ParallelVec::from(vec![big]).repeat(size);
    c.bench_function(&format!("iter_parallelvec_big_5x_{}", size), |b| {
        b.iter(|| {
            for (item_1, item_2, item_3, item_4, item_5) in vec.iter_mut() {
                black_box(item_1).inc();
                black_box(item_2).inc();
                black_box(item_3).inc();
                black_box(item_4).inc();
                black_box(item_5).inc();
            }
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    for size in [10, 100, 1000, 100000] {
        bench_iter_2(c, size);
        bench_iter_3(c, size);
        bench_iter_4(c, size);
        bench_iter_5(c, size);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
