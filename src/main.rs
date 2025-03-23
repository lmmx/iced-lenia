use iced::{
    executor, Application, Color, Command, Element,
    Length, Point, Settings, Subscription,
    Theme,
    widget::canvas::{self, Canvas, Frame, Path, Geometry},
    mouse::{Cursor, Interaction},
    Rectangle, Size,
};
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rayon::prelude::*;
use std::f32::consts::PI;

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 800.0;

const D: usize = 3; // 3D positions (we ignore z for drawing)
const N: usize = 900;
const DT: f32 = 0.1;

/// The typical Lenia "bell" function.
fn bell(x: f32, m: f32, s: f32) -> f32 {
    (-((x - m) / s).powi(2)).exp()
}

/// A simple repulsion function.
fn repulse(x: f32) -> f32 {
    (1.0 - x).max(0.0).powi(2)
}

/// Numerically integrates bell(r, w) * surface_area_factor * r^(D-1)
/// over a radius interval, matching the Python reference normalization.
fn compute_kernel_sum(d: usize, r: f32, w: f32) -> f32 {
    let lower = (r - 4.0 * w).max(0.0);
    let upper = r + 4.0 * w;
    let steps = 51;
    let delta = (upper - lower) / (steps - 1) as f32;

    let dimension_factor = match d {
        2 => 2.0 * PI, // for 2D (circumference)
        3 => 4.0 * PI, // for 3D (surface area)
        _ => panic!("compute_kernel_sum: only d=2 or d=3 is implemented."),
    };

    let mut sum = 0.0;
    let mut last_val = None;
    for i in 0..steps {
        let dist = lower + (i as f32) * delta;
        let val = bell(dist, r, w) * dimension_factor * dist.powi((d - 1) as i32);
        if let Some(prev) = last_val {
            sum += 0.5 * (val + prev) * delta; // trapezoidal integration
        }
        last_val = Some(val);
    }
    sum
}

/// Compute the energy for one particle at position `x_i` against all others in `X`.
fn energy(
    X: &Array2<f32>,
    x_i: &Array1<f32>,
    kernel_sum: f32,
    r: f32,
    w: f32,
    m: f32,
    s: f32,
    c_rep: f32,
) -> f32 {
    let distances = (X - x_i)
        .mapv(|v| v.powi(2))
        .sum_axis(ndarray::Axis(1))
        .mapv(f32::sqrt)
        .mapv(|val| val.max(1e-10));

    let u = distances.mapv(|d| bell(d, r, w)).sum() / kernel_sum;
    let g = bell(u, m, s);
    let r_ener = distances.mapv(repulse).sum() * c_rep / 2.0;
    r_ener - g
}

/// Compute a numerical gradient for `x_i` using finite differences.
fn numerical_gradient(
    X: &Array2<f32>,
    xi: &Array1<f32>,
    kernel_sum: f32,
    r: f32,
    w: f32,
    m: f32,
    s: f32,
    c_rep: f32,
    h: f32,
) -> Array1<f32> {
    let mut grad = Array1::zeros(D);
    for dim in 0..D {
        let mut x_plus = xi.clone();
        x_plus[dim] += h;
        let f_plus = energy(X, &x_plus, kernel_sum, r, w, m, s, c_rep);

        let mut x_minus = xi.clone();
        x_minus[dim] -= h;
        let f_minus = energy(X, &x_minus, kernel_sum, r, w, m, s, c_rep);

        grad[dim] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}

/// The main struct, including the particle positions and their energies.
struct ParticleLenia {
    particles: Array2<f32>,
    energies: Vec<f32>,
    cache: canvas::Cache,

    // Lenia parameters
    r: f32,
    w: f32,
    m: f32,
    s: f32,
    c_rep: f32,

    kernel_sum: f32,
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
}

impl Application for ParticleLenia {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: Self::Flags) -> (Self, Command<Message>) {
        // Lenia parameters; these may be tuned further.
        let r = 2.0;
        let w = 0.64;
        let m = 0.72;
        let s = 0.26;
        let c_rep = 1.0;
        let kernel_sum = compute_kernel_sum(D, r, w);

        // Initialize particles randomly.
        let particles = Array2::random((N, D), Normal::new(0.0, 1.0).unwrap());
        let energies = vec![0.0; N];

        (
            Self {
                particles,
                energies,
                cache: canvas::Cache::new(),
                r,
                w,
                m,
                s,
                c_rep,
                kernel_sum,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "Particle Lenia Simulation".into()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Tick => {
                self.step();
                self.cache.clear();
            }
        }
        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick)
    }

    fn view(&self) -> Element<Message> {
        Canvas::new(self)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

impl ParticleLenia {
    /// Advance one simulation time step.
    fn step(&mut self) {
        let x_prev = self.particles.clone();

        // Copy scalar parameters so the parallel closure doesn't capture &self.
        let kernel_sum = self.kernel_sum;
        let r = self.r;
        let w = self.w;
        let m = self.m;
        let s = self.s;
        let c_rep = self.c_rep;

        // Update particle positions in parallel.
        let updates: Vec<Array1<f32>> = (0..N)
            .into_par_iter()
            .map(|i| {
                let xi = x_prev.row(i).to_owned();
                let grad = numerical_gradient(&x_prev, &xi, kernel_sum, r, w, m, s, c_rep, 1e-4);
                xi - grad * DT
            })
            .collect();

        let new_positions = Array2::from_shape_fn((N, D), |(i, j)| updates[i][j]);
        self.particles.assign(&new_positions);

        // Compute energies for each particle in parallel.
        let new_energies: Vec<f32> = (0..N)
            .into_par_iter()
            .map(|i| {
                let xi = new_positions.row(i).to_owned();
                energy(&new_positions, &xi, kernel_sum, r, w, m, s, c_rep)
            })
            .collect();
        self.energies = new_energies;
    }
}

impl<Message> canvas::Program<Message, Theme> for ParticleLenia {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        // Compute min and max energy values to normalize our color mapping.
        let min_energy = self.energies.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_energy = self.energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let geometry = self.cache.draw(renderer, bounds.size(), |frame: &mut Frame| {
            // Draw background.
            frame.fill_rectangle(Point::ORIGIN, frame.size(), Color::BLACK);

            // Render each particle with an energy-based color.
            for (i, particle) in self.particles.rows().into_iter().enumerate() {
                let x = (bounds.width / 2.0) + particle[0] * (bounds.width / 4.0);
                let y = (bounds.height / 2.0) + particle[1] * (bounds.height / 4.0);

                // Normalize energy value to [0, 1].
                let energy_value = self.energies[i];
                let factor = if max_energy - min_energy > 0.0 {
                    (energy_value - min_energy) / (max_energy - min_energy)
                } else {
                    0.5
                };

                // Map factor to a color. Here, low energy gives more green,
                // and high energy gives more red (blue is fixed).
                let circle_color = Color::from_rgb(factor, 1.0 - factor, 0.5);

                let circle = Path::circle(Point::new(x, y), 2.0);
                frame.fill(&circle, circle_color);
            }
        });

        vec![geometry]
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: Cursor,
    ) -> Interaction {
        Interaction::default()
    }
}

fn main() -> iced::Result {
    ParticleLenia::run(Settings {
        window: iced::window::Settings {
            size: Size::new(WIDTH, HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    })
}
