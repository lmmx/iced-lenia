use iced::{
    executor, Application, Color, Command, Element,
    Length, Point, Settings, Subscription,
    Theme, // some versions rename Style to Theme, etc.
    widget::canvas::{self, Canvas, Frame, Path, Geometry},
    mouse::{Cursor, Interaction},
    Rectangle, Size,
};
use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 800.0;

const D: usize = 3; // Dimensions
const N: usize = 900; // Number of particles
const DT: f32 = 0.1;

fn bell(x: f32, m: f32, s: f32) -> f32 {
    (-((x - m) / s).powi(2)).exp()
}

fn repulse(x: f32) -> f32 {
    (1.0 - x).max(0.0).powi(2)
}

fn energy(X: &Array2<f32>, xi: &Array1<f32>) -> f32 {
    let r = 2.0;
    let w = 0.64;
    let m = 0.72;
    let s = 0.26;
    let c_rep = 1.0;

    // Subtract xi from each row in X. This is broadcast if shape is (N, D) - (D).
    let distances = (X - xi)
        .mapv(|v| v.powi(2))
        .sum_axis(ndarray::Axis(1))
        .mapv(f32::sqrt)
        .mapv(|v| v.max(1e-10));

    // For simplicity, you say “1.0” as kernel_sum. Modify as needed.
    let kernel_sum: f32 = 1.0;

    let u: f32 = distances.mapv(|d| bell(d, r, w)).sum() / kernel_sum;
    let g = bell(u, m, s);
    let r_ener = distances.mapv(repulse).sum() * c_rep / 2.0;

    r_ener - g
}

// Our main struct:
struct ParticleLenia {
    particles: Array2<f32>,
    cache: canvas::Cache,
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
        let particles = Array2::random((N, D), Normal::new(0.0, 1.0).unwrap());

        (
            Self {
                particles,
                cache: canvas::Cache::new(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Particle Lenia Simulation")
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
        // ~60 fps (16 ms)
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
    fn step(&mut self) {
        let X_prev = self.particles.clone();
        for i in 0..N {
            let xi = X_prev.row(i); // This is a view
            let grad = numerical_gradient(&X_prev, &xi.to_owned(), 1e-4);
            // Convert the view `xi` into an owned Array1 so we can do the subtraction:
            let update = xi.to_owned() - grad * DT;
            self.particles.row_mut(i).assign(&update);
        }
    }
}

// We must implement Program<Message, Theme> (with the correct type params).
impl<Message> canvas::Program<Message, Theme> for ParticleLenia {
    // If you want to store some ephemeral canvas state, put it here:
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer, 
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        // Use the new cache.draw(...) signature:
        let geometry = self.cache.draw(renderer, bounds.size(), |frame: &mut Frame| {
            frame.fill_rectangle(Point::ORIGIN, frame.size(), Color::BLACK);

            // Draw each particle:
            for particle in self.particles.rows() {
                let x = (bounds.width / 2.0) + particle[0] * (bounds.width / 4.0);
                let y = (bounds.height / 2.0) + particle[1] * (bounds.height / 4.0);

                let circle = Path::circle(Point::new(x, y), 2.0);
                frame.fill(&circle, Color::from_rgb(0.0, 1.0, 0.7));
            }
        });

        // Return a Vec of geometry
        vec![geometry]
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: Cursor,
    ) -> Interaction {
        // By default, no special mouse interactions
        Interaction::default()
    }
}

// Simple numerical gradient approximation
fn numerical_gradient(X: &Array2<f32>, xi: &Array1<f32>, h: f32) -> Array1<f32> {
    let mut grad = Array1::zeros(D);
    for i in 0..D {
        let mut x_plus = xi.clone();
        x_plus[i] += h;
        let f_plus = energy(X, &x_plus);

        let mut x_minus = xi.clone();
        x_minus[i] -= h;
        let f_minus = energy(X, &x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}

fn main() -> iced::Result {
    ParticleLenia::run(Settings {
        window: iced::window::Settings {
            // Use Size::new instead of a tuple
            size: Size::new(WIDTH, HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    })
}
