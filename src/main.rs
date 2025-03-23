use iced::{Application, Settings, Size};

use iced_lenia::ParticleLenia;
use iced_lenia::{WIDTH, HEIGHT};

fn main() -> iced::Result {
    ParticleLenia::run(Settings {
        window: iced::window::Settings {
            size: Size::new(WIDTH, HEIGHT),
            ..Default::default()
        },
        ..Default::default()
    })
}
