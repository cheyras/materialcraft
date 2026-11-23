use crate::CookContext;
use craft_texture::{Image, Kind};
use materialcraft_core::OutputFormat;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputResize {
    Copy,
    Filter,
    Levels,
    PatternBackground,
    Sampled,
}

pub(crate) fn resized_input(context: &CookContext, port: usize, mode: InputResize) -> Option<Image> {
    resized_texture(context, port, mode).map(|(image, _)| image)
}

pub(super) fn blur_input(context: &CookContext, intensity: f32) -> (Image, OutputFormat) {
    // A zero intensity copies through the destination format. Even a subpixel blur uses the working precision.
    let mode = if intensity == 0.0 { InputResize::Copy } else { InputResize::Filter };
    resized_texture(context, 0, mode).unwrap_or_else(|| (Image::filled(context.w, context.h, Kind::Gray, [0.0; 4]), OutputFormat::F32))
}

pub(super) fn sampled_input(context: &CookContext, port: usize) -> (Image, OutputFormat) {
    resized_texture(context, port, InputResize::Sampled).unwrap_or_else(|| (Image::filled(context.w, context.h, Kind::Gray, [0.0; 4]), context.format))
}

fn clear_zero_signs(pixel: [f32; 4]) -> [f32; 4] {
    pixel.map(|v| if v == 0.0 { 0.0 } else { v })
}

/// Color conversion preserves negative zero in every channel when all three RGB signs are negative.
pub(super) fn converted_color(pixel: [f32; 4]) -> [f32; 4] {
    if pixel[..3].iter().any(|v| v.is_sign_positive()) { clear_zero_signs(pixel) } else { pixel }
}

fn prepare_pattern_background(image: Image, mode: InputResize, source: OutputFormat, enlarged: bool) -> Image {
    if mode != InputResize::PatternBackground {
        return image;
    }
    let gray = image.kind() == Kind::Gray;
    let flush = source == OutputFormat::F32;
    let enlarged_color = enlarged && !gray && matches!(source, OutputFormat::F16 | OutputFormat::F32);
    if !flush && !enlarged_color {
        return image;
    }
    let needs_change = |&v: &f32| (flush && v.is_subnormal()) || ((gray || enlarged_color) && v.to_bits() == (-0.0_f32).to_bits());
    if !image.data().iter().any(needs_change) {
        return image;
    }
    image.map(image.kind(), |pixel| {
        let pixel = if flush { pixel.map(crate::function::flush_subnormal) } else { pixel };
        if gray {
            clear_zero_signs(pixel)
        } else if enlarged_color {
            converted_color(pixel)
        } else {
            pixel
        }
    })
}

fn resized_texture(context: &CookContext, port: usize, mode: InputResize) -> Option<(Image, OutputFormat)> {
    let mut image = context.input(port)?.clone();
    let mut source = context.input_formats.get(port).copied().unwrap_or(if image.integer_sampling() { OutputFormat::U16 } else { OutputFormat::F32 });
    image = prepare_pattern_background(image, mode, source, false);
    let target = craft_texture::clamp_size(context.w, context.h);
    let size = (image.width(), image.height());
    // A sampled filter can take its final taps directly from an input up to one level larger on either axis.
    let direct =
        mode == InputResize::Sampled && (target.0..=target.0.saturating_mul(2)).contains(&size.0) && (target.1..=target.1.saturating_mul(2)).contains(&size.1);
    if size == target || direct {
        return Some((image, source));
    }
    let storage = if mode == InputResize::Copy {
        context.format
    } else {
        use OutputFormat::*;
        match (source, context.format) {
            (F32, _) | (_, F32) => F32,
            (F16, _) | (_, F16) => F16,
            (U8, U8) => U8,
            _ => U16,
        }
    };
    let advance = |now: u32, end: u32| {
        if now.saturating_mul(2) < end {
            now.saturating_mul(2)
        } else if end.saturating_mul(2) < now {
            now.div_ceil(2)
        } else {
            now
        }
    };
    loop {
        // Leave each axis within one doubling or halving for the final two-dimensional filter.
        let current = (image.width(), image.height());
        let intermediate = (advance(current.0, target.0), advance(current.1, target.1));
        let (w, h) = if intermediate == current { target } else { intermediate };
        image = crate::texture_sampler::resize_input_level(&image, w, h, source, context.tiling_axes());
        image = prepare_pattern_background(image, mode, source, (w, h) == target && (w > size.0 || h > size.1));
        // A final reduction can feed the Levels curve directly. An enlargement stores the
        // filtered texels first, which matters when the working image uses half precision.
        if mode == InputResize::Levels && (w, h) == target && w <= current.0 && h <= current.1 {
            return Some((image, source));
        }
        image = crate::output_precision::apply(image, storage);
        if (w, h) == target {
            return Some((image, storage));
        }
        source = storage;
    }
}
