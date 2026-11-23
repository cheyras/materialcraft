//! Render package font outlines in normalized text coordinates.
use std::collections::BTreeMap;

use craft_font_catalog::{FontOutline, RunGlyph, RunInput, ShapePlanCache, shape_run};
use harfrust::{Direction, Shaper};
use read_fonts::TableProvider;
use vello_cpu::{
    Pixmap, RenderContext, Resources,
    color::palette::css::WHITE,
    kurbo::{Affine, BezPath, Rect, Shape},
};

use crate::{CookContext, CookResult, raster::composite_text};

// A short string can repeat a complex glyph across many wrapping tiles. Bound the
// outline storage and submitted path elements independently of the character count.
const MAX_ELEMENTS: usize = 1_000_000;

struct Glyph {
    path: BezPath,
    bounds: Rect,
}

fn shape_line(line: &str, shaper: &Shaper<'_>, plans: &ShapePlanCache) -> Result<Vec<RunGlyph>, String> {
    let bidi = unicode_bidi::BidiInfo::new(line, None);
    let mut glyphs = Vec::new();
    for paragraph in &bidi.paragraphs {
        let (levels, runs) = bidi.visual_runs(paragraph, paragraph.range.clone());
        for range in runs {
            let (Some(text), Some(before), Some(after)) = (line.get(range.clone()), line.get(..range.start), line.get(range.end..)) else {
                return Err("invalid text shaping range".into());
            };
            let direction = if levels.get(range.start).is_some_and(|level| level.is_rtl()) { Direction::RightToLeft } else { Direction::LeftToRight };
            let input = RunInput {
                text,
                source_start: range.start,
                pre_context: before,
                post_context: after,
                direction,
                language: None,
                features: &[],
                character_clusters: false,
                uppercase: false,
            };
            let run = shape_run(shaper, &input, |ch| ch, |face, dir, script, lang, features| plans.get_or_create(face, None, dir, script, lang, features))
                .map_err(|e| e.to_string())?;
            glyphs.extend(run);
        }
    }
    Ok(glyphs)
}

fn copies(low: f64, high: f64, period: f64, tiled: bool) -> Result<std::ops::RangeInclusive<i32>, String> {
    if !tiled {
        return Ok(0..=0);
    }
    let first = (-high / period).ceil();
    let last = ((period - low) / period).floor();
    if first < f64::from(i32::MIN) || last > f64::from(i32::MAX) {
        return Err("text transform exceeds the tile coordinate range".into());
    }
    Ok(first as i32..=last as i32)
}

struct Drawing {
    context: RenderContext,
    width: u16,
    height: u16,
    tiled: [bool; 2],
    remaining: usize,
}

impl Drawing {
    fn glyph(&mut self, ink: &Glyph, transform: Affine) -> Result<(), String> {
        if ink.path.elements().is_empty() {
            return Ok(());
        }
        let (width, height) = (f64::from(self.width), f64::from(self.height));
        let bounds = transform.transform_rect_bbox(ink.bounds);
        if !bounds.is_finite() {
            return Err("non-finite text outline transform".into());
        }
        let columns = copies(bounds.x0, bounds.x1, width, self.tiled[0])?;
        let rows = copies(bounds.y0, bounds.y1, height, self.tiled[1])?;
        if (!self.tiled[0] && (bounds.x1 <= 0.0 || bounds.x0 >= width)) || (!self.tiled[1] && (bounds.y1 <= 0.0 || bounds.y0 >= height)) {
            return Ok(());
        }
        for row in rows {
            for column in columns.clone() {
                self.remaining = self.remaining.checked_sub(ink.path.elements().len()).ok_or("text outlines exceed the drawing budget")?;
                let offset = Affine::translate((f64::from(column) * width, f64::from(row) * height));
                self.context.set_transform(offset * transform);
                self.context.fill_path(&ink.path);
            }
        }
        Ok(())
    }
}

/// Draw a Text node with the bytes of its resolved package font.
pub fn text_with_font(context: &CookContext<'_>, bytes: &[u8]) -> CookResult {
    let face = FontOutline::from_bytes(bytes, 0, &[]).map_err(|e| e.to_string())?;
    if face.metrics().units_per_em == 0 {
        return Err("font has no design units per em".into());
    }
    let font = harfrust::FontRef::from_index(bytes, 0).map_err(|e| e.to_string())?;
    let header = font.hhea().map_err(|e| e.to_string())?;
    let height_units = f64::from(header.ascender().to_i16()) - f64::from(header.descender().to_i16());
    if height_units <= 0.0 {
        return Err("font has no positive horizontal line height".into());
    }
    let pitch = height_units + f64::from(header.line_gap().to_i16());
    let data = harfrust::ShaperData::new(&font);
    let shaper = data.shaper(&font).build();
    let plans = ShapePlanCache::default();
    let (w, h) = craft_texture::clamp_size(context.w, context.h);
    let width = u16::try_from(w).map_err(|_| "text width exceeds the raster size")?;
    let height = u16::try_from(h).map_err(|_| "text height exceeds the raster size")?;
    let mut drawing = Drawing { context: RenderContext::new(width, height), width, height, tiled: context.tiling_axes(), remaining: MAX_ELEMENTS };
    drawing.context.set_paint(WHITE);

    let [a, b, c, d] = context.v4("matrix").map(f64::from);
    let inverse = if a * d == b * c { Affine::IDENTITY } else { Affine::new([a, b, c, d, 0.0, 0.0]).inverse() };
    let [x, y] = context.v2("position").map(f64::from);
    let scale = f64::from(context.f("fontsize").max(0.0)) / height_units;
    let frame = Affine::scale_non_uniform(f64::from(width), f64::from(height))
        * Affine::translate((0.5 + x, 0.5 + y))
        * inverse
        * Affine::scale_non_uniform(scale, -scale);
    if !frame.as_coeffs().into_iter().all(f64::is_finite) {
        return Err("non-finite text transform".into());
    }
    let text = context.text("text").chars().take(super::text::MAX_CHARS).collect::<String>().replace("\\n", "\n");
    let mut cache = BTreeMap::new();
    let mut outline_budget = MAX_ELEMENTS;
    for (line_index, line) in text.split('\n').enumerate() {
        let glyphs = shape_line(line, &shaper, &plans)?;
        let mut cursor = 0.0;
        let mut positions = Vec::with_capacity(glyphs.len());
        let baseline = -(line_index as f64) * pitch;
        let mut left = 0.0;
        let mut right = 0.0;
        for glyph in glyphs {
            if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(glyph.glyph_id) {
                let path = face.outline(glyph.glyph_id).map_err(|e| e.to_string())?;
                outline_budget = outline_budget.checked_sub(path.elements().len()).ok_or("font exceeds the outline storage budget")?;
                entry.insert(Glyph { bounds: path.bounding_box(), path });
            }
            let ink = cache.get(&glyph.glyph_id).ok_or("missing glyph outline")?;
            let x = cursor + f64::from(glyph.x_offset);
            if positions.is_empty() {
                left = x + ink.bounds.x0;
            }
            right = x + ink.bounds.x1;
            positions.push((glyph.glyph_id, x, baseline + f64::from(glyph.y_offset)));
            cursor += f64::from(glyph.x_advance);
        }
        let shift = match context.i("alignment") {
            0 => 0.0,
            2 => left - right,
            _ => (left - right) * 0.5,
        };
        for (id, x, y) in positions {
            if let Some(ink) = cache.get(&id) {
                drawing.glyph(ink, frame * Affine::translate((x + shift, y)))?;
            }
        }
    }
    let mut target = Pixmap::new(width, height);
    drawing.context.flush();
    drawing.context.render(&mut target, &mut Resources::new());
    Ok(vec![composite_text(context, |x, y| {
        let index = y as usize * usize::from(width) + x as usize;
        target.data().get(index).map_or(0.0, |pixel| f32::from(pixel.a) / 255.0)
    })])
}
