use super::*;
use materialcraft_core::{OutputFormat, Resource, ResourceBytes};

/// A two-glyph TrueType face. U+0041 is a rectangle with a variable cap height.
fn rectangle_font(height: i16) -> Vec<u8> {
    fn words(values: &[i16]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_be_bytes()).collect()
    }
    fn checksum(data: &[u8]) -> u32 {
        data.chunks(4).fold(0u32, |sum, chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            sum.wrapping_add(u32::from_be_bytes(word))
        })
    }

    let mut head = vec![0; 54];
    head[..4].copy_from_slice(&0x10000u32.to_be_bytes());
    head[12..16].copy_from_slice(&0x5f0f3cf5u32.to_be_bytes());
    head[18..20].copy_from_slice(&1024i16.to_be_bytes());
    head[40..44].copy_from_slice(&words(&[512, height]));
    let mut hhea = vec![0; 36];
    hhea[..4].copy_from_slice(&0x10000u32.to_be_bytes());
    hhea[4..12].copy_from_slice(&words(&[768, -256, 0, 640]));
    hhea[34..].copy_from_slice(&2i16.to_be_bytes());
    let mut maxp = vec![0; 32];
    maxp[..4].copy_from_slice(&0x10000u32.to_be_bytes());
    maxp[4..10].copy_from_slice(&words(&[2, 4, 1]));
    let mut cmap = words(&[0, 1, 3, 1, 0, 12]);
    cmap.extend(words(&[4, 32, 0, 4, 4, 1, 0, 65, -1, 0, 65, -1, -64, 1, 0, 0]));
    let mut glyf = words(&[1, 0, 0, 512, height, 3, 0]);
    glyf.extend([1; 4]);
    glyf.extend(words(&[0, 0, 512, 0, 0, height, 0, -height]));
    let loca = words(&[0, 0, (glyf.len() / 2) as i16]);
    let tables = [(b"cmap", cmap), (b"glyf", glyf), (b"head", head), (b"hhea", hhea), (b"hmtx", words(&[640, 0, 640, 0])), (b"loca", loca), (b"maxp", maxp)];
    let mut font = words(&[1, 0, 7, 64, 2, 48]);
    font.resize(12 + 16 * tables.len(), 0);
    let mut head_start = 0;
    for (index, (tag, data)) in tables.into_iter().enumerate() {
        let offset = font.len();
        if tag == b"head" {
            head_start = offset;
        }
        let entry = &mut font[12 + 16 * index..12 + 16 * (index + 1)];
        entry[..4].copy_from_slice(tag);
        entry[4..8].copy_from_slice(&checksum(&data).to_be_bytes());
        entry[8..12].copy_from_slice(&(offset as u32).to_be_bytes());
        entry[12..].copy_from_slice(&(data.len() as u32).to_be_bytes());
        font.extend(data);
        font.resize(font.len().next_multiple_of(4), 0);
    }
    let adjustment = 0xb1b0_afba_u32.wrapping_sub(checksum(&font));
    font[head_start + 8..head_start + 12].copy_from_slice(&adjustment.to_be_bytes());
    font
}

#[test]
fn package_font_outlines_render_and_invalidate_the_parent_instance() {
    let (mut parent, instance) = instancing("Preview", "Letter");
    parent.size = [6, 5];
    parent.format = OutputFormat::F32;
    let mut child = Graph::new("Letter");
    child.size = [6, 5];
    let text = child.add_node("text", [0.0; 2]);
    child.node_mut(text).unwrap().params.extend([
        ("text".into(), Value::Text("A".into())),
        ("fontdata".into(), Value::Text("Ink".into())),
        ("fontsize".into(), Value::Float(0.5)),
    ]);
    let output = child.add_node("output", [100.0, 0.0]);
    child.connect(text, "output", output, "input").unwrap();
    let mut package = Package::new("Original lettering");
    package.graphs = vec![parent, child];
    package.resources.push(Resource::new("Ink", "Ink.ttf", ResourceBytes::new(rectangle_font(512))));
    let cache = Mutex::new(Cache::default());
    let options = Options::new(&no_res);
    for (height, first_row) in [(512, 8), (256, 12)] {
        package.resources[0].bytes = ResourceBytes::new(rectangle_font(height));
        let direct = cook(&package, 1, &options, &cache);
        assert!(direct.nodes[&text].error.is_none(), "{:?}", direct.nodes[&text].error);
        let nested = cook(&package, 0, &options, &cache);
        assert!(!nested.nodes[&instance].cached);
        let image = nested.image(instance, 0).unwrap();
        assert_eq!((image.width(), image.height()), (64, 32));
        for y in 0..32 {
            for x in 0..64 {
                let ink = (24..40).contains(&x) && (first_row..16).contains(&y);
                assert_eq!(image.get_gray(x, y, false), f32::from(ink), "({x}, {y}) at cap height {height}");
            }
        }
        assert!(cook(&package, 0, &options, &cache).nodes[&instance].cached);
    }
}
