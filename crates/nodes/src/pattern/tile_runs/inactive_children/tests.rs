use super::ranges;

// Each of the 16,384 parents draws before its four children. Move the first
// selected draws in one incoming pair to the tile edge, adding one copy each.
fn draw_ranks(edge_draws: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut copies = 0_u32;
    for draw in 0_usize..81_920 {
        starts.push(copies);
        let offset = draw.checked_sub(50 * 1280 + 2);
        let wraps = offset.is_some_and(|n| n.is_multiple_of(23) && n / 23 < edge_draws);
        copies += 1 + u32::from(wraps);
        if (draw + 1).is_multiple_of(80) {
            ends.push(copies);
        }
    }
    let mut ranks = vec![usize::MAX; copies as usize];
    for (rank, copy) in ranges(&ends).unwrap().into_iter().flatten().enumerate() {
        assert_eq!(ranks[copy as usize], usize::MAX);
        ranks[copy as usize] = rank;
    }
    assert!(!ranks.contains(&usize::MAX));
    starts.into_iter().map(|copy| ranks[copy as usize]).collect()
}

#[test]
fn a_full_receiving_pair_reorders_the_following_two_batches() {
    // Each pair shared a pixel in two signed F32 renders. One extra edge copy
    // changes which half-quadrant submits its remaining draws first.
    for (edges, before) in [(19, true), (20, false)] {
        let rank = draw_ranks(edges);
        for (left, right) in [(66560, 59050), (36589, 57785), (39011, 58236), (67164, 57437), (40323, 57678)] {
            assert_eq!(rank[left] < rank[right], before, "{edges} edge copies, draws {left}, {right}");
        }
    }
}

#[test]
fn receiving_tails_and_deferred_siblings_follow_the_full_batches() {
    let rank = draw_ranks(0);
    for (left, right, before) in [(29320, 59050, false), (2960, 72020, false), (50750, 18422, true), (3019, 29753, false)] {
        assert_eq!(rank[left] < rank[right], before, "draws {left}, {right}");
    }
}
