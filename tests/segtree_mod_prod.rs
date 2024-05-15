use segtree::{query::{Mod, ProdQuery}, SegTree};


#[test]
fn mod_prod_query() {
    let segtree = SegTree::from_iter_query(Mod::new(ProdQuery, 221u32), 1u32..=20);
    assert_eq!(segtree.query(0..20), 0);
    assert_eq!(segtree.query(0..10), 201);
    assert_eq!(segtree.query(10..20), 0);
    assert_eq!(segtree.query(5..15), 52);
    assert_eq!(segtree.query(1..4), 24);
    assert_eq!(segtree.query(1..1), 1);
}