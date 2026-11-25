//! Dispatches depend on earlier writes to overlapping bytes and on earlier reads of bytes they overwrite. Atomic OR operations on the error flag commute. A kernel in [`FLAG_READERS`] also loads its flag, so it depends on operations that share its slot. A legacy trace without slot ranges treats the whole flag buffer as shared.

use crate::trace::{Access, Batch, Dispatch, Range, Target};

/// Kernels that load the error flag (`atomic_load_explicit(error, ...)`).
pub const FLAG_READERS: [&str; 6] = [
    "qsa_score_accumulate",
    "qsa_chronological_factors",
    "qsa_chronological_channels",
    "qsa_chronological_serial_factors",
    "qsa_verify_rows_factors",
    "qsa_verify_rows_channels",
];

pub fn reads_flag(dispatch: &Dispatch) -> bool {
    FLAG_READERS.contains(&dispatch.function())
}

fn flag_conflicts(a: Target, b: Target) -> bool {
    match (a, b) {
        (Target::ErrorFlag, Target::ErrorFlag | Target::ErrorFlagSlot { .. })
        | (Target::ErrorFlagSlot { .. }, Target::ErrorFlag) => true,
        (
            Target::ErrorFlagSlot {
                offset: a_off,
                bytes: a_bytes,
            },
            Target::ErrorFlagSlot {
                offset: b_off,
                bytes: b_bytes,
            },
        ) => a_off < b_off.saturating_add(b_bytes) && b_off < a_off.saturating_add(a_bytes),
        _ => false,
    }
}

fn span_conflicts(
    first: Range,
    first_access: Access,
    second: Range,
    second_access: Access,
) -> bool {
    (first_access == Access::Write || second_access == Access::Write) && first.overlaps(&second)
}

/// Whether `later` must stay ordered after `earlier`.
pub fn conflicts(earlier: &Dispatch, later: &Dispatch) -> bool {
    // Every flag-bound kernel writes its assigned flag slot; ordering matters only when one side also reads an overlapping slot.
    if reads_flag(earlier) || reads_flag(later) {
        for a in &earlier.bindings {
            if !matches!(a.target, Target::ErrorFlag | Target::ErrorFlagSlot { .. }) {
                continue;
            }
            for b in &later.bindings {
                if flag_conflicts(a.target, b.target) {
                    return true;
                }
            }
        }
    }
    for a in &earlier.bindings {
        let Target::Range(first, first_access) = a.target else {
            continue;
        };
        for b in &later.bindings {
            let Target::Range(second, second_access) = b.target else {
                continue;
            };
            if span_conflicts(first, first_access, second, second_access) {
                return true;
            }
        }
        if let Some(second) = later.indirect
            && span_conflicts(first, first_access, second, Access::Read)
        {
            return true;
        }
    }
    if let Some(first) = earlier.indirect {
        for b in &later.bindings {
            let Target::Range(second, second_access) = b.target else {
                continue;
            };
            if span_conflicts(first, Access::Read, second, second_access) {
                return true;
            }
        }
    }
    false
}

/// Dependency edges `(earlier, later)` between dispatch positions of a batch in its own order; not transitively reduced.
pub fn edges(dispatches: &[&Dispatch]) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    for later in 0..dispatches.len() {
        for earlier in 0..later {
            if conflicts(dispatches[earlier], dispatches[later]) {
                found.push((earlier, later));
            }
        }
    }
    found
}

/// The earliest level each dispatch can run at: one more than the highest level among the dispatches it depends on.
pub fn levels(count: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut level = vec![0; count];
    // Edges are generated with `earlier < later`, so one pass in `later` order sees every predecessor's final level.
    let mut sorted = edges.to_vec();
    sorted.sort_by_key(|&(earlier, later)| (later, earlier));
    for (earlier, later) in sorted {
        level[later] = level[later].max(level[earlier] + 1);
    }
    level
}

/// A batch's dependency shape: how much concurrency the data allow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dispatches: usize,
    pub barriers: usize,
    pub edges: usize,
    /// Dispatches on the longest dependency chain.
    pub depth: usize,
    /// The widest level of the as-soon-as-possible schedule.
    pub widest: usize,
}

pub fn shape(batch: &Batch) -> Shape {
    let dispatches: Vec<&Dispatch> = batch.dispatches().collect();
    let found = edges(&dispatches);
    let level = levels(dispatches.len(), &found);
    let depth = level.iter().max().map_or(0, |deepest| deepest + 1);
    let mut width = vec![0; depth];
    for &at in &level {
        width[at] += 1;
    }
    Shape {
        dispatches: dispatches.len(),
        barriers: batch.barriers(),
        edges: found.len(),
        depth,
        widest: width.into_iter().max().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Binding, Range};

    fn dispatch_with_flag(
        function: &str,
        flag_target: Target,
        bindings: &[(u64, u64, u64, Access)],
    ) -> Dispatch {
        let mut all = vec![Binding {
            index: 7,
            target: flag_target,
        }];
        for (index, &(slot, offset, bytes, access)) in bindings.iter().enumerate() {
            all.push(Binding {
                index: index as u64,
                target: Target::Range(
                    Range {
                        slot,
                        offset,
                        bytes,
                    },
                    access,
                ),
            });
        }
        Dispatch {
            pipeline: format!("{function}:4294967295"),
            aligned: true,
            grid: [1, 1, 1],
            group: [128, 1, 1],
            bindings: all,
            constants: Vec::new(),
            indirect: None,
        }
    }

    fn dispatch(function: &str, bindings: &[(u64, u64, u64, Access)]) -> Dispatch {
        dispatch_with_flag(function, Target::ErrorFlag, bindings)
    }

    #[test]
    fn writes_order_overlapping_reads_and_writes_only() {
        // Arrange
        let writer = dispatch("a", &[(0, 0, 16, Access::Write)]);
        let overlapping_reader = dispatch("b", &[(0, 12, 8, Access::Read)]);
        let adjacent_reader = dispatch("c", &[(0, 16, 8, Access::Read)]);
        let other_reader = dispatch("d", &[(0, 0, 16, Access::Read)]);

        // Act / Assert
        assert!(conflicts(&writer, &overlapping_reader));
        assert!(conflicts(&overlapping_reader, &writer));
        assert!(!conflicts(&writer, &adjacent_reader));
        assert!(!conflicts(&overlapping_reader, &other_reader));
        assert!(conflicts(
            &writer,
            &dispatch("e", &[(0, 0, 4, Access::Write)])
        ));
    }

    #[test]
    fn the_shared_error_flag_orders_only_its_readers() {
        // Arrange
        let first = dispatch("a", &[(0, 0, 4, Access::Write)]);
        let second = dispatch("b", &[(1, 0, 4, Access::Write)]);
        let reader = dispatch("qsa_score_accumulate", &[(2, 0, 4, Access::Write)]);

        // Act / Assert
        assert!(!conflicts(&first, &second));
        assert!(conflicts(&first, &reader));
        assert!(conflicts(&reader, &second));
    }

    #[test]
    fn levels_follow_the_longest_chain() {
        // Arrange
        let a = dispatch("a", &[(0, 0, 4, Access::Write)]);
        let b = dispatch("b", &[(1, 0, 4, Access::Write)]);
        let c = dispatch("c", &[(0, 0, 4, Access::Read), (2, 0, 4, Access::Write)]);
        let d = dispatch("d", &[(2, 0, 4, Access::Read), (1, 0, 4, Access::Read)]);
        let all = [&a, &b, &c, &d];

        // Act
        let found = edges(&all);
        let level = levels(all.len(), &found);

        // Assert
        assert_eq!(found, vec![(0, 2), (1, 3), (2, 3)]);
        assert_eq!(level, vec![0, 0, 1, 2]);
    }

    #[test]
    fn disjoint_error_flag_slots_decouple_flag_readers() {
        let writer_slot0 = dispatch_with_flag(
            "rms_norm",
            Target::ErrorFlagSlot {
                offset: 0,
                bytes: 4,
            },
            &[(0, 0, 16, Access::Write)],
        );
        let qsa_pass1 = dispatch_with_flag(
            "qsa_score_stage",
            Target::ErrorFlagSlot {
                offset: 4,
                bytes: 4,
            },
            &[(1, 0, 16, Access::Write)],
        );
        let qsa_pass2_reader = dispatch_with_flag(
            "qsa_score_accumulate",
            Target::ErrorFlagSlot {
                offset: 4,
                bytes: 4,
            },
            &[(2, 0, 16, Access::Write)],
        );
        let legacy_writer = dispatch("rms_norm", &[(3, 0, 16, Access::Write)]);

        // Disjoint ErrorFlagSlot ranges do not conflict even with FLAG_READERS.
        assert!(!conflicts(&writer_slot0, &qsa_pass2_reader));
        assert!(!conflicts(&qsa_pass2_reader, &writer_slot0));
        // Same ErrorFlagSlot orders reader with writer on that slot.
        assert!(conflicts(&qsa_pass1, &qsa_pass2_reader));
        assert!(conflicts(&qsa_pass2_reader, &qsa_pass1));
        // Legacy Target::ErrorFlag conservatively conflicts with any ErrorFlagSlot when a reader is present.
        assert!(conflicts(&legacy_writer, &qsa_pass2_reader));
        assert!(!conflicts(&legacy_writer, &writer_slot0));
    }

    #[test]
    fn indirect_dispatch_conflicts_on_12_byte_args_span() {
        let writer = dispatch("prep", &[(0, 16, 12, Access::Write)]);
        let mut indirect_consumer = dispatch("run", &[(1, 0, 64, Access::Write)]);
        indirect_consumer.indirect = Some(Range {
            slot: 0,
            offset: 16,
            bytes: 12,
        });
        let mut indirect_disjoint = dispatch("run", &[(1, 0, 64, Access::Write)]);
        indirect_disjoint.indirect = Some(Range {
            slot: 0,
            offset: 28,
            bytes: 12,
        });

        assert!(conflicts(&writer, &indirect_consumer));
        assert!(!conflicts(&writer, &indirect_disjoint));
    }
}
