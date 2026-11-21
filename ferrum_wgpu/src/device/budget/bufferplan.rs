use super::*;

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::TooLarge { name, bytes, limit } => {
                write!(
                    f,
                    "buffer '{name}' needs {bytes} bytes, adapter binding limit is {limit}"
                )
            }
            PlanError::OverBudget { need, have } => {
                write!(f, "needs {need} bytes of GPU memory, budget is {have}")
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// Every buffer an engine will create, listed before any is created.
#[derive(Default, Debug)]
pub struct BufferPlan {
    pub(super) rows: Vec<(String, u64)>,
}

impl BufferPlan {
    /// Add a buffer to the plan.
    pub fn add(&mut self, name: &str, bytes: u64) {
        self.rows.push((name.to_string(), bytes));
    }

    /// The planned buffers as `(name, bytes)` rows, in insertion order.
    pub fn rows(&self) -> &[(String, u64)] {
        &self.rows
    }

    /// Sum of all planned bytes.
    pub fn total(&self) -> u64 {
        self.rows.iter().map(|r| r.1).sum()
    }

    /// Refuse (before anything is allocated) if any buffer exceeds `max_binding` or the sum
    /// exceeds `budget` (`None` budget = unknown, only the binding limit is enforced).
    pub fn check(&self, max_binding: u64, budget: Option<Budget>) -> Result<(), PlanError> {
        if let Some((name, bytes)) = self.rows.iter().find(|r| r.1 > max_binding) {
            return Err(PlanError::TooLarge {
                name: name.clone(),
                bytes: *bytes,
                limit: max_binding,
            });
        }
        match budget {
            Some(b) if self.total() > b.bytes => Err(PlanError::OverBudget {
                need: self.total(),
                have: b.bytes,
            }),
            _ => Ok(()),
        }
    }
}
