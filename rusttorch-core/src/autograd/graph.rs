//! Computation graph traversal and backward pass.
//!
//! Implements reverse-mode automatic differentiation via topological sort
//! of the computation graph followed by gradient propagation.

use crate::autograd::variable::Variable;
use crate::error::Result;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Compute gradients for all leaf variables by backpropagating from `root`.
///
/// The root must be a scalar (single-element) variable.
/// Gradients are accumulated on all leaf variables with `requires_grad = true`.
pub fn backward(root: &Variable) -> Result<()> {
    // Build topological ordering via DFS
    let topo_order = topological_sort(root);

    // Initialize gradient map: root gets gradient of 1.0
    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    grads.insert(root.id(), Tensor::from_vec(vec![1.0], &[1]));

    // Propagate gradients in reverse topological order
    for var in &topo_order {
        let var_id = var.id();
        let grad_output = match grads.get(&var_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flows to this node
        };

        let inner = var.inner.borrow();

        // If this is a leaf, just keep the accumulated gradient
        if inner.is_leaf {
            continue;
        }

        // If this node has a grad_fn, propagate gradients to inputs
        if let Some(ref grad_fn) = inner.grad_fn {
            let inputs = grad_fn.inputs();
            let input_grads = grad_fn.backward(&grad_output)?;

            for (input_var, maybe_grad) in inputs.iter().zip(input_grads.iter()) {
                if let Some(grad) = maybe_grad {
                    let input_id = input_var.id();
                    match grads.get(&input_id) {
                        Some(existing) => {
                            let accumulated = crate::ops::add(existing, grad)?;
                            grads.insert(input_id, accumulated);
                        }
                        None => {
                            grads.insert(input_id, grad.clone());
                        }
                    }
                }
            }
        }
    }

    // Store final gradients on leaf variables
    for var in &topo_order {
        let is_leaf = var.inner.borrow().is_leaf;
        let requires_grad = var.inner.borrow().requires_grad;
        if is_leaf && requires_grad {
            if let Some(grad) = grads.remove(&var.id()) {
                var.accumulate_grad(&grad)?;
            }
        }
    }

    Ok(())
}

/// Topological sort of the computation graph rooted at `root`.
///
/// Returns nodes in reverse dependency order (root first, leaves last).
/// This is the order needed for backpropagation.
fn topological_sort(root: &Variable) -> Vec<Variable> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    topo_dfs(root, &mut visited, &mut order);
    order
}

fn topo_dfs(var: &Variable, visited: &mut HashSet<usize>, order: &mut Vec<Variable>) {
    let id = var.id();
    if visited.contains(&id) {
        return;
    }
    visited.insert(id);

    // First, add this node (we want reverse topological order — root first)
    order.push(var.clone());

    // Then visit inputs
    let inner = var.inner.borrow();
    if let Some(ref grad_fn) = inner.grad_fn {
        let inputs = grad_fn.inputs();
        drop(inner); // Release borrow before recursing
        for input in &inputs {
            topo_dfs(input, visited, order);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_topological_sort_simple() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2]), true);
        let c = a.add(&b).unwrap();
        let d = c.sum().unwrap();

        let order = topological_sort(&d);
        // d -> c -> a, b (root first, leaves last)
        assert_eq!(order.len(), 4);
        assert_eq!(order[0].id(), d.id());
    }

    #[test]
    fn test_topological_sort_diamond() {
        // Diamond: a -> b, a -> c, b+c -> d
        let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]), true);
        let b = a.mul_scalar(3.0);
        let c = a.mul_scalar(5.0);
        let d = b.add(&c).unwrap();

        let order = topological_sort(&d);
        // d should be first, a should appear once
        assert_eq!(order[0].id(), d.id());
        let unique_ids: HashSet<usize> = order.iter().map(|v| v.id()).collect();
        assert_eq!(unique_ids.len(), order.len()); // No duplicates
    }
}
