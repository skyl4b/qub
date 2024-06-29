use ndarray::prelude::*;
use num::{complex::Complex, Float};

use crate::qubit::Qubit;

/// A quantum gate is a unitary operator that acts on a qubit.
#[derive(Debug, PartialEq)]
pub struct QuGate<T: Float> {
    matrix: Array2<Complex<T>>,
}

impl<T: Float + 'static> QuGate<T> {
    /// Create a new quantum gate with the given matrix.
    pub fn new(matrix: Array2<Complex<T>>) -> Self {
        Self { matrix }
    }

    /// Apply the quantum gate to the given qubit.
    pub fn apply(&self, qubit: &Qubit<T>) -> Qubit<T> {
        Qubit {
            state: self.matrix.dot(&qubit.state),
        }
    }
}

impl<T: Float + 'static> QuGate<T> {
    /// Create a Pauli-X gate.
    pub fn pauli_x() -> Self {
        Self::new(array![
            [
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::one(), T::zero())
            ],
            [
                Complex::new(T::one(), T::zero()),
                Complex::new(T::zero(), T::zero())
            ],
        ])
    }

    /// Create a Pauli-Y gate.
    pub fn pauli_y() -> Self {
        Self::new(array![
            [
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::one())
            ],
            [
                Complex::new(T::zero(), T::one()),
                Complex::new(T::zero(), T::zero())
            ],
        ])
    }

    /// Create a Pauli-Z gate.
    pub fn pauli_z() -> Self {
        Self::new(array![
            [
                Complex::new(T::one(), T::zero()),
                Complex::new(T::zero(), T::zero())
            ],
            [
                Complex::new(T::zero(), T::zero()),
                Complex::new(-T::one(), T::zero())
            ],
        ])
    }

    /// Create a Hadamard gate.
    pub fn hadamard() -> Self {
        let norm_factor = T::one() / (T::from(2.0).unwrap().sqrt());
        Self::new(array![
            [
                Complex::new(norm_factor, T::zero()),
                Complex::new(norm_factor, T::zero())
            ],
            [
                Complex::new(norm_factor, T::zero()),
                Complex::new(-norm_factor, T::zero())
            ],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_upper_case_globals)]
    const i: Complex<f64> = Complex::I;

    #[test]
    fn pauli_x() {
        let qubit = Qubit::<f64>::zero();
        let x_gate = QuGate::pauli_x();

        assert_eq!(x_gate.apply(&qubit), Qubit::one());
    }

    #[test]
    fn pauli_y() {
        let qubit = Qubit::<f64>::zero();
        let y_gate = QuGate::pauli_y();

        assert_eq!(
            y_gate.apply(&qubit),
            Qubit::new(0.0 + 0.0 * i, 0.0 + 1.0 * i)
        );
    }

    #[test]
    fn pauli_z() {
        let qubit = Qubit::<f64>::zero();
        let z_gate = QuGate::pauli_z();

        assert_eq!(z_gate.apply(&qubit), Qubit::zero());
    }

    #[test]
    fn hadamard() {
        let qubit = Qubit::zero();
        let h_gate = QuGate::hadamard();
        let norm_factor = 1.0 / 2.0.sqrt();

        assert_eq!(
            h_gate.apply(&qubit),
            Qubit::new(norm_factor + 0.0 * i, norm_factor + 0.0 * i,)
        );
    }
}
