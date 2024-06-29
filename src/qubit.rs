use ndarray::prelude::*;
use num::{complex::Complex, Float};
use rand::prelude::*;

/// A qubit is a quantum bit.
/// It is a two-level quantum system that can be in a superposition of the |0⟩ and |1⟩ states.
/// The state of a qubit is described by a complex vector of size 2.
#[derive(Debug, PartialEq)]
pub struct Qubit<T: Float> {
    /// The state of the qubit, represented as a complex vector of size 2.
    pub(crate) state: Array1<Complex<T>>,
}

impl<T: Float> Qubit<T> {
    /// Create a new qubit with alpha amplitude to the |0⟩
    /// and beta amplitude to the |1⟩ state.
    pub fn new(alpha: Complex<T>, beta: Complex<T>) -> Self {
        Self {
            state: array![alpha, beta],
        }
    }

    /// Create a new qubit in the |0⟩ state.
    pub fn zero() -> Self {
        Self::new(
            Complex::new(T::one(), T::zero()),
            Complex::new(T::zero(), T::zero()),
        )
    }

    /// Create a new qubit in the |1⟩ state.
    pub fn one() -> Self {
        Self::new(
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::one(), T::zero()),
        )
    }

    /// Get the current state of the qubit.
    pub fn get_state(&self) -> &Array1<Complex<T>> {
        &self.state
    }

    /// Get the current amplitudes of the |0> state.
    pub fn alpha(&self) -> Complex<T> {
        self.state[0]
    }

    /// Get the current amplitudes of the |1> state.
    pub fn beta(&self) -> Complex<T> {
        self.state[1]
    }

    /// Get the probabilities of the qubit being in the |0⟩ and |1⟩ states.
    pub fn probabilities(&self) -> (T, T) {
        let p = self.state.mapv(|x| x.norm_sqr());

        (p[0], p[1])
    }

    /// Get the probabilities of the qubit being in the |0⟩ state.
    pub fn zero_probability(&self) -> T {
        self.probabilities().0
    }

    /// Get the probabilities of the qubit being in the |1⟩ state.
    pub fn one_probability(&self) -> T {
        self.probabilities().1
    }

    /// Validate the qubit state.
    pub fn validate(&self) -> bool {
        let (p0, p1) = self.probabilities();
        p0 + p1 == T::one()
    }

    /// Measure the qubit in the computational basis.
    /// Collapse the qubit to either the |0⟩ or |1⟩ state.
    pub fn measure(&self) -> Self {
        if random::<f64>() < self.zero_probability().to_f64().unwrap() {
            Self::zero()
        } else {
            Self::one()
        }
    }
}

impl<T: Float> Default for Qubit<T> {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_upper_case_globals)]
    const i: Complex<f64> = Complex::I;

    #[test]
    fn create_qubit() {
        let qubit = Qubit::new(0.5 + 0.0 * i, 0.5 + 0.0 * i);
        assert_eq!(qubit.alpha(), 0.5 + 0.0 * i);
        assert_eq!(qubit.beta(), 0.5 + 0.0 * i);
    }

    #[test]
    fn zero() {
        let qubit = Qubit::<f64>::zero();
        assert_eq!(qubit, qubit.measure());
    }

    #[test]
    fn one() {
        let qubit = Qubit::<f64>::one();
        assert_eq!(qubit, qubit.measure());
    }
}
