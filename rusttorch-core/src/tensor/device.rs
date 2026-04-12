//! Device abstraction — where a tensor's data lives.
//!
//! This is the first pillar of the multi-backend refactor. All existing
//! ops currently run on `Device::Cpu` because storage is still backed by
//! ndarray, but every tensor now carries a `Device` tag so that call sites
//! can start distinguishing CPU from GPU tensors. As kernels are migrated
//! to a backend trait, dispatch will key off this field.
//!
//! GPU support is feature-gated behind `--features cuda`; enabling it
//! surfaces the `Cuda(device_id)` variant and the `to()` method starts
//! accepting CUDA devices (once kernels exist). Without the feature the
//! enum only has `Cpu`, so CPU-only builds are not bloated with GPU code.

/// Physical device where a tensor's storage lives.
///
/// The current implementation only supports `Cpu`. The `Cuda` variant is
/// gated behind the `cuda` feature and exists so that downstream API
/// contracts (`Tensor::device()`, `Tensor::to()`, the `Backend` trait) can
/// be stabilized before real CUDA kernels land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host CPU. This is the default and the only fully-supported device
    /// today.
    #[default]
    Cpu,

    /// NVIDIA CUDA device at the given ordinal. Feature-gated.
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Device {
    /// Shorthand constructor for the CPU device.
    #[inline]
    pub const fn cpu() -> Self {
        Device::Cpu
    }

    /// Shorthand constructor for a CUDA device at ordinal `id`. Only
    /// available with `--features cuda`.
    #[cfg(feature = "cuda")]
    #[inline]
    pub const fn cuda(id: usize) -> Self {
        Device::Cuda(id)
    }

    /// True if this device is the host CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// True if this device is a CUDA device. Always false when the `cuda`
    /// feature is disabled.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, Device::Cuda(_))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => write!(f, "cuda:{id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn cpu_predicates() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_cuda());
    }

    #[test]
    fn cpu_display() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
    }

    #[test]
    fn cpu_equality_and_copy() {
        let a = Device::cpu();
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn cpu_debug_format() {
        assert_eq!(format!("{:?}", Device::Cpu), "Cpu");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_variant_predicates() {
        let d = Device::cuda(0);
        assert!(!d.is_cpu());
        assert!(d.is_cuda());
        assert_eq!(format!("{d}"), "cuda:0");
    }
}
