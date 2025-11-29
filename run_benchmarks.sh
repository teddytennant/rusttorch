#!/bin/bash
#
# RustTorch Benchmark Runner
# Author: Theodore Tennant (@teddytennant)
#
# This script runs all benchmarks and generates performance reports
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo " RustTorch Benchmark Suite"
echo "================================================================================"
echo ""

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust toolchain."
    echo "Visit: https://rustup.rs/"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

# Function to run Rust benchmarks
run_rust_benchmarks() {
    echo "Running Rust Benchmarks (Criterion)"
    echo "--------------------------------------------------------------------------------"
    cd rusttorch-core

    # Check if this is a baseline run
    if [ "$1" == "--save-baseline" ]; then
        echo "Saving baseline as: $2"
        cargo bench --bench tensor_ops -- --save-baseline "$2"
    elif [ "$1" == "--baseline" ]; then
        echo "Comparing against baseline: $2"
        cargo bench --bench tensor_ops -- --baseline "$2"
    else
        echo "Running benchmarks (no baseline comparison)"
        cargo bench --bench tensor_ops
    fi

    cd ..
    echo ""
    echo "Rust benchmark results saved to: rusttorch-core/target/criterion/"
    echo "Open rusttorch-core/target/criterion/report/index.html to view detailed results"
    echo ""
}

# Function to run Python comparison benchmarks
run_python_benchmarks() {
    echo "Running Python Comparison Benchmarks (RustTorch vs PyTorch)"
    echo "--------------------------------------------------------------------------------"

    # Check if PyTorch is installed
    if ! python3 -c "import torch" &> /dev/null; then
        echo "Warning: PyTorch not installed. Installing..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi

    cd benchmarks
    python3 compare_pytorch.py
    cd ..
    echo ""
}

# Function to generate profiling report
generate_profile() {
    echo "Generating CPU Profile with perf"
    echo "--------------------------------------------------------------------------------"

    if ! command -v perf &> /dev/null; then
        echo "Warning: perf not found. Install with: sudo apt-get install linux-tools-generic"
        echo "Skipping profiling..."
        return
    fi

    cd rusttorch-core
    echo "Profiling tensor operations..."
    perf record --call-graph=dwarf cargo bench --bench tensor_ops -- add/1000 2>&1 | head -20
    perf report --stdio > ../profile_report.txt 2>&1 || true
    cd ..

    if [ -f profile_report.txt ]; then
        echo "Profile report saved to: profile_report.txt"
    fi
    echo ""
}

# Function to check for Rust updates
check_dependencies() {
    echo "Checking Dependencies"
    echo "--------------------------------------------------------------------------------"
    echo "Rust version:    $(rustc --version)"
    echo "Cargo version:   $(cargo --version)"
    echo "Python version:  $(python3 --version)"

    if python3 -c "import torch" &> /dev/null; then
        python3 -c "import torch; print('PyTorch version: ' + torch.__version__)"
    else
        echo "PyTorch version: Not installed"
    fi

    if python3 -c "import rusttorch" &> /dev/null; then
        echo "RustTorch:       Installed"
    else
        echo "RustTorch:       Not installed (build with: cd rusttorch-py && maturin develop --release)"
    fi
    echo ""
}

# Parse command line arguments
case "$1" in
    --rust)
        shift
        run_rust_benchmarks "$@"
        ;;
    --python)
        run_python_benchmarks
        ;;
    --profile)
        generate_profile
        ;;
    --check)
        check_dependencies
        ;;
    --all)
        check_dependencies
        run_rust_benchmarks
        run_python_benchmarks
        ;;
    --help)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --rust                    Run Rust benchmarks only"
        echo "  --rust --save-baseline NAME   Save benchmark baseline"
        echo "  --rust --baseline NAME        Compare against baseline"
        echo "  --python                  Run Python comparison benchmarks"
        echo "  --profile                 Generate CPU profile with perf"
        echo "  --check                   Check dependencies and versions"
        echo "  --all                     Run all benchmarks (default)"
        echo "  --help                    Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                        # Run all benchmarks"
        echo "  $0 --rust --save-baseline main   # Save baseline"
        echo "  $0 --rust --baseline main        # Compare to baseline"
        echo "  $0 --python               # Run Python benchmarks only"
        echo ""
        exit 0
        ;;
    *)
        check_dependencies
        run_rust_benchmarks
        run_python_benchmarks
        ;;
esac

echo "================================================================================"
echo " Benchmarks Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  - Review Rust benchmark results in: rusttorch-core/target/criterion/report/index.html"
echo "  - See PERFORMANCE.md for detailed performance guide"
echo "  - Run with --profile to identify hotspots"
echo ""
