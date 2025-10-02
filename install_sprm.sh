#!/bin/bash

# SPRM Installation Script
# This script creates an environment named "SPRM" and installs all required dependencies
# It supports conda, pyenv, and falls back to venv + pip

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1-2
}

# Function to check if Python version is compatible
check_python_version() {
    local version=$1
    local major=$(echo $version | cut -d'.' -f1)
    local minor=$(echo $version | cut -d'.' -f2)
    
    if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
        return 0
    else
        return 1
    fi
}

# Function to install with conda
install_with_conda() {
    print_status "Using conda to create SPRM environment..."
    
    # Check if conda is available
    if ! command_exists conda; then
        print_error "conda command not found!"
        return 1
    fi
    
    # Check if environment already exists
    if conda env list | grep -q "^SPRM "; then
        print_warning "SPRM environment already exists. Removing it..."
        conda env remove -n SPRM -y
    fi
    
    # Create conda environment with Python 3.8+
    print_status "Creating conda environment 'SPRM' with Python 3.9..."
    conda create -n SPRM python=3.9 -y
    
    # Activate environment and install dependencies
    print_status "Installing dependencies in conda environment..."
    conda run -n SPRM pip install --upgrade pip
    conda run -n SPRM pip install .
    
    print_success "SPRM installed successfully using conda!"
    print_status "To activate the environment, run: conda activate SPRM"
}

# Function to install with pyenv
install_with_pyenv() {
    print_status "Using pyenv to create SPRM environment..."
    
    # Check if pyenv is available
    if ! command_exists pyenv; then
        print_error "pyenv command not found!"
        return 1
    fi
    
    # Check if pyenv-virtualenv is available
    if ! pyenv versions | grep -q "SPRM"; then
        # Install Python 3.9 if not available
        if ! pyenv versions | grep -q "3.9"; then
            print_status "Installing Python 3.9 via pyenv..."
            pyenv install 3.9.18
        fi
        
        # Create virtual environment
        print_status "Creating pyenv virtual environment 'SPRM'..."
        pyenv virtualenv 3.9.18 SPRM
    else
        print_warning "SPRM virtual environment already exists in pyenv"
    fi
    
    # Activate environment and install dependencies
    print_status "Installing dependencies in pyenv environment..."
    pyenv activate SPRM
    pip install --upgrade pip
    pip install .
    
    print_success "SPRM installed successfully using pyenv!"
    print_status "To activate the environment, run: pyenv activate SPRM"
}

# Function to install with venv (fallback)
install_with_venv() {
    print_status "Using venv to create SPRM environment (fallback method)..."
    
    # Check Python version
    local python_version=$(get_python_version)
    if ! check_python_version "$python_version"; then
        print_error "Python version $python_version is not compatible. SPRM requires Python 3.8+"
        return 1
    fi
    
    # Remove existing environment if it exists
    if [ -d "SPRM_env" ]; then
        print_warning "Removing existing SPRM_env directory..."
        rm -rf SPRM_env
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment 'SPRM_env'..."
    python3 -m venv SPRM_env
    
    # Activate environment and install dependencies
    print_status "Installing dependencies in virtual environment..."
    source SPRM_env/bin/activate
    pip install --upgrade pip
    pip install .
    
    print_success "SPRM installed successfully using venv!"
    print_status "To activate the environment, run: source SPRM_env/bin/activate"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying SPRM installation..."
    
    # Try to import SPRM
    if command_exists conda && conda env list | grep -q "^SPRM "; then
        conda run -n SPRM python -c "import sprm; print('SPRM import successful')" 2>/dev/null
    elif command_exists pyenv && pyenv versions | grep -q "SPRM"; then
        pyenv activate SPRM
        python -c "import sprm; print('SPRM import successful')" 2>/dev/null
    elif [ -d "SPRM_env" ]; then
        source SPRM_env/bin/activate
        python -c "import sprm; print('SPRM import successful')" 2>/dev/null
    else
        print_error "Could not verify SPRM installation"
        return 1
    fi
    
    print_success "SPRM installation verified successfully!"
}

# Main installation logic
main() {
    print_status "SPRM Installation Script"
    print_status "========================="
    print_status "This script will create an environment named 'SPRM' and install all dependencies."
    print_status ""
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ] || [ ! -d "sprm" ]; then
        print_error "Please run this script from the SPRM repository root directory (where setup.py is located)"
        exit 1
    fi
    
    # Try installation methods in order of preference
    if command_exists conda; then
        print_status "Found conda. Using conda for environment management..."
        install_with_conda
        verify_installation
    elif command_exists pyenv; then
        print_status "Found pyenv. Using pyenv for environment management..."
        install_with_pyenv
        verify_installation
    else
        print_warning "Neither conda nor pyenv found. Using venv as fallback..."
        install_with_venv
        verify_installation
    fi
    
    print_status ""
    print_success "Installation completed successfully!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Activate your environment:"
    if command_exists conda && conda env list | grep -q "^SPRM "; then
        print_status "   conda activate SPRM"
    elif command_exists pyenv && pyenv versions | grep -q "SPRM"; then
        print_status "   pyenv activate SPRM"
    elif [ -d "SPRM_env" ]; then
        print_status "   source SPRM_env/bin/activate"
    fi
    print_status "2. Run SPRM demo: cd demo && MPLBACKEND=Agg python ../SPRM.py --help"
    print_status "3. Or install the demo files and run: cd demo && sh run_sprm.sh"
    print_status ""
    print_status "Note: MPLBACKEND=Agg is set to avoid GUI threading issues on macOS"
}

# Run main function
main "$@"
