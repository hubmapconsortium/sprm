#!/bin/bash

# SPRM Demo Runner Script
# This script automatically detects and activates the SPRM environment
# and runs the SPRM demo with the provided image and mask files

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

# Function to activate SPRM environment
activate_sprm_environment() {
    print_status "Detecting and activating SPRM environment..."
    
    # Check if we're in the demo directory
    if [ ! -f "../setup.py" ] || [ ! -d "../sprm" ]; then
        print_error "Please run this script from the demo directory"
        exit 1
    fi
    
    # Try conda first
    if command_exists conda && conda env list | grep -q "^SPRM "; then
        print_status "Found SPRM conda environment. Activating..."
        # Initialize conda for this shell session
        eval "$(conda shell.bash hook)"
        conda activate SPRM
        print_success "Activated conda environment 'SPRM'"
        return 0
    fi
    
    # Try pyenv
    if command_exists pyenv && pyenv versions | grep -q "SPRM"; then
        print_status "Found SPRM pyenv environment. Activating..."
        eval "$(pyenv init -)"
        pyenv activate SPRM
        print_success "Activated pyenv environment 'SPRM'"
        return 0
    fi
    
    # Try venv fallback
    if [ -d "../SPRM_env" ]; then
        print_status "Found SPRM venv environment. Activating..."
        source ../SPRM_env/bin/activate
        print_success "Activated venv environment 'SPRM_env'"
        return 0
    fi
    
    # Check if SPRM is installed globally
    if python3 -c "import sprm" 2>/dev/null; then
        print_warning "SPRM appears to be installed globally. Using system Python."
        return 0
    fi
    
    print_error "SPRM environment not found!"
    print_error "Please run the installation script first: ./install_sprm.sh"
    exit 1
}

# Function to check if demo files exist
check_demo_files() {
    print_status "Checking for demo files..."
    
    # Check for image file in img/ subdirectory
    if [ ! -f "img/image_demo.ome.tiff" ]; then
        print_error "Demo image file 'img/image_demo.ome.tiff' not found!"
        print_error "Please download demo files from:"
        print_error "https://drive.google.com/drive/folders/1denyZ1SFoWpWrPO9UbSdcF2DvHEv6ovN?usp=sharing"
        print_error "and place them in the img/ and mask/ subdirectories"
        exit 1
    fi
    
    # Check for mask file in mask/ subdirectory
    if [ ! -f "mask/mask_demo.ome.tiff" ]; then
        print_error "Demo mask file 'mask/mask_demo.ome.tiff' not found!"
        print_error "Please download demo files from:"
        print_error "https://drive.google.com/drive/folders/1denyZ1SFoWpWrPO9UbSdcF2DvHEv6ovN?usp=sharing"
        print_error "and place them in the img/ and mask/ subdirectories"
        exit 1
    fi
    
    print_success "Demo files found in img/ and mask/ directories"
}

# Main execution
main() {
    print_status "SPRM Demo Runner"
    print_status "================"
    
    # Define the path to the sprm executable
    SPRM_PATH="../SPRM.py"
    
    # Check if SPRM.py exists
    if [ ! -f "$SPRM_PATH" ]; then
        print_error "SPRM.py not found at $SPRM_PATH"
        print_error "Please ensure you're running this from the demo directory"
        exit 1
    fi
    
    # Create output directory
    if [ ! -e "sprm_demo_outputs" ]; then
        print_status "Creating output directory..."
        mkdir "sprm_demo_outputs"
    fi
    
    # Check demo files
    check_demo_files
    
    # Activate SPRM environment
    activate_sprm_environment
    
    # Verify SPRM can be imported
    print_status "Verifying SPRM installation..."
    if ! python3 -c "import sprm" 2>/dev/null; then
        print_error "SPRM import failed. Please check your installation."
        exit 1
    fi
    print_success "SPRM import successful"
    
    # Run SPRM
    print_status "Running SPRM demo..."
    print_status "Command: MPLBACKEND=Agg python -u $SPRM_PATH --img-dir img/image_demo.ome.tiff --mask-dir mask/mask_demo.ome.tiff --output-dir sprm_demo_outputs --processes 1"
    
    # Set matplotlib backend to Agg to avoid GUI threading issues on macOS
    export MPLBACKEND=Agg
    
    if python -u "$SPRM_PATH" --img-dir img/image_demo.ome.tiff --mask-dir mask/mask_demo.ome.tiff --output-dir sprm_demo_outputs --processes 1 > sprm_demo_outputs/sprm_demo_outputs.log 2>&1; then
        print_success "SPRM demo completed successfully!"
        print_status "Results saved to: sprm_demo_outputs/"
        print_status "Log file: sprm_demo_outputs/sprm_demo_outputs.log"
    else
        print_error "SPRM demo failed. Check the log file for details:"
        print_error "sprm_demo_outputs/sprm_demo_outputs.log"
        exit 1
    fi
}

# Run main function
main "$@"
