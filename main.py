import subprocess
import os
import time
from datetime import datetime
from pathlib import Path

from codecarbon import EmissionsTracker


def run(cfg_path, region="CAN", pue=1.2):
    """
    Run easydistill experiment with energy tracking.
    
    Args:
        cfg_path: Path to easydistill configuration file
        region: Country code for carbon intensity calculation
        pue: Power Usage Effectiveness multiplier
    """
    print(f"Starting experiment at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {cfg_path}")
    print(f"Region: {region}, PUE: {pue}")
    
    # Validate config file exists
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    
    start_time = time.time()
    
    try:
        with EmissionsTracker(country_iso_code=region, pue=pue) as tracker:
            print("Starting easydistill process...")
            result = subprocess.run(["easydistill", "--config", cfg_path], check=True, capture_output=True, text=True)
            print("easydistill completed successfully")
            
    except subprocess.CalledProcessError as e:
        print(f"easydistill failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    
    total_time = time.time() - start_time
    print(f"Experiment completed in {total_time/60:.2f} minutes")


def run_phase(cfg_path, phase_name, region="CAN", pue=1.2):
    """
    Run a specific phase of the experiment with energy tracking.
    
    Args:
        cfg_path: Path to easydistill configuration file
        phase_name: Name of the phase (e.g., "data_generation", "distillation", "evaluation")
        region: Country code for carbon intensity calculation
        pue: Power Usage Effectiveness multiplier
    """
    print(f"Starting {phase_name} phase at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create phase-specific output file
    output_dir = Path("energy_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{timestamp}_{phase_name}_emissions.csv"
    
    start_time = time.time()
    
    try:
        with EmissionsTracker(
            country_iso_code=region, 
            pue=pue,
            output_file=str(output_file),
            project_name=f"{phase_name}_{timestamp}"
        ) as tracker:
            print(f"Running {phase_name} phase...")
            result = subprocess.run(["easydistill", "--config", cfg_path], check=True, capture_output=True, text=True)
            print(f"{phase_name} phase completed successfully")
            
    except subprocess.CalledProcessError as e:
        print(f"{phase_name} phase failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error in {phase_name} phase: {e}")
        raise
    
    total_time = time.time() - start_time
    print(f"{phase_name} phase completed in {total_time/60:.2f} minutes")
    print(f"Energy data saved to: {output_file}")
    
    return str(output_file)


if __name__ == "__main__":
    # Example usage - you can modify these parameters as needed
    config_path = "configs/qwen/kd_black_box_q7b_to_q05b.json"
    region = "CAN"  # Change to your region (e.g., "USA", "DEU", "GBR")
    pue = 1.2      # Power Usage Effectiveness
    
    print("="*60)
    print("ENERGY TRACKING EXPERIMENT")
    print("="*60)
    
    # Option 1: Run complete experiment
    print("\nRunning complete experiment...")
    run(config_path, region=region, pue=pue)
    
    # Option 2: Run individual phases (uncomment to use)
    # print("\nRunning individual phases...")
    # phases = ["data_generation", "distillation", "evaluation"]
    # for phase in phases:
    #     run_phase(config_path, phase, region=region, pue=pue)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print("Energy consumption data saved to CSV files")
    print("Check the energy_results/ directory for detailed emissions data")

