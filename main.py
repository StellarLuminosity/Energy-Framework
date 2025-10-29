import subprocess

from codecarbon import EmissionsTracker


def run(cfg_path, region="CAN", pue=1.2):
    with EmissionsTracker(country_iso_code=region, pue=pue) as tracker:
        subprocess.run(["easydistill", "--config", cfg_path], check=True)


if __name__ == "__main__":
    run("configs/qwen/kd_black_box_q7b_to_q05b.json")
    # tracker.stop() called automatically; kWh & gCO2e saved to output CSV
    # captures data synthesis, distillation, and eval as separate runs if you invoke them separately

