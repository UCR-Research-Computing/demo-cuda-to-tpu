#!/usr/bin/env python3
import subprocess
import time
import re
import threading
import sys
from pathlib import Path
from typing import Generator, Optional, Any
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.syntax import Syntax

console = Console()

# Configuration
PROJECT = "ucr-research-computing"
GPU_VM = "demo-gpu-a100"
GPU_ZONE = "us-central1-a"
TPU_VM = "demo-tpu-v5e"
TPU_ZONE = "us-west1-c"

# Locate Payloads
BASE_DIR = Path(__file__).parent
PAYLOAD_DIR = BASE_DIR / "payloads"
LEGACY_PAYLOAD = PAYLOAD_DIR / "train_legacy.py"
JAX_PAYLOAD = PAYLOAD_DIR / "train_jax.py"

def header(title: str) -> None:
    console.clear()
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", style="white", expand=False))

def step(title: str, description: str, command_preview: Optional[str] = None) -> None:
    console.print(f"\n[bold yellow]Step:[/bold yellow] [bold]{title}[/bold]")
    console.print(description)
    if command_preview:
        console.print("\n[bold dim]Command to be executed:[/bold dim]")
        syntax = Syntax(command_preview, "bash", theme="monokai", line_numbers=False)
        console.print(syntax)
    
    console.input("\n[dim]Press Enter to execute...[/dim]")

def wait_for_ssh(vm: str, zone: str, is_tpu: bool = False) -> bool:
    """Loops until SSH is available."""
    attempts = 0
    while attempts < 40:  # Increase to ~3.5 minutes
        if is_tpu:
            cmd = f"gcloud compute tpus tpu-vm ssh {vm} --project={PROJECT} --zone={zone} --quiet --command='echo ready'"
        else:
            cmd = f"gcloud compute ssh {vm} --project={PROJECT} --zone={zone} --quiet --command='echo ready'"
            
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return True
        time.sleep(5)
        attempts += 1
    return False

def run_ssh_command(vm: str, zone: str, command: str, is_tpu: bool = False) -> Generator[str, None, None]:
    """Executes SSH command and yields output line by line."""
    if is_tpu:
        base_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh"]
    else:
        base_cmd = ["gcloud", "compute", "ssh"]

    full_cmd = base_cmd + [
        vm,
        f"--project={PROJECT}",
        f"--zone={zone}",
        "--quiet",
        "--command", command
    ]
    process = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            yield line.strip()
    process.wait()

def create_dashboard(gpu_status: str, tpu_status: str, progress_group: Progress) -> Table:
    """Creates the dashboard layout."""
    # Main grid: 2 columns
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    
    # Top Row: GPU and TPU Panels
    grid.add_row(
        Panel(gpu_status, title="🔥 NVIDIA A100 (Legacy)", border_style="green", height=12),
        Panel(tpu_status, title="⚡ Google TPU v5e (Target)", border_style="blue", height=12)
    )
    
    # Bottom Row: Progress (Single row, let it span by creating a new table or using a layout)
    # Using a container table for the progress to simplify layout
    container = Table.grid(expand=True)
    container.add_column()
    container.add_row(grid)
    container.add_row(Panel(progress_group, title="⏱️ Simulation Progress", border_style="yellow"))
    
    return container

def cleanup() -> None:
    header("🧹 Cleaning Up Resources")
    console.print("[bold yellow]Destroying Infrastructure...[/bold yellow]")
    
    def nuke_gpu() -> None:
        check = subprocess.run(f"gcloud compute instances describe {GPU_VM} --project={PROJECT} --zone={GPU_ZONE}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode != 0:
            console.print(f"[dim]{GPU_VM} not found, skipping.[/dim]")
            return

        console.print(f"Deleting {GPU_VM}...")
        subprocess.run(f"gcloud compute instances delete {GPU_VM} --project={PROJECT} --zone={GPU_ZONE} --quiet", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"[bold red]Deleted[/bold red] {GPU_VM}")

    def nuke_tpu() -> None:
        check = subprocess.run(f"gcloud compute tpus tpu-vm describe {TPU_VM} --project={PROJECT} --zone={TPU_ZONE}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode != 0:
            console.print(f"[dim]{TPU_VM} not found, skipping.[/dim]")
            return

        console.print(f"Deleting {TPU_VM}...")
        subprocess.run(f"gcloud compute tpus tpu-vm delete {TPU_VM} --project={PROJECT} --zone={TPU_ZONE} --quiet", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"[bold red]Deleted[/bold red] {TPU_VM}")

    t1 = threading.Thread(target=nuke_gpu)
    t2 = threading.Thread(target=nuke_tpu)
    t1.start()
    t2.start()
    
    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        console.print("\n[bold red]Cleanup Interrupted! Resources may still exist.[/bold red]")
        sys.exit(1)
        
    console.print("\n[bold green]✨ Cleanup Complete.[/bold green]")

def run() -> None:
    print("ursa-major-demo-cuda-to-tpu v0.3.4")
    if "--version" in sys.argv:
        return

    if "--cleanup" in sys.argv:
        cleanup()
        return

    header("🚀 CUDA to TPU: The Accelerator Race")

    # --- SCENE 1: The Context & Conversion ---
    step("Context: The 'Legacy' Training Loop", 
    """
    We start with a standard [bold red]PyTorch[/bold red] training script ([cyan]train_legacy.py[/cyan]).
    It trains a heavy **ResNet-50** on ImageNet-sized inputs using explicit CUDA device management.
    """)
    
    # ... (reading code) ...

    step("The Result", 
    """
    The PyTorch code has been transformed into a [bold blue]JAX/Flax[/bold blue] training loop.
    This enables XLA compilation and native TPU execution with massive throughput for **ResNet-50**.
    """)

    # --- SCENE 2: Concurrent Provisioning ---
    gpu_create_cmd = f"gcloud compute instances create {GPU_VM} --project={PROJECT} --zone={GPU_ZONE} --machine-type=a2-highgpu-1g --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --quiet"
    tpu_create_cmd = f"gcloud compute tpus tpu-vm create {TPU_VM} --project={PROJECT} --zone={TPU_ZONE} --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite --quiet"

    step("Provisioning the Iron", 
    """
    Simultaneously summoning [bold green]Nvidia A100[/bold green] and [bold blue]Google TPU v5e[/bold blue]...
    """, command_preview=f"{gpu_create_cmd}\n{tpu_create_cmd}")

    provisioning_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[status]}")
    )
    
    gpu_p_task = provisioning_progress.add_task("[green]Nvidia A100 VM", total=100, status="Starting...")
    tpu_p_task = provisioning_progress.add_task("[blue]Google TPU VM", total=100, status="Starting...")

    provisioning_results = {"gpu": False, "tpu": False}

    def run_with_retry(cmd: str, task_id: Any, task_name: str) -> bool:
        max_retries = 3
        last_error = ""
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                provisioning_progress.update(task_id, status=f"Retrying ({attempt}/{max_retries})...")
                time.sleep(5)
            
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                return True
            last_error = res.stderr
        
        console.print(f"\n[bold red]Error provisioning {task_name}:[/bold red]\n{last_error.strip()}")
        return False

    def do_gpu() -> None:
        # Check if exists
        check = subprocess.run(f"gcloud compute instances describe {GPU_VM} --project={PROJECT} --zone={GPU_ZONE}", shell=True, capture_output=True)
        if check.returncode != 0:
            provisioning_progress.update(gpu_p_task, status="Creating...")
            if not run_with_retry(gpu_create_cmd, gpu_p_task, "GPU"):
                provisioning_progress.update(gpu_p_task, completed=100, status="[bold red]FAILED[/bold red]")
                return
        
        provisioning_results["gpu"] = True
        provisioning_progress.update(gpu_p_task, completed=100, status="[bold green]ONLINE[/bold green]")

    def do_tpu() -> None:
        # Check if exists
        check = subprocess.run(f"gcloud compute tpus tpu-vm describe {TPU_VM} --project={PROJECT} --zone={TPU_ZONE}", shell=True, capture_output=True)
        if check.returncode != 0:
            provisioning_progress.update(tpu_p_task, status="Creating...")
            if not run_with_retry(tpu_create_cmd, tpu_p_task, "TPU"):
                provisioning_progress.update(tpu_p_task, completed=100, status="[bold red]FAILED[/bold red]")
                return
        
        provisioning_results["tpu"] = True
        provisioning_progress.update(tpu_p_task, completed=100, status="[bold green]ONLINE[/bold green]")

    t_gpu = threading.Thread(target=do_gpu)
    t_tpu = threading.Thread(target=do_tpu)
    
    with Live(provisioning_progress, refresh_per_second=4):
        t_gpu.start()
        t_tpu.start()
        while t_gpu.is_alive() or t_tpu.is_alive():
            time.sleep(0.2)

    # Check for failures
    if not provisioning_results["gpu"] or not provisioning_results["tpu"]:
        console.print("\n[bold red]Provisioning Failed. Aborting and Cleaning up...[/bold red]")
        cleanup()
        sys.exit(1)

    with console.status("[bold yellow]Waiting for SSH connectivity (Booting)..."):
        if not wait_for_ssh(GPU_VM, GPU_ZONE, is_tpu=False):
            console.print(f"[bold red]Error:[/bold red] {GPU_VM} failed to boot or is unreachable.")
            return
        if not wait_for_ssh(TPU_VM, TPU_ZONE, is_tpu=True):
            console.print(f"[bold red]Error:[/bold red] {TPU_VM} failed to boot or is unreachable.")
            return

    console.print("\n[bold green]✅ Infrastructure is Ready & SSH Reachable.[/bold green]")
    console.input("\n[dim]Press Enter to Sync Code...[/dim]")

    # --- SCENE 3: The Bridge ---
    with console.status("[bold green]Syncing files..."):
        if not LEGACY_PAYLOAD.exists() or not JAX_PAYLOAD.exists():
            console.print(f"[bold red]Error: Payload files not found at {PAYLOAD_DIR}[/bold red]")
            return

        res_gpu = subprocess.run(f"gcloud compute scp {LEGACY_PAYLOAD} {GPU_VM}:~/ --project={PROJECT} --zone={GPU_ZONE} --quiet", shell=True)
        if res_gpu.returncode != 0:
             console.print(f"[bold red]Error:[/bold red] Failed to SCP payload to {GPU_VM}.")
             return

        res_tpu = subprocess.run(f"gcloud compute tpus tpu-vm scp {JAX_PAYLOAD} {TPU_VM}:~/ --project={PROJECT} --zone={TPU_ZONE} --quiet", shell=True)
        if res_tpu.returncode != 0:
             console.print(f"[bold red]Error:[/bold red] Failed to SCP payload to {TPU_VM}.")
             return

    console.print("[green]✅ Codebase Deployed to Cloud Layers.[/green]")
    console.input("\n[dim]Press Enter to START THE RACE...[/dim]")

    # --- SCENE 4: The Race ---
    header("🏁 The Benchmark: ResNet-50 Training Race")
    gpu_log: list[str] = []
    tpu_log: list[str] = []
    progress = Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TextColumn("{task.fields[info]}"))
    gpu_task = progress.add_task("[green]ResNet-50 (A100)", total=300, info="Initializing...")
    tpu_task = progress.add_task("[blue]ResNet-50 (TPU)", total=300, info="Initializing...")

    def run_gpu() -> None:
        # Pre-installed on image: torch torchvision
        cmd = "python3 -u train_legacy.py"
        for line in run_ssh_command(GPU_VM, GPU_ZONE, cmd):
            if "Step" in line:
                try:
                    # Line format: Step 10/300 | Loss: ...
                    match = re.search(r"Step (\d+)/(\d+)", line)
                    if match:
                        step_num = int(match.group(1))
                        progress.update(gpu_task, completed=step_num, info=f"Step {step_num}/300")
                except Exception:
                    pass
            gpu_log.append(line)
        progress.update(gpu_task, completed=300, info="[bold green]DONE[/bold green]")

    def run_tpu() -> None:
        # Install JAX and Flax
        cmd = "pip3 install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax optax --quiet && python3 -u train_jax.py"
        for line in run_ssh_command(TPU_VM, TPU_ZONE, cmd, is_tpu=True):
            if "Step" in line:
                try:
                    match = re.search(r"Step (\d+)/(\d+)", line)
                    if match:
                        step_num = int(match.group(1))
                        progress.update(tpu_task, completed=step_num, info=f"Step {step_num}/300")
                except Exception:
                    pass
            tpu_log.append(line)
        progress.update(tpu_task, completed=300, info="[bold blue]DONE[/bold blue]")


    t1, t2 = threading.Thread(target=run_gpu), threading.Thread(target=run_tpu)
    t1.start()
    t2.start()

    with Live(create_dashboard("", "", progress), refresh_per_second=4) as live:
        while t1.is_alive() or t2.is_alive():
            live.update(create_dashboard("\n".join(gpu_log[-10:]), "\n".join(tpu_log[-10:]), progress))
            time.sleep(0.25)

    console.print("\n[bold green]✨ Benchmark Complete.[/bold green]")
    
    console.input("\n[bold red]Press Enter to DESTROY Cloud Resources...[/bold red]")
    cleanup()

def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by User.[/bold red]")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()