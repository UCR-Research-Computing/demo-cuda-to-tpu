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
LEGACY_PAYLOAD = PAYLOAD_DIR / "nbody_legacy.py"
JAX_PAYLOAD = PAYLOAD_DIR / "nbody_jax.py"

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

def run_local_command(cmd: str) -> None:
    """Runs a local command and prints output."""


def run_ssh_command(vm: str, zone: str, command: str) -> Generator[str, None, None]:
    """Executes SSH command and yields output line by line."""
    full_cmd = [
        "gcloud", "compute", "ssh", vm,
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

def main() -> None:
    if "--version" in sys.argv:
        print("ursa-major-demo-cuda-to-tpu v0.1.2")
        return

    header("🚀 CUDA to TPU: The Accelerator Race")

    # --- SCENE 1: The Context ---
    step("Context: The 'Handwritten Kernel' Problem", 
    """
    We start with a Python script ([cyan]nbody_legacy.py[/cyan]) using [bold red]Numba CUDA[/bold red].
    It's performant but hardware-locked.
    """)

    # Simulate AI Analysis
    with console.status("[bold magenta]Gemini is analyzing the codebase..."):
        time.sleep(2.5)  # Dramatic pause
    console.print("[bold magenta]✨ Analysis Complete: Identified Parallelizable Kernels.[/bold magenta]")

    step("The AI Solution", 
    """
    Gemini CLI refactored the CUDA logic into high-level [bold blue]JAX[/bold blue].
    We now have [cyan]nbody_jax.py[/cyan] ready for TPU execution.
    """)

    # --- SCENE 2: Concurrent Provisioning ---
    gpu_create_cmd = f"gcloud compute instances create {GPU_VM} --project={PROJECT} --zone={GPU_ZONE} --machine-type=a2-highgpu-1g --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --quiet"
    tpu_create_cmd = f"gcloud compute tpus tpu-vm create {TPU_VM} --project={PROJECT} --zone={TPU_ZONE} --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite --quiet"

    step("Provisioning the Iron", 
    """
    Simultaneously summoning [bold green]Nvidia A100[/bold green] and [bold blue]Google TPU v5e[/bold blue]...
    
    This demonstrates the power of the 'Execution Engine' to orchestrate multiple architectures at once.
    """, command_preview=f"{gpu_create_cmd}\n{tpu_create_cmd}")

    provisioning_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[status]}")
    )
    
    gpu_p_task = provisioning_progress.add_task("[green]Nvidia A100 VM", total=100, status="Starting...")
    tpu_p_task = provisioning_progress.add_task("[blue]Google TPU VM", total=100, status="Starting...")

    def do_gpu() -> None:
        # Check if exists
        check = subprocess.run(f"gcloud compute instances describe {GPU_VM} --project={PROJECT} --zone={GPU_ZONE}", shell=True, capture_output=True)
        if check.returncode != 0:
            provisioning_progress.update(gpu_p_task, status="Creating...")
            subprocess.run(gpu_create_cmd, shell=True, capture_output=True)
        provisioning_progress.update(gpu_p_task, completed=100, status="[bold green]ONLINE[/bold green]")

    def do_tpu() -> None:
        # Check if exists
        check = subprocess.run(f"gcloud compute tpus tpu-vm describe {TPU_VM} --project={PROJECT} --zone={TPU_ZONE}", shell=True, capture_output=True)
        if check.returncode != 0:
            provisioning_progress.update(tpu_p_task, status="Creating...")
            subprocess.run(tpu_create_cmd, shell=True, capture_output=True)
        provisioning_progress.update(tpu_p_task, completed=100, status="[bold green]ONLINE[/bold green]")

    t_gpu = threading.Thread(target=do_gpu)
    t_tpu = threading.Thread(target=do_tpu)
    
    with Live(provisioning_progress, refresh_per_second=4):
        t_gpu.start()
        t_tpu.start()
        while t_gpu.is_alive() or t_tpu.is_alive():
            time.sleep(0.2)

    with console.status("[bold yellow]Waiting for SSH connectivity (Booting)..."):
        if not wait_for_ssh(GPU_VM, GPU_ZONE, is_tpu=False):
            console.print(f"[bold red]Error:[/bold red] {GPU_VM} failed to boot.")
            return
        if not wait_for_ssh(TPU_VM, TPU_ZONE, is_tpu=True):
            console.print(f"[bold red]Error:[/bold red] {TPU_VM} failed to boot.")
            return

    console.print("\n[bold green]✅ Infrastructure is Ready & SSH Reachable.[/bold green]")
    console.input("\n[dim]Press Enter to Sync Code...[/dim]")

    # --- SCENE 3: The Bridge ---
    with console.status("[bold green]Syncing files..."):
        if not LEGACY_PAYLOAD.exists() or not JAX_PAYLOAD.exists():
            console.print(f"[bold red]Error: Payload files not found at {PAYLOAD_DIR}[/bold red]")
            return

        subprocess.run(f"gcloud compute scp {LEGACY_PAYLOAD} {GPU_VM}:~/ --project={PROJECT} --zone={GPU_ZONE} --quiet", shell=True)
        subprocess.run(f"gcloud compute scp {JAX_PAYLOAD} {TPU_VM}:~/ --project={PROJECT} --zone={TPU_ZONE} --quiet", shell=True)
    console.print("[green]✅ Codebase Deployed to Cloud Layers.[/green]")
    console.input("\n[dim]Press Enter to START THE RACE...[/dim]")

    # --- SCENE 4: The Race ---
    header("🏁 The Benchmark: Legacy CUDA vs Pure JAX/TPU")
    gpu_log: list[str] = []
    tpu_log: list[str] = []
    progress = Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TextColumn("{task.fields[info]}"))
    gpu_task = progress.add_task("[green]Legacy CUDA Simulation", total=10000, info="Initializing...")
    tpu_task = progress.add_task("[blue]Pure JAX Simulation", total=10000, info="Initializing...")

    def run_gpu() -> None:
        # Ensure numba is installed
        cmd = "pip3 install numba --quiet && python3 nbody_legacy.py --n 16384 --iter 10000"
        for line in run_ssh_command(GPU_VM, GPU_ZONE, cmd):
            if "Iteration" in line:
                try:
                    match = re.search(r"Iteration (\d+)", line)
                    if match:
                        it = int(match.group(1))
                        progress.update(gpu_task, completed=it, info=f"Step {it}")
                except Exception:
                    pass
            gpu_log.append(line)
        progress.update(gpu_task, completed=10000, info="[bold green]DONE[/bold green]")

    def run_tpu() -> None:
        cmd = "python3 nbody_jax.py --n 16384 --iter 10000"
        for line in run_ssh_command(TPU_VM, TPU_ZONE, cmd):
            if "Iteration" in line:
                try:
                    match = re.search(r"Iteration (\d+)", line)
                    if match:
                        it = int(match.group(1))
                        progress.update(tpu_task, completed=it, info=f"Step {it}")
                except Exception:
                    pass
            tpu_log.append(line)
        progress.update(tpu_task, completed=10000, info="[bold blue]DONE[/bold blue]")


    t1, t2 = threading.Thread(target=run_gpu), threading.Thread(target=run_tpu)
    t1.start()
    t2.start()

    with Live(create_dashboard("", "", progress), refresh_per_second=4) as live:
        while t1.is_alive() or t2.is_alive():
            live.update(create_dashboard("\n".join(gpu_log[-10:]), "\n".join(tpu_log[-10:]), progress))
            time.sleep(0.25)

    console.print("\n[bold green]✨ Benchmark Complete.[/bold green]")

if __name__ == "__main__":
    main()
