"""
CathodeScreen CLI — predict cathode stability from the command line.

Usage:
    cathode-screen predict structure.cif
    cathode-screen predict *.cif --output results.csv
    cathode-screen predict dir/ --batch --output results.csv
    cathode-screen info
    cathode-screen health
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from cathode_screen.client import CathodeClient, PredictionResult

console = Console()


def _resolve_cif_paths(paths: tuple) -> List[Path]:
    """Expand directories and glob patterns to CIF file paths."""
    result: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            result.extend(sorted(path.glob("*.cif")))
        elif path.exists():
            result.append(path)
        else:
            # Try glob
            parent = path.parent if path.parent.exists() else Path(".")
            matches = sorted(parent.glob(path.name))
            if matches:
                result.extend(matches)
            else:
                console.print(f"[red]Not found:[/red] {p}")
    return result


def _print_result(result: PredictionResult) -> None:
    """Pretty-print a single prediction result."""
    # Decision badge
    color = {"DFT": "green", "KEEP": "green", "HOLD": "yellow", "MAYBE": "yellow"}.get(
        result.decision, "red"
    )

    console.print()
    console.print(f"[bold]{result.material_id}[/bold]")
    console.print(f"  Decision:    [{color} bold]{result.decision}[/{color} bold]")
    console.print(f"  E_hull:      {result.ehull:.4f} eV/atom")
    console.print(f"  P(stable):   {result.p_stable:.1%}")
    console.print(f"  Uncertainty: {result.uncertainty}")
    console.print(
        f"  90% CI:      [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}] eV"
    )

    if result.properties is not None:
        props = result.properties
        console.print()
        console.print("  [bold]Cathode Properties[/bold]")
        if props.voltage is not None:
            conf = f" ({props.voltage_confidence})" if props.voltage_confidence else ""
            console.print(f"  Voltage:     {props.voltage:.2f} V vs Li/Li+{conf}")
        if props.gravimetric_capacity is not None:
            console.print(f"  Capacity:    {props.gravimetric_capacity:.0f} mAh/g")
        if props.gravimetric_energy is not None:
            console.print(f"  Energy:      {props.gravimetric_energy:.0f} Wh/kg")
        if props.density is not None:
            console.print(f"  Density:     {props.density:.2f} g/cm3")
        if props.tm_elements:
            console.print(f"  TM elements: {', '.join(props.tm_elements)}")
        if props.anion_framework:
            console.print(f"  Anion:       {props.anion_framework}")
        if props.composite_score is not None:
            score_pct = props.composite_score * 100
            score_color = "green" if score_pct > 70 else "yellow" if score_pct > 40 else "red"
            console.print(f"  Score:       [{score_color}]{score_pct:.0f}%[/{score_color}]")


def _results_to_csv(results: List[PredictionResult], output: Path) -> None:
    """Write results to CSV."""
    fieldnames = [
        "material_id", "decision", "ehull_eV", "p_stable", "uncertainty",
        "ci_lower", "ci_upper",
        "voltage_V", "capacity_mAhg", "energy_Whkg", "density_gcm3",
        "tm_elements", "anion_framework", "composite_score",
    ]

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "material_id": r.material_id,
                "decision": r.decision,
                "ehull_eV": f"{r.ehull:.4f}",
                "p_stable": f"{r.p_stable:.3f}",
                "uncertainty": r.uncertainty,
                "ci_lower": f"{r.confidence_interval[0]:.4f}",
                "ci_upper": f"{r.confidence_interval[1]:.4f}",
            }
            if r.properties:
                row["voltage_V"] = f"{r.properties.voltage:.2f}" if r.properties.voltage else ""
                row["capacity_mAhg"] = (
                    f"{r.properties.gravimetric_capacity:.0f}"
                    if r.properties.gravimetric_capacity else ""
                )
                row["energy_Whkg"] = (
                    f"{r.properties.gravimetric_energy:.0f}"
                    if r.properties.gravimetric_energy else ""
                )
                row["density_gcm3"] = (
                    f"{r.properties.density:.2f}" if r.properties.density else ""
                )
                row["tm_elements"] = (
                    ",".join(r.properties.tm_elements) if r.properties.tm_elements else ""
                )
                row["anion_framework"] = r.properties.anion_framework or ""
                row["composite_score"] = (
                    f"{r.properties.composite_score:.3f}"
                    if r.properties.composite_score is not None else ""
                )
            writer.writerow(row)

    console.print(f"\n[green]Results saved to {output}[/green]")


@click.group()
@click.option("--api-key", envvar="CATHODE_API_KEY", default=None, help="API key")
@click.option("--api-url", envvar="CATHODE_API_URL", default=None, help="API base URL")
@click.pass_context
def main(ctx, api_key: Optional[str], api_url: Optional[str]):
    """CathodeScreen CLI — AI-powered cathode material screening."""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["api_url"] = api_url


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output CSV file")
@click.option("--batch/--no-batch", default=False, help="Use batch endpoint")
@click.pass_context
def predict(ctx, paths: tuple, output: Optional[str], batch: bool):
    """Predict stability for CIF files.

    Examples:
        cathode-screen predict structure.cif
        cathode-screen predict *.cif -o results.csv
        cathode-screen predict data/structures/ --batch -o results.csv
    """
    cif_paths = _resolve_cif_paths(paths)
    if not cif_paths:
        console.print("[red]No CIF files found.[/red]")
        sys.exit(1)

    console.print(f"[bold]Screening {len(cif_paths)} structure(s)...[/bold]")

    client = CathodeClient(
        api_key=ctx.obj.get("api_key"),
        base_url=ctx.obj.get("api_url"),
    )

    results: List[PredictionResult] = []

    if batch and len(cif_paths) > 1:
        with console.status("Running batch inference..."):
            results = client.predict_batch(cif_paths)
    else:
        for i, cif_path in enumerate(cif_paths):
            with console.status(f"[{i+1}/{len(cif_paths)}] {cif_path.name}..."):
                try:
                    result = client.predict(cif_path)
                    results.append(result)
                except Exception as e:
                    console.print(f"[red]Error: {cif_path.name}: {e}[/red]")

    if not results:
        console.print("[red]No predictions returned.[/red]")
        sys.exit(1)

    # Display results
    if output:
        _results_to_csv(results, Path(output))
    elif len(results) == 1:
        _print_result(results[0])
    else:
        # Table format for multiple results
        table = Table(title=f"CathodeScreen Results ({len(results)} structures)")
        table.add_column("Material", style="bold")
        table.add_column("Decision", justify="center")
        table.add_column("E_hull (eV)", justify="right")
        table.add_column("P(stable)", justify="right")
        table.add_column("Voltage (V)", justify="right")
        table.add_column("Capacity", justify="right")
        table.add_column("Score", justify="right")

        for r in results:
            color = {"DFT": "green", "KEEP": "green", "HOLD": "yellow", "MAYBE": "yellow"}.get(
                r.decision, "red"
            )
            voltage = f"{r.voltage:.2f}" if r.voltage else "-"
            capacity = f"{r.capacity:.0f}" if r.capacity else "-"
            score = f"{r.composite_score:.0%}" if r.composite_score else "-"

            table.add_row(
                r.material_id,
                f"[{color}]{r.decision}[/{color}]",
                f"{r.ehull:.4f}",
                f"{r.p_stable:.1%}",
                voltage,
                capacity,
                score,
            )

        console.print(table)

    # Summary
    n_keep = sum(1 for r in results if r.decision in ("DFT", "KEEP"))
    n_maybe = sum(1 for r in results if r.decision in ("HOLD", "MAYBE"))
    n_kill = sum(1 for r in results if r.decision in ("SKIP", "KILL"))
    console.print(
        f"\n[green]{n_keep} KEEP[/green] / "
        f"[yellow]{n_maybe} MAYBE[/yellow] / "
        f"[red]{n_kill} KILL[/red]"
    )

    client.close()


@main.command()
@click.pass_context
def info(ctx):
    """Show model information."""
    client = CathodeClient(
        api_key=ctx.obj.get("api_key"),
        base_url=ctx.obj.get("api_url"),
    )
    try:
        data = client.model_info()
        console.print("[bold]CathodeScreen Model Info[/bold]")
        for key, value in data.items():
            if key != "artifact_manifest":
                console.print(f"  {key}: {value}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        client.close()


@main.command()
@click.pass_context
def health(ctx):
    """Check API health."""
    client = CathodeClient(
        api_key=ctx.obj.get("api_key"),
        base_url=ctx.obj.get("api_url"),
    )
    ok = client.health()
    if ok:
        console.print("[green]API is healthy[/green]")
    else:
        console.print("[red]API is unreachable[/red]")
        sys.exit(1)
    client.close()


if __name__ == "__main__":
    main()
