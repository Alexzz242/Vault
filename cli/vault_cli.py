"""
vault_cli.py — Command-line interface for Vault.

Usage:
  vault index /path/to/folder
  vault search "what did I write about X?"
  vault status
  vault clear
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import print as rprint

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.group()
def vault():
    """🔍 Vault — private AI search over your local files."""
    pass


@vault.command()
@click.argument("path", type=click.Path(exists=True))
def index(path):
    """Index a folder so Vault can search it."""
    from vault_core.indexer import index_directory
    from vault_core.embedder import check_ollama

    if not check_ollama():
        console.print("[red]❌ Ollama is not running. Start it with: ollama serve[/red]")
        sys.exit(1)

    folder = Path(path).resolve()
    console.print(f"\n[bold cyan]📂 Indexing:[/bold cyan] {folder}\n")

    result = index_directory(folder, verbose=True)

    console.print(Panel(
        f"[green]✅ Done![/green]\n"
        f"Files indexed: [bold]{result['files_indexed']}[/bold] / {result['total_found']}\n"
        f"Chunks stored: [bold]{result['chunks_total']}[/bold]\n"
        f"Errors: [bold red]{result['errors']}[/bold red]",
        title="Indexing Complete",
    ))


@vault.command()
@click.argument("question")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
def search(question, no_stream):
    """Ask a question about your files."""
    from vault_core.embedder import check_ollama

    if not check_ollama():
        console.print("[red]❌ Ollama is not running. Start it with: ollama serve[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]🔍 Searching for:[/bold cyan] {question}\n")

    if no_stream:
        from vault_core.llm import ask
        result = ask(question)
        console.print(Panel(result["answer"], title="Answer"))
        if result["sources"]:
            console.print(f"[dim]Sources: {', '.join(result['sources'])}[/dim]")
    else:
        from vault_core.llm import ask_stream
        console.print("[bold]Answer:[/bold] ", end="")
        for token in ask_stream(question):
            print(token, end="", flush=True)
        print("\n")


@vault.command()
def status():
    """Show how many files and chunks are indexed."""
    from vault_core.vector_store import count, get_collection
    from vault_core.embedder import check_ollama

    ollama_ok = check_ollama()
    total_chunks = count()

    table = Table(title="Vault Status", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Ollama", "✅ Running" if ollama_ok else "❌ Not running")
    table.add_row("Indexed chunks", str(total_chunks))

    console.print(table)


@vault.command()
@click.confirmation_option(prompt="⚠️  This will delete all indexed data. Are you sure?")
def clear():
    """Wipe the entire index."""
    from vault_core.vector_store import clear_all
    clear_all()
    console.print("[green]✅ Index cleared.[/green]")


if __name__ == "__main__":
    vault()
