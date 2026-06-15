import fire
from src.cli import CLI


def main() -> None:
    """Launch the RAG CLI."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
