"""Point d entree historique. Depuis la racine : python -m tests.libraryTest --output runs/essai --quick."""
from signlab.cli import main

if __name__ == "__main__":
    import sys
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
