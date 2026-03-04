from pathlib import Path

# chemin vers ton code legacy
base_path = Path("./legacy_code")

# extensions à traiter
extensions = [".py", ".js"]

# parcourir tous les fichiers
for file_path in base_path.rglob("*"):
    if file_path.suffix in extensions:
        try:
            # lecture et remplacement des caractères invalides
            text = file_path.read_text(encoding="utf-8", errors="replace")

            # écriture du texte "nettoyé" dans le même fichier
            file_path.write_text(text, encoding="utf-8")

        except Exception as e:
            print(f"[ERROR] Impossible de traiter {file_path}: {e}")

print("[INFO] Tous les fichiers traités avec remplacement des caractères invalides.")
