from pathlib import Path

# === Hauptordner (hier anpassen!) ===
root_dir = Path("training/alternativ_daten")

# === Alle .txt-Dateien rekursiv löschen ===
deleted = 0
for txt_file in root_dir.rglob("*.txt"):
    try:
        txt_file.unlink()
        print(f"🗑️  Gelöscht: {txt_file}")
        deleted += 1
    except Exception as e:
        print(f"  Fehler beim Löschen von {txt_file}: {e}")

print(f"\n Fertig – {deleted} .txt-Dateien gelöscht.")
