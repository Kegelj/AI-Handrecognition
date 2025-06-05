from pathlib import Path

# 📂 Ordner mit allen Daten
base_dir = Path("training/alle_daten")

# 🔁 Alle txt-Dateien suchen
for txt_file in base_dir.rglob("*.txt"):
    # 🔎 Prüfe, ob ein Bild mit demselben Namen existiert
    stem = txt_file.stem
    folder = txt_file.parent

    jpg_exists = (folder / f"{stem}.jpg").exists()
    png_exists = (folder / f"{stem}.png").exists()

    if not (jpg_exists or png_exists):
        print(f"🗑️ Lösche: {txt_file}")
        txt_file.unlink()  # Löschen

print("✅ Alle überflüssigen .txt-Dateien wurden gelöscht.")
