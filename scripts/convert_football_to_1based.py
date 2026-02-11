from pathlib import Path
import shutil

p = Path("football.txt")
if not p.exists():
    raise FileNotFoundError("football.txt not found in current directory")

bak = p.with_suffix(".txt.bak")
shutil.copy2(p, bak)

out_lines = []
with p.open("r", encoding="utf-8", errors="replace") as f:
    # read header
    header = f.readline()
    if not header:
        raise ValueError("Empty file")
    out_lines.append(header.rstrip("\n"))
    # read size line (may be comments/blank before it)
    size_line = ""
    while True:
        line = f.readline()
        if line is None or line == "":
            raise ValueError("Unexpected EOF while reading size line")
        if line.strip() == "" or line.lstrip().startswith("%"):
            out_lines.append(line.rstrip("\n"))
            continue
        size_line = line.rstrip("\n")
        out_lines.append(size_line)
        break

    # process remaining lines
    for line in f:
        raw = line.rstrip("\n")
        s = raw.strip()
        if s == "" or s.startswith("%"):
            out_lines.append(raw)
            continue
        parts = s.split()
        # require at least two indices; otherwise keep line as-is
        if len(parts) < 2:
            out_lines.append(raw)
            continue
        try:
            r = int(parts[0])
            c = int(parts[1])
        except ValueError:
            out_lines.append(raw)
            continue
        # convert to 1-based if currently 0-based (i.e., any index == 0)
        # We'll be conservative: if any index == 0, add 1 to both r and c
        if r == 0 or c == 0:
            r += 1
            c += 1
        # otherwise, if minimal index > 0, assume already 1-based; keep as-is
        rest = " ".join(parts[2:]) if len(parts) > 2 else ""
        if rest:
            out_lines.append(f"{r} {c} {rest}")
        else:
            out_lines.append(f"{r} {c}")

# write back
with p.open("w", encoding="utf-8", newline="\n") as f:
    for ln in out_lines:
        f.write(ln + "\n")

print(f"Converted football.txt to 1-based indices. Backup saved as: {bak}")