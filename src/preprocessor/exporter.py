class DatasetExporter:
    def export(
        self,
        analysis,
        instruments,
        sections,
        output_path
    ):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[GLOBAL]\n")
            f.write(f"BPM={analysis['global_bpm']}\n")
            f.write(f"TIME_SIG={analysis['time_sig'][0]}/{analysis['time_sig'][1]}\n\n")

            f.write("[INSTRUMENTS]\n")
            for k, v in instruments.items():
                f.write(f"{k}={v}\n")

            for sec in sections:
                f.write(f"\n[SECTION:{sec.name}]\n")
                f.write(f"BARS={sec.bars}\n")
                f.write("PROG=\n")
                for bar in sec.prog_grid:
                    f.write("| " + " ".join(bar) + " |\n")
                f.write(
                    "CTRL=" + " ".join(
                        f"{k}:{v}" for k, v in sec.control_tokens.items()
                    ) + "\n"
                )
