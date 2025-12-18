class DatasetExporter:
    def export(
        self,
        conductor_bundle,
        instruments,
        output_path
    ):
        global_meta = conductor_bundle["global"]
        form = conductor_bundle["form"]
        sections = conductor_bundle["sections"]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[GLOBAL]\n")
            f.write(f"BPM={global_meta['bpm']}\n")
            f.write(f"TIME_SIG={global_meta['time_sig'][0]}/{global_meta['time_sig'][1]}\n")
            f.write(f"GRID_UNIT={global_meta['grid_unit']}\n")
            if global_meta.get("key"):
                f.write(f"KEY={global_meta['key']}\n")
            f.write("\n")

            f.write("[INSTRUMENTS]\n")
            for k, v in instruments.items():
                f.write(f"{k}={v}\n")

            f.write("\n[FORM]\n")
            f.write(" > ".join(form) + "\n")

            for sec in sections:
                f.write(f"\n[SECTION:{sec.name}]\n")
                f.write(f"BARS={sec.bars}\n")
                if sec.bpm is not None and sec.bpm != global_meta["bpm"]:
                    f.write(f"BPM={sec.bpm}\n")
                if sec.time_sig is not None and (sec.time_sig.numerator, sec.time_sig.denominator) != tuple(global_meta["time_sig"]):
                    f.write(f"TIME_SIG={sec.time_sig.numerator}/{sec.time_sig.denominator}\n")
                if sec.key is not None and sec.key != global_meta.get("key"):
                    f.write(f"KEY={sec.key}\n")

                f.write("PROG=\n")
                for bar in sec.prog_grid:
                    f.write("| " + " ".join(bar) + " |\n")
                f.write(
                    "CTRL=" + " ".join(
                        f"{k}:{v}" for k, v in sec.control_tokens.items()
                    ) + "\n"
                )
