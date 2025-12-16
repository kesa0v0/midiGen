import pretty_midi
import music21
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 구조 분석기 (Structure Analyzer)
# ==========================================
class StructureAnalyzer:
    def __init__(self, pm, bpm, beats_per_bar=4):
        self.pm = pm
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.seconds_per_bar = (60 / bpm) * beats_per_bar

    def _get_bar_features(self):
        total_time = self.pm.get_end_time()
        total_bars = int(np.ceil(total_time / self.seconds_per_bar))
        
        chroma_list = []
        energy_list = []
        
        for b in range(total_bars):
            start = b * self.seconds_per_bar
            end = start + self.seconds_per_bar
            
            velocities = []
            pitch_classes = [0] * 12
            
            for instr in self.pm.instruments:
                for note in instr.notes:
                    if start <= note.start < end:
                        velocities.append(note.velocity)
                        if not instr.is_drum:
                            pitch_classes[note.pitch % 12] += 1
            
            # Chroma Normalization
            s = sum(pitch_classes)
            chroma_vec = np.array(pitch_classes) / s if s > 0 else np.zeros(12)
            chroma_list.append(chroma_vec)
            
            # Energy (Avg Velocity * Density)
            if velocities:
                avg_vel = np.mean(velocities) / 127.0
                density = len(velocities)
                energy = avg_vel * density
            else:
                energy = 0
            energy_list.append(energy)
            
        return np.array(chroma_list), np.array(energy_list)

    def analyze_structure(self):
        chromas, energies = self._get_bar_features()
        num_bars = len(chromas)
        if num_bars == 0: return []

        # Segmentation
        boundaries = [0]
        window_size = 4
        i = 0
        while i < num_bars - window_size:
            curr_win = chromas[i : i+window_size].flatten().reshape(1, -1)
            next_win = chromas[i+window_size : i+window_size*2].flatten().reshape(1, -1)
            
            if next_win.shape[1] != curr_win.shape[1]: break

            sim = cosine_similarity(curr_win, next_win)[0][0]
            last_boundary = boundaries[-1]
            segment_len = i + window_size - last_boundary
            
            if sim < 0.6 and segment_len >= 8:
                boundary_point = i + window_size
                boundaries.append(boundary_point)
                i += window_size
            else:
                i += 2

        boundaries.append(num_bars)
        boundaries = sorted(list(set(boundaries)))

        # Labeling
        sections = []
        max_energy = np.max(energies) if len(energies) > 0 else 1
        
        for k in range(len(boundaries)-1):
            start = boundaries[k]
            end = boundaries[k+1]
            length = end - start
            
            if length < 4 and len(sections) > 0: # Merge short sections
                sections[-1]['length'] += length
                continue

            seg_energy = np.mean(energies[start:end])
            norm_energy = seg_energy / max_energy if max_energy > 0 else 0
            
            label = "VERSE"
            if k == 0: label = "INTRO" if norm_energy < 0.6 else "VERSE"
            elif k == len(boundaries)-2: label = "OUTRO"
            else:
                if norm_energy > 0.8: label = "CHORUS"
                elif norm_energy > 0.6: label = "BRIDGE"
                else: label = "VERSE"

            sections.append({'type': label, 'start_bar': start, 'length': length})
        
        return sections

# ==========================================
# 2. 스마트 악기 분류기 (Smart Role Assigner)
# ==========================================
class SmartRoleAssigner:
    def __init__(self):
        self.keywords = {
            "DRUMS": ["drum", "perc", "kit", "bd", "snare", "hat", "timpani"],
            "BASS": ["bass", "sub", "808", "low", "tuba", "cello", "bassoon"],
            "MELODY": ["vocal", "vox", "lead", "melody", "solo", "main", "flute", "oboe", "violin", "trumpet"],
            "HARMONY": ["piano", "key", "str", "pad", "organ", "guit", "chord", "ensemble", "brass"]
        }

    def get_track_stats(self, instrument):
        if not instrument.notes: return None
        pitches = [n.pitch for n in instrument.notes]
        starts = sorted([n.start for n in instrument.notes])
        unique_starts = len(set(starts))
        polyphony = len(starts) / unique_starts if unique_starts > 0 else 1.0
        return {"avg_pitch": np.mean(pitches), "polyphony": polyphony}

    def assign_role(self, instrument):
        if instrument.is_drum: return "DRUMS", "STANDARD"
        
        prog = instrument.program
        stats = self.get_track_stats(instrument)
        if not stats: return "IGNORE", "NONE"

        # Orchestra Mapping (GM)
        if 40 <= prog <= 55: # Strings
            if prog in [42, 43]: return "BASS", "STRINGS"
            if stats['polyphony'] < 1.2 and stats['avg_pitch'] > 60: return "MELODY", "STRINGS"
            return "HARMONY", "STRINGS"
        if 56 <= prog <= 63: # Brass
            if prog == 58: return "BASS", "BRASS"
            if prog == 56 and stats['polyphony'] < 1.2: return "MELODY", "BRASS"
            return "HARMONY", "BRASS"
        if 64 <= prog <= 79: # Winds
            if prog in [67, 70]: return "BASS", "WIND"
            return "MELODY", "WIND"

        # Name & Stats Inference
        name = instrument.name.lower()
        for role, kws in self.keywords.items():
            for kw in kws:
                if kw in name:
                    spec = "SYNTH"
                    if "piano" in name: spec = "PIANO"
                    elif "str" in name: spec = "STRINGS"
                    elif "bass" in name: spec = "ELECTRIC"
                    return role, spec
        
        # Fallback based on stats
        if stats['avg_pitch'] < 48 and stats['polyphony'] < 1.2: return "BASS", "SUB"
        if stats['polyphony'] > 1.2: return "HARMONY", "PAD"
        if stats['avg_pitch'] >= 48 and stats['polyphony'] <= 1.2: return "MELODY", "SYNTH"
        return "HARMONY", "PIANO"

    def analyze_midi_roles(self, pm):
        summary = {"MELODY": "NONE", "HARMONY": "NONE", "BASS": "NONE", "DRUMS": "NONE"}
        candidates = {"MELODY": [], "BASS": [], "HARMONY": [], "DRUMS": []}
        
        for instr in pm.instruments:
            role, specific = self.assign_role(instr)
            if role != "IGNORE":
                candidates[role].append((instr, specific, len(instr.notes)))
        
        for role in summary.keys():
            if candidates[role]:
                best = sorted(candidates[role], key=lambda x: x[2], reverse=True)[0]
                summary[role] = best[1]
        return summary

# ==========================================
# 3. 메인 변환기 (Midi To Text Converter)
# ==========================================
class MidiToDSLConverter:
    def __init__(self, midi_path):
        self.pm = pretty_midi.PrettyMIDI(midi_path)
        
        # 0. 전처리: 피아노 솔로 분리
        self._preprocess_piano_solo()
        
        # 1. 박자표 및 템포 분석
        self.beats_per_bar, self.time_signature = self._analyze_time_signature()
        self.bpm = int(self.pm.get_tempo_changes()[1][0]) if len(self.pm.get_tempo_changes()[1]) > 0 else 120
        
        # 2. 글로벌 키 분석 (초기값)
        self.music21_score = music21.converter.parse(midi_path)
        self.global_key = self.music21_score.analyze('key')
        
    def _preprocess_piano_solo(self):
        """트랙이 1개인 피아노 곡을 왼손(Bass)/오른손(Melody)으로 분리"""
        melodic = [i for i in self.pm.instruments if not i.is_drum]
        if len(melodic) != 1: return
        solo = melodic[0]
        if not (0 <= solo.program <= 7): return
        
        pitches = [n.pitch for n in solo.notes]
        if not pitches: return
        split_point = int(np.mean(pitches))
        
        bass = pretty_midi.Instrument(program=solo.program, name="Piano Left")
        melody = pretty_midi.Instrument(program=solo.program, name="Piano Right")
        
        for n in solo.notes:
            if n.pitch < split_point: bass.notes.append(n)
            else: melody.notes.append(n)
            
        self.pm.instruments.remove(solo)
        self.pm.instruments.append(bass)
        self.pm.instruments.append(melody)

    def _analyze_time_signature(self):
        if not self.pm.time_signature_changes: return 4, "4/4"
        ts = self.pm.time_signature_changes[0]
        return ts.numerator, f"{ts.numerator}/{ts.denominator}"

    def _detect_local_key(self, start, end):
        """특정 구간의 Local Key 감지"""
        stream = music21.stream.Stream()
        notes = []
        for instr in self.pm.instruments:
            if instr.is_drum: continue
            for n in instr.notes:
                if n.start >= start and n.end <= end:
                    notes.append(n.pitch)
        
        if len(notes) < 10: return None
        for p in notes: stream.append(music21.note.Note(p))
        try: return stream.analyze('key')
        except: return None

    def _analyze_chord_progression(self, start, end, current_key):
        """Grid 단위 코드 분석"""
        notes = []
        for instr in self.pm.instruments:
            if instr.is_drum: continue
            for n in instr.notes:
                if (n.start < end) and (n.end > start):
                    notes.append(n.pitch)
        
        if not notes: return None
        try:
            c = music21.chord.Chord(list(set(notes)))
            rn = music21.roman.romanNumeralFromChord(c, current_key)
            return rn.romanNumeral # ex: V, iv
        except: return "N.C."

    def _calculate_controls(self, notes, duration):
        if not notes: return "MID", "NORMAL", "STATIC"
        
        # Dynamic
        avg_vel = np.mean([n.velocity for n in notes])
        if avg_vel < 60: dyn = "LOW"
        elif avg_vel > 100: dyn = "HIGH"
        else: dyn = "MID"
        
        # Density
        den_val = len(notes) / duration
        if den_val < 3: den = "SPARSE"
        elif den_val > 10: den = "DENSE"
        else: den = "NORMAL"
        
        # Movement
        start_p = notes[0].pitch
        end_p = notes[-1].pitch
        if end_p > start_p + 5: mov = "ASCENDING"
        elif end_p < start_p - 5: mov = "DESCENDING"
        else: mov = "STATIC"
        
        return dyn, den, mov

    def convert(self):
        output_lines = []
        
        # [GLOBAL]
        key_str = f"{self.global_key.tonic.name}_{self.global_key.mode.upper()}".replace("-", "b").replace("#", "s")
        output_lines.append("[GLOBAL]")
        output_lines.append(f"[BPM:{self.bpm}]")
        output_lines.append(f"[KEY:{key_str}]")
        output_lines.append(f"[TIME_SIG:{self.time_signature}]")
        output_lines.append("")
        
        # [INSTRUMENTS]
        role_assigner = SmartRoleAssigner()
        roles_summary = role_assigner.analyze_midi_roles(self.pm)
        output_lines.append("[INSTRUMENTS]")
        for k, v in roles_summary.items():
             output_lines.append(f"  {k}: {v}")
        output_lines.append("")

        # [SECTION & PROG]
        analyzer = StructureAnalyzer(self.pm, self.bpm, self.beats_per_bar)
        detected_sections = analyzer.analyze_structure()
        
        seconds_per_beat = 60 / self.bpm
        seconds_per_bar = seconds_per_beat * self.beats_per_bar
        
        current_active_key = self.global_key # 전조 추적용 변수

        for sec in detected_sections:
            # 1. Local Key Detection (Modulation check)
            sec_start_time = sec['start_bar'] * seconds_per_bar
            sec_end_time = (sec['start_bar'] + sec['length']) * seconds_per_bar
            local_key = self._detect_local_key(sec_start_time, sec_end_time)
            
            if local_key:
                current_active_key = local_key
            
            output_lines.append(f"[SECTION:{sec['type']}]")
            output_lines.append(f"[BARS:{sec['length']}]")
            
            prog_bars_list = []
            all_section_notes = []

            # Bar Loop
            for b in range(sec['length']):
                bar_abs_idx = sec['start_bar'] + b
                bar_start_time = bar_abs_idx * seconds_per_bar
                
                bar_tokens = []
                last_chord_in_bar = "N.C."
                
                # Beat Loop (Grid)
                for beat in range(self.beats_per_bar):
                    beat_start = bar_start_time + (beat * seconds_per_beat)
                    beat_end = beat_start + seconds_per_beat
                    
                    # 박자 단위 분석 (현재 Active Key 기준)
                    detected_chord = self._analyze_chord_progression(beat_start, beat_end, current_active_key)
                    
                    token = "N.C."
                    if detected_chord is None:
                        # 노트 없음 -> 지속(-) or N.C.
                        token = "-" if beat > 0 else "N.C."
                    else:
                        if beat == 0:
                            token = detected_chord
                        else:
                            if detected_chord == last_chord_in_bar:
                                token = "-"
                            else:
                                token = detected_chord
                    
                    if token != "-":
                        last_chord_in_bar = token
                    
                    bar_tokens.append(token)
                    
                    # Control용 노트 수집
                    for instr in self.pm.instruments:
                        if instr.is_drum: continue
                        for n in instr.notes:
                             if (n.start < beat_end) and (n.end > beat_start):
                                all_section_notes.append(n)
                
                prog_bars_list.append(" ".join(bar_tokens))

            # PROG Formatting
            formatted_prog = []
            for i in range(0, len(prog_bars_list), 4):
                chunk = prog_bars_list[i:i+4]
                formatted_prog.append(" | ".join(chunk))
            
            output_lines.append(f"[PROG: {' | '.join(formatted_prog)}]")
            
            # Control Tokens
            dyn, den, mov = self._calculate_controls(all_section_notes, seconds_per_bar * sec['length'])
            output_lines.append(f"[DYNAMIC:{dyn}]")
            output_lines.append(f"[DENSITY:{den}]")
            output_lines.append(f"[MOVEMENT:{mov}]")
            output_lines.append("") 
            
        return "\n".join(output_lines)

# --- 실행 예시 ---
if __name__ == "__main__":
    # 사용 예시
    converter = MidiToDSLConverter("sample_song.mid")
    print(converter.convert())